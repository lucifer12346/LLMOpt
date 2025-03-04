import argparse
import os
import math
import time
import torch
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.load_sft_dataset import SFTHintsDataset
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

'''
Training LLM using Huggingface Accelerate + Deepspeed.
'''

def parse_option():
    parser = argparse.ArgumentParser()
    
    # global args
    parser.add_argument('--per_device_train_batch_size', type = int, default = 4,
                        help = 'batch size per gpu device.')
    parser.add_argument('--block_size', type = int, default = 2048,
                        help = 'block size, i.e., the length of training sequences.')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pretrained_model_name_or_path', type = str, default = "deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--lr', type = float, default = 5e-5, help = "5e-5 for pre-training, 5e-6 for fine-tuning.")
    parser.add_argument('--checkpointing_steps', type = int, default = 300)
    parser.add_argument('--tensorboard_log_dir', type = str, default = "./train_logs")
    parser.add_argument('--output_ckpt_dir', type = str, default = "./ckpts")
    # parser.add_argument("--t")
    # args for supervised fine-tuning
    parser.add_argument('--model_type', type = str, default = "select")
    parser.add_argument('--training_data_dir', type = str, default = "train_20240127.json")
    parser.add_argument('--without_id', type = int, default = -1)
    parser.add_argument('--constant_lr', type = int, default = 0)
    opt = parser.parse_args()

    return opt



def sanity_check(input, target, tokenizer):
    print("Start Sanity Check -------->")
    for t, m in zip(input, target):
        decoded = tokenizer.decode([t])
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<-------- End Sanity Check")

    assert len(input) == len(target), f"length mismatch: {len(input)} vs {len(target)}"

def checkpoint_model(accelerator, model, tokenizer, output_ckpt_dir, last_global_step):    
    '''
    Utility fuction for only checkpointing the model dictionary (i.e., only model parameters)
    '''
    ckpt_path = os.path.join(output_ckpt_dir, "ckpt-{}".format(last_global_step))
    accelerator.print("checkpointing model state dict at {}".format(ckpt_path))
    unwrapped_model = accelerator.unwrap_model(model)
    # TODO: currently, there is a small bug that saves a full checkpoint data for each shard when enable zero1 and 2. 
    # See https://github.com/microsoft/DeepSpeed/issues/3303. solution: waiting upgrade of accelerate and deepspeed
    unwrapped_model.save_pretrained(
        ckpt_path, 
        is_main_process = accelerator.is_main_process, 
        save_function = accelerator.save,
        state_dict = accelerator.get_state_dict(model),
        max_shard_size = "100GB"
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_path)
    
    return

def train(opt):
    set_seed(opt.seed)
    stats = [1,1,1,1,1]
    if opt.without_id < 5 and opt.without_id > -1:
        stats[opt.without_id] = 0
    elif opt.without_id == 6:
        stats = [0,0,0,0,0]
    writer = SummaryWriter(opt.tensorboard_log_dir)
    accelerator = Accelerator()
    print("accelerator.is_main_process:", accelerator.is_main_process)
    print("accelerator.device:", accelerator.device)
    print("stats: ", stats)
    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    
    accelerator.print(opt)
    accelerator.print("tokens per batch:", total_batch_size * opt.block_size)
    accelerator.print("sequences per batch:", total_batch_size)
    accelerator.print("using LLM from:", opt.pretrained_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(opt.pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True) # attn_implementation="flash_attention_2"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # enable gradient checkpointing to save GPU memory, but this action would slowdown the training speed 20-30%
    model.gradient_checkpointing_enable()

    dataset = SFTHintsDataset(opt.training_data_dir, tokenizer, opt.block_size, "train", opt.model_type, stats)
    print(f"dataset length: {len(dataset)}")
    if accelerator.is_main_process:
        sanity_check(dataset[0]["input_ids"], dataset[0]["labels"], tokenizer)
    dataloader = DataLoader(dataset, batch_size = opt.per_device_train_batch_size, shuffle = True, drop_last = True)
    num_total_batches = math.ceil(opt.epochs * math.ceil(len(dataset) / total_batch_size)) # number of total batches
    optimizer = AdamW(model.parameters(), lr = opt.lr, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 0.1)

    num_warm_up_batches = int(num_total_batches * 0.05) # 5% of total batches for warm up
    
    if opt.constant_lr == 1:
        from torch.optim.lr_scheduler import ConstantLR
        lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
    else:
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer = optimizer, 
            warmup_epochs = num_warm_up_batches * accelerator.num_processes, # * accelerator.num_processes
            max_epochs = num_total_batches * accelerator.num_processes,   # * accelerator.num_processes
            warmup_start_lr = 0.0, 
            eta_min = 0.1 * opt.lr
        )
    print("before prepare")
    # optimizer, dataloader, lr_scheduler = accelerator.prepare(optimizer, dataloader, lr_scheduler)
    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(optimizer, model, dataloader, lr_scheduler)
    print("after prepare")
    print(type(optimizer))
    print(type(model))
    print(type(dataloader))
    print(type(lr_scheduler))

    accumulation_loss = 0
    global_completed_steps = 0
    model.train()
    st = time.time()
    
    
    for epoch in range(opt.epochs):
        print("This is epoch:", epoch+1)
        for batch in tqdm.tqdm(dataloader):
            # if batch["attention_mask"].all():
            # accelerator.print("\n----------------\n".join(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = False)))
            # accelerator.print(batch["input_ids"])
            # accelerator.print(batch["attention_mask"])
            # accelerator.print(batch["labels"])
            accelerator.print(batch["input_ids"].shape)
            accelerator.print(batch['attention_mask'].shape)
            # `accelerator.accumulate(model)` aims to set right `sync_gradients` state based on the recorded training steps
            with accelerator.accumulate(model):
                
                outputs = model(**batch)
                loss = outputs.loss
                accumulation_loss += loss.detach().float()
                # if accelerator.is_main_process:
                #     for param in model.parameters():
                #         if param.grad is not None:
                #             accelerator.print(f"Parameter grad is on device: {param.grad.device}")
                #     accelerator.print(f"Loss dtype: {loss.dtype}, Loss device: {loss.device}")


                # when deepspeed is enabled, `accelerator.backward(loss)` is doing optimizer.step(), optimizer.zero_grad(), and grad accumulation automatically. 
                # see `if self.is_gradient_accumulation_boundary():` line in path-to-env/site-packages/deepspeed/runtime/engine.py
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` examples
            if accelerator.sync_gradients:
                global_completed_steps += 1
                accelerator.print("GPU 0, step {}, loss {}".format(global_completed_steps, accumulation_loss / accelerator.gradient_accumulation_steps))
                accelerator.print("GPU 0, step {}, lr state dict:".format(global_completed_steps), lr_scheduler.state_dict())
                accelerator.print(time.time()-st)
                st = time.time()

                writer.add_scalar(
                    'train-loss/gpu-{}'.format(accelerator.process_index), 
                    accumulation_loss / accelerator.gradient_accumulation_steps, 
                    global_completed_steps
                )
                writer.add_scalar(
                    'learning-rate/gpu-{}'.format(accelerator.process_index), 
                    lr_scheduler.get_last_lr()[0], 
                    global_completed_steps
                )
                # reset accumulation_loss to 0
                accumulation_loss = 0

                # save checkpoints for each checkpointing_steps total batch size
                if global_completed_steps % opt.checkpointing_steps == 0:
                    accelerator.print("after {} global training steps, save a checkpoint".format(global_completed_steps))
                    accelerator.wait_for_everyone()
                    checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)

        accelerator.print("in the end of an epoch, save a checkpoint")
        accelerator.wait_for_everyone()
        checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)

if __name__ == "__main__":
    opt = parse_option()
    # os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    train(opt)
     