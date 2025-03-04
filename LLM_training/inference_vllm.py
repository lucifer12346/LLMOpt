import argparse
import json
import os
import re
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def prepare_sql2hints_prefix_sequence(data, choices = 1, statistics = [1,1,1,1,1]):
    prefix_seq = "*** sql ***\n{}\n".format(data['sql'])
    if statistics[0] == 1:
        prefix_seq += f"*** table information *** :\n{data['card_tb']}\n"
    if statistics[1] == 1:
        prefix_seq += f"*** NDV *** :\n {data['ndv']}"
    if statistics[2] == 1:
        prefix_seq += f"*** main value and times *** :\n {data['main_value']}"
    if statistics[3] == 1:
        prefix_seq += f"*** min and max *** :\n {data['min_max']}"
    if statistics[4] == 1:
        prefix_seq += f"*** histograms *** :\n {data['hists']}"
    if choices == 1:
        idx = 0
        prefix_seq = prefix_seq  + f"*** possible plans ***:\n"
        hints_set = data['shuffled_hints']
        for hint in hints_set:
            prefix_seq = prefix_seq + f"*** {idx} ***\n{hint}\n"
            idx += 1
    return prefix_seq

def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        return ""

def infer_data(llm, tokenizer, input_dataset, sampling_params, mode, stats=[1,1,1,1,1]):
    select = 1
    if "generate" in mode:
        select = 0 
    input_cp = input_dataset.copy()
    
    if select == 1:
        chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prepare_sql2hints_prefix_sequence(data, choices = 1, statistics=stats)}],
                add_generation_prompt = True, tokenize = False
            ) for data in input_cp]
    else:
        chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prepare_sql2hints_prefix_sequence(data, choices = 0, statistics=stats)}],
                add_generation_prompt = True, tokenize = False
            ) for data in input_cp]
    outputs = llm.generate(chat_prompts, sampling_params)
    
    responses = []
    for output, data in zip(outputs, input_cp):
        response = [o.text for o in output.outputs]
        if "idx" in opt.model_mode:
            pred_idxes = response.copy()
            response = []
            all_hints = data['shuffled_hints']
            for idx in pred_idxes:
                try:
                    if int(idx) < len(all_hints):
                        response.append(all_hints[int(idx)])
                except Exception as e:
                    print(e)
        responses.append(response)
         
        
    return responses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type = str, default = "/fs/fast/u2021000902/previous_nvme/xxx")
    parser.add_argument("--input_file", type = str, help = "the input file path (prompts)")
    parser.add_argument("--output_file", type = str, help = "the output file path (results)")
    parser.add_argument("--tensor_parallel_size", type = int, help = "the number of used GPUs", default = 4)
    parser.add_argument("--n", type = int, help = "the number of generated responses", default = 4)
    parser.add_argument("--temperature", type = float, help = "temperature of llm's sampling", default = 1.0)
    parser.add_argument("--model_mode", type=str, help = "the mode of model, select or generate", default="select")
    parser.add_argument("--without_id", type=int, default = -1)
    parser.add_argument("--cont", type=int, default = 0)
    opt = parser.parse_args()
    stats = [1,1,1,1,1]
    if opt.without_id > -1 and opt.without_id < 5:
        stats[opt.without_id] = 0
    elif opt.without_id == 6:
        stats = [0,0,0,0,0]
    print(opt)
    
    select_model = 1 if "select" in opt.model_mode else 0
    input_dataset = json.load(open(opt.input_file))
    if opt.cont == 1:
        for data in input_dataset:
            data['shuffled_hints'] = list(set(data['pred_hints']))
            
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path, trust_remote_code=True)
    
    if "Qwen2.5-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [151645] # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
    elif "deepseek-coder-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [32021]
    elif "DeepSeek-Coder-V2" in opt.pretrained_model_name_or_path:
        stop_token_ids = [100001]
    elif "OpenCoder-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [96539]
    elif "Meta-Llama-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [128009, 128001]
    else:
        print("Use Qwen2.5's stop tokens by default.")
        stop_token_ids = [151645]

    print("stop_token_ids:", stop_token_ids)

    max_model_len = 8192 # used to allocate KV cache memory in advance
    max_input_len = 6144
    max_output_len = 2048 # (max_input_len + max_output_len) must <= max_model_len
    
    print("temperature:", opt.temperature)
    sampling_params = SamplingParams(
        temperature = opt.temperature, 
        max_tokens = max_output_len,
        n = opt.n,
        stop_token_ids = stop_token_ids
    )

    llm = LLM(
        model = opt.pretrained_model_name_or_path,
        dtype = "bfloat16", 
        tensor_parallel_size = opt.tensor_parallel_size,
        max_model_len = max_model_len
    )
    results = []
    import time
    start_time = time.time()
    outputs = infer_data(llm, tokenizer, input_dataset, sampling_params, opt.model_mode, stats)
    temp = 1
    for data, output in zip(input_dataset, outputs):
        responses = output
        if opt.cont == 1:
            data["chosen_hints"] = responses
        else:
            data["pred_hints"] = responses
        results.append(data)
    end_time = time.time()
    print("inference time")
    print(end_time - start_time)
    with open(opt.output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(results, indent = 2, ensure_ascii = False))
    print("finish inference with VLLM")