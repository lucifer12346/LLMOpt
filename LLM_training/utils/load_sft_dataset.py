import json
import torch
import numpy as np
import random
from datasets import Dataset
from torch.utils.data import Dataset
import re 
def remove_sql_comments(sql):
    # Remove single-line comments
    sql = re.sub(r'--.*', '', sql)
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    sql = re.sub(r'\n+', '\n', sql, flags=re.DOTALL)
    return sql.strip()


def prepare_sql2hints_prefix_sequence(data, choices = 1, statistics = [1,1,1,1,1]):
    
    # import random
    # random.shuffle(data['all_hints'])
    data['sql'] = remove_sql_comments(data['sql'])
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
        for hint in data['shuffled_hints']:
            prefix_seq = prefix_seq + f"*** {idx} ***\n{hint}\n"
            idx += 1
    return prefix_seq

def find_sublist_index(lst, sublist):
    sublist_length = len(sublist)
    for i in range(len(lst) - sublist_length + 1):
        if lst[i:i + sublist_length] == sublist:
            return i
    return -1

def obtain_labels(input_ids, assistant_start_token_ids):
    '''
    Mask everything before assistant_start_token_ids with -100
    '''
    assistant_start_idx = find_sublist_index(input_ids, assistant_start_token_ids)
    if assistant_start_idx == -1:
        labels = input_ids
        print("length of the output sequence exceeds max length")
    else:
        labels = [-100] * assistant_start_idx + input_ids[assistant_start_idx: ]
    assert len(input_ids) == len(labels)

    return labels

def prepare_inputs_and_labels(chat_history, tokenizer, max_tokens):
    # print(chat_history)
    prefix_ids = tokenizer.apply_chat_template(chat_history, tokenize=True, add_generation_prompt=False)
    seq_length = len(prefix_ids)
    
    # assistant_start_token_ids = [151644, 77091] # for Qwen2.5's tokenizer, the start token ids of the Assistant (<|im_start|>assistant)
    assistant_start_token_ids = [128006, 78191, 128007] # for Llama's tokenizer, the sratr token ids of the Assistant(<|start_header_id|>assistant<|end_header_id|>)
    # assistant_start_token_ids = [13518,21289,25] #For Deepseek
    
    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradientsF
        labels = obtain_labels(prefix_ids, assistant_start_token_ids) + [-100] * pad_length
    else: # no padding
        print("the current input sequence {} exceeds the max_tokens, we will truncate it.".format(seq_length))
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + prefix_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = obtain_labels(input_ids, assistant_start_token_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }

def prepare_inputs(chat_history, tokenizer, max_prefix_length):
    input_ids = tokenizer.apply_chat_template(chat_history, tokenize=True, add_generation_prompt=True)
    seq_length = len(input_ids)
    
    if seq_length > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        print(len(input_ids))
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
    
    attention_mask = [1] * len(input_ids)
    # print("input length: ",seq_length)
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64),
    }

class SFTHintsDataset(Dataset):
    def __init__(self, sql2hints_data_dir, tokenizer, max_tokens, mode, model_type, stats):
        super().__init__()
        if tokenizer.eos_token_id == None:
            assert ValueError("tokenizer's EOS token is is empty.")
        if tokenizer.bos_token_id == None:
            assert ValueError("tokenizer's BOS token is is empty.")
        if tokenizer.pad_token_id == None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        with open(sql2hints_data_dir, 'r') as  fin:
            dataset = json.load(fin)

        self.dataset = []
        self.mode = mode
        self.stats = stats
        self.model_type = model_type
        for data in dataset:
            data['mode'] = model_type
                
            self.dataset.append(data.copy())
            
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        
            

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.model_type == "generate":
            input_seq = prepare_sql2hints_prefix_sequence(data, choices=0, statistics= self.stats)
            output_seq = data['best_hints']
        elif self.model_type == "select_idx":
            input_seq = prepare_sql2hints_prefix_sequence(data, statistics= self.stats)
            output_seq = data['shuffled_hints'].index(data['best_hints'])

        if self.mode == "train":
            chat_history = [
                {"role": "user", "content":str(input_seq)},
                {"role": "assistant", "content": str(output_seq)}
            ]
            
            return prepare_inputs_and_labels(chat_history, self.tokenizer, self.max_tokens)
        elif self.mode == "eval":
            chat_history = [
                {"role": "user", "content": input_seq}
            ]
            return prepare_inputs(chat_history, self.tokenizer, self.max_tokens)

    def __len__(self):
        return len(self.dataset)

