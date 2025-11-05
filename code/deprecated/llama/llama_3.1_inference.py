from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaTokenizer
try:
    from transformers import LlamaForSequenceClassification
except ImportError:
    from transformers import AutoModelForSequenceClassification as LlamaForSequenceClassification

import transformers
import torch
# import replicate
import os
import time
import pandas as pd
import json
import argparse
from pathlib import Path

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../llmprop_and_matbert'))
from create_args_parser import *

def extract_ans_from_chat_llm(result):
    # Find the content within curly braces
    start_index = result.find('{')
    end_index = result.find('}')

    # Extract the content and format as JSON
    json_content = result[start_index:end_index + 1]
    return json_content

def extract_ans_from_next_token_llm(result):
    result = str(result)
    if len(result) != 0:
        output = result.split()
    else:
        output = result
    return output

def writeToJSON(data, where_to_save):
    """
    data: a dictionary that contains data to save
    where_to_save: the name of the file to write on
    """
    with open(where_to_save, "w", encoding="utf8") as outfile:
        json.dump(data, outfile)

        
def readJSON(input_file):
    """
    1. arguments
        input_file: a json file to read
    2. output
        a json objet in a form of a dictionary
    """
    with open(input_file, "r", encoding="utf-8", errors='ignore') as infile:
        json_object = json.load(infile, strict=False)
    return json_object

def generate(model, tokenizer, prompts, max_len, batch_size):
    results = []
    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        batch_size=batch_size,
    )

    # pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]


    sequences = pipe(
            prompts,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_len,
            truncation=True,
            return_full_text=False
        )

    cnt_seqs = 0
    
    for seqs in sequences:
        for seq in seqs:
            results.append(extract_ans_from_chat_llm(seq['generated_text']))
            print("cnt_seqs,cnt_seq:",cnt_seqs,cnt_seq)
        
    
    return results

if __name__ == "__main__":
    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Training and testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")

    # set parameters
    args = args_parser()
    config = vars(args)
    
    dataset_name = config.get('dataset_name') 
    input_type = config.get('input_type') # description, structure, or composition
    prompt_type = config.get('prompt_type') # 'few_shot'( see five examples) /zero-shot
    batch_size = config.get("batch_size")
    max_len = config.get('max_len')
    property_name = config.get("property_name") # property name in dataset
    model_name = config.get("model_name")
    data_path = config.get("data_path")
    results_path = config.get("results_path")
    
    results_path = f"{results_path}/{dataset_name}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    start = time.time()
    
    model = "/public/share/model/Meta-Llama-3.1-8B-Instruct"
    # hf_model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model, padding=True,local_files_only=True,trust_remote_code=True) 
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    data = pd.read_csv(f"../{data_path}/{dataset_name}/{dataset_name}_inference_prompts_data.csv")
    # print(f"../{data_path}/{dataset_name}/{dataset_name}_inference_prompts_data.csv")
    data = data.dropna(subset=[property_name])

    prompts = list(data[f'{property_name}_{input_type}_{prompt_type}'])

    results = generate(model, tokenizer, prompts, max_len, batch_size)

    save_path = f"{results_path}/{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.json"
    writeToJSON(results, save_path)

    end = time.time()
    print('took:', end-start,'seconds')