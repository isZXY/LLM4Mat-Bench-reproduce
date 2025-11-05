from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaTokenizer
import torch
import os
import time
import pandas as pd
import json
import argparse
from datasets import Dataset
from tqdm import tqdm
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


if __name__ == "__main__":

    # '''check if the GPU is available'''
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

    # '''set parameters'''
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
    
    save_path = f"{results_path}/{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}_{batch_size}.json"

    start = time.time()
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[")

    first_item = True # for comma
    
    # '''load Data as Dataset'''
    data = pd.read_csv(f"../{data_path}/{dataset_name}/{dataset_name}_inference_prompts_data.csv")
    data = data.dropna(subset=[property_name])
    prompts = list(data[f'{property_name}_{input_type}_{prompt_type}'])

    # dataset = Dataset.from_dict({"prompt": prompts_list})
    
    '''load model'''
    model_path = "/public/share/model/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding=True,use_fast=False, local_files_only=True) 

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        batch_size=batch_size,
    )


    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    print("Model device map:", pipe.model.hf_device_map)

    for i in tqdm(range(0, len(prompts), batch_size), total=(len(prompts)+batch_size-1)//batch_size, desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        
        sequences = pipe(
            batch_prompts,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_len,
            truncation=True,
            return_full_text=False
        )
        batch_results = []
        for seqs in sequences:
            for seq in seqs:
                # 这里可以用你的 extract_ans_from_chat_llm
                batch_results.append(extract_ans_from_chat_llm(seq['generated_text']))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # write-in
        with open(save_path, "a", encoding="utf-8") as f:
            for item in batch_results:
                if not first_item:
                    f.write(",")  # 每个元素前加逗号
                f.write(json.dumps(item, ensure_ascii=False))
                first_item = False


    with open(save_path, "a", encoding="utf-8") as f:
        f.write("]")

    end = time.time()
    print('took:', end-start,'seconds')