from transformers import AutoTokenizer
import torch
import os
import time
import pandas as pd
import json
import argparse
from datasets import Dataset
from tqdm import tqdm
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), './llmprop_and_matbert'))
from create_args_parser import *

from vllm import LLM, SamplingParams
from vllm.callbacks import ConsoleCallbackHandler

# 自定义回调，只更新进度，不打印文本
class ProgressCallback(Callback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Generating prompts")
    def on_generation_start(self, request_id, prompt):
        pass
    def on_output(self, request_id, output):
        # 当每条 prompt 生成完成时更新进度
        self.pbar.update(1)
    def on_generation_end(self):
        self.pbar.close()


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

    
    
    # '''load Data as Dataset'''
    data = pd.read_csv(f"../{data_path}/{dataset_name}/{dataset_name}_inference_prompts_data.csv")
    data = data.dropna(subset=[property_name])
    prompts = list(data[f'{property_name}_{input_type}_{prompt_type}'])

    # dataset = Dataset.from_dict({"prompt": prompts_list})
    
    '''load model'''
    model_path = "/public/share/model/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True) 

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",  #原文float16
        tensor_parallel_size=torch.cuda.device_count()  
    )

    progress_cb = ProgressCallback(total=len(prompts))


    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,  # 可根据需要调整
        top_k=10,
        max_tokens=max_len,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    print(f"Start generation on {len(prompts)} prompts ...")

    outputs = llm.generate(
        prompts,
        sampling_params,
        max_batch_size= batch_size,  # 可调整
        callbacks=[progress_cb]  # 显示生成进度
    )
    first_item = True # for comma


    # --- 写入结果 ---
    with open(save_path, "a", encoding="utf-8") as f:
        f.write("[")
        for output in tqdm(outputs, desc="Writing results"):
            generated_text = output.outputs[0].text
            ans = extract_ans_from_chat_llm(generated_text)
            if not first_item:
                f.write(",")
            f.write(json.dumps(ans, ensure_ascii=False))
            first_item = False
        f.write("]")

    end = time.time()
    print('took:', end-start,'seconds')