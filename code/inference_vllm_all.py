import os,sys
import json
import time
import torch
from datetime import datetime, timezone, timedelta

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
sys.path.append(os.path.join(os.path.dirname(__file__), './llmprop_and_matbert'))
from create_args_parser import *

LOG_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/inference_run_all.log"

def log_print(msg):
    print(msg)
    cst = timezone(timedelta(hours=8))  # 中国标准时间 UTC+8
    timestamp = datetime.now(cst).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")

def extract_ans_from_chat_llm(result):
    """从大模型输出中提取大括号内 JSON 内容"""
    start_index = result.find("{")
    end_index = result.find("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_content = result[start_index:end_index + 1]
        return json_content.strip()
    else:
        return result.strip()


def write_jsonl_line(where_to_save, record):
    """将单条结果写入 JSONL 文件"""
    with open(where_to_save, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    
    os.chdir("/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/code")


    # ==== 检查 GPU ====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log_print(f"🔧 Number of available devices: {torch.cuda.device_count()}")
        log_print(f"Current device is: {torch.cuda.current_device()}")
        log_print(f"Training and testing on {torch.cuda.device_count()} GPUs!")
        log_print("-" * 50)
    else:
        log_print("⚠️ No GPU available, running on CPU.")
        device = torch.device("cpu")



    # ==== 参数 ====
    args = args_parser()
    config = vars(args)
    
    dataset_name = config.get('dataset_name') 
    input_type = config.get('input_type') # description, structure, or composition
    prompt_type = config.get('prompt_type') # 'few_shot'( see five examples) /zero-shot
    max_len = config.get('max_len')
    property_name = config.get("property_name") # property name in dataset
    model_path = config.get("model_path")
    results_path = config.get("results_path")
    model_name = os.path.basename(model_path)
    batch_size = config.get("batch_size")

    os.makedirs(results_path, exist_ok=True)
    save_path = f"{results_path}/{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_{max_len}.json"

    


    # ==== 载入数据 ====
    data_path = f"../data/{dataset_name}/{dataset_name}_inference_prompts_data.csv"
    log_print(f"📂 Loading data from {data_path}")
    data = pd.read_csv(data_path)
    data = data.dropna(subset=[property_name])
    prompt_col = f"{property_name}_{input_type}_{prompt_type}"
    prompts = list(data[prompt_col])
    log_print(f"✅Loaded {len(prompts)} prompts for inference.")

    # ==== 初始化 vLLM ====
    log_print(f"🚀 Loading model from {model_path} ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),  
        dtype="bfloat16",  
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    sampling_params = SamplingParams(
        temperature=0.8,    
        top_k=10,          
        top_p=1.0,        
        max_tokens=256,  
        stop=["</s>", "\n\n\n"],
    )


    # ==== 推理 ====
    log_print("🧠 Start inference ...")
    start_time = time.time()

    total_prompts = len(prompts)
    num_batches = (total_prompts + batch_size - 1) // batch_size

    # 如果文件已存在，则跳过已完成部分
    completed = 0
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            completed = sum(1 for _ in f)
        log_print(f"⏩ Found existing file with {completed} completed samples, resuming...")

    for batch_idx in tqdm(range(completed // batch_size, num_batches), desc="Inference Progress", ncols=100):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_prompts)
        batch_prompts = prompts[start:end]
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            log_print(f"❌ Error during batch {batch_idx}: {e}")
            continue

        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text if len(output.outputs) > 0 else ""
            clean_result = extract_ans_from_chat_llm(response_text)

            record = {
                "response": clean_result
            }

            write_jsonl_line(save_path, record)

        # 显示进度
        done = end / total_prompts * 100
        log_print(f"✅ Completed {end}/{total_prompts} ({done:.1f}%)")


    end_time = time.time()
    elapsed = end_time - start_time
    log_print(f"\n🎯 Inference completed in {elapsed/60:.2f} minutes.")
    log_print(f"Results saved to: {save_path}")