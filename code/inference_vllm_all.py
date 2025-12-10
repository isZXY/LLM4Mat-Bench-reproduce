import os,sys
import json
import time
import torch
from datetime import datetime, timezone, timedelta
import pdb
import pandas as pd
import re
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
sys.path.append(os.path.join(os.path.dirname(__file__), './llmprop_and_matbert'))
from create_args_parser import *

LOG_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/inference_run_all.log"

def log_print(msg):
    """打印信息并写入日志文件"""
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

def parse_llama_prompt_to_messages(llama_prompt: str) -> List[Dict]:
    """
    将 Llama 风格的预渲染 Prompt 字符串解析为标准的 List[Dict] 消息格式。

    Args:
        llama_prompt: 包含 <s>, [INST], <<SYS>>, <</SYS>>, [/INST] 等标记的字符串。

    Returns:
        标准的 Hugging Face 消息列表格式：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
    """
    # 1. 定义正则表达式模式
    # 这个模式用于捕获 <<SYS>> 和 <</SYS>> 之间的 SYSTEM 内容
    system_pattern = r'<<SYS>>\s*(.*?)\s*<</SYS>>'
    
    # 2. 匹配并提取 SYSTEM 内容
    # re.DOTALL 使得 . 能够匹配换行符
    system_match = re.search(system_pattern, llama_prompt, re.DOTALL)
    
    system_content = ""
    if system_match:
        # 清理捕获到的内容中的多余空白符
        system_content = system_match.group(1).strip()
    
    # 3. 提取 USER 内容
    # USER 内容位于 SYSTEM 块之后，[/INST] 标记之前
    # 先找到 <<SYS>> 块的结束位置，然后从那里开始查找 [/INST]
    
    # 移除 system block 和 inst/sys tokens
    # r"(\[INST\].*?\[/INST\])" 捕获整个 INST/SYS 块
    inst_block_pattern = r"\[INST\]\s*(.*?)\s*\[/INST\]"
    inst_block_match = re.search(inst_block_pattern, llama_prompt, re.DOTALL)
    
    user_content = ""
    if inst_block_match:
        # 捕获 INST 和 /INST 之间的所有内容
        inst_content = inst_block_match.group(1).strip()
        
        # 从 INST 块内容中移除 SYSTEM 块，剩下的就是 USER 内容
        # 注意：这里需要处理没有 SYSTEM 块的情况
        if system_content:
            # 使用 re.escape 来确保特殊字符（如<<, >>）被正确匹配
            cleaned_system_content = re.escape(f"<<SYS>>\n{system_content}\n<</SYS>>")
            # 移除 system block，strip() 清理两侧空白
            user_content = re.sub(cleaned_system_content, '', inst_content, flags=re.DOTALL).strip()
        else:
            # 如果没有 system block，INST 块内容就是 user content
            user_content = inst_content
            
        # 移除 Llama 的起始 token <s> (如果存在)
        if user_content.startswith('<s>'):
             user_content = user_content[3:].strip()
            
    # 4. 构造标准 messages 列表
    messages: List[Dict] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    
    if user_content:
        messages.append({"role": "user", "content": user_content})
        
    return messages
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
    prompts_raw = list(data[prompt_col])

    # ==== 将llama格式prompts转换为huggingface标准格式 ====
    prompts: List[List[Dict]] = []
    for llama_prompt in prompts_raw:
        # 确保 llama_prompt 是字符串类型
        if pd.isna(llama_prompt):
            continue
            
        # 调用解析函数
        parsed_messages = parse_llama_prompt_to_messages(str(llama_prompt))
        
        if parsed_messages:
            prompts.append(parsed_messages)
        else:
            log_print(f"⚠️ Warning: Could not parse prompt:\n{llama_prompt[:100]}...")

    
    log_print(f"✅ Successfully parsed {len(prompts)} prompts into standard messages format.")

        

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

    # ==== 新增：获取 Tokenizer 对象===
    try:
        tokenizer = llm.get_tokenizer()
        log_print("🔧 Successfully retrieved tokenizer for manual template application.")
    except Exception as e:
        log_print(f"❌ Error retrieving tokenizer: {e}. Cannot manually apply chat template.")
        sys.exit(1)


    ## 设置模型特定采样参数
    # model_basename = os.path.basename(model_path)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=10, 
        top_p=1,
        max_tokens=256,
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


        # 2. *** 核心修改：手动渲染 messages 为字符串列表 ***
        batch_prompts_strings = []
        for messages in batch_prompts:
            # 使用 tokenizer 的 apply_chat_template 进行渲染
            rendered_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True # 必须添加，以指示模型开始生成
            )
            batch_prompts_strings.append(rendered_prompt)


        try:
            
            
            # # 🔍 打印应用模板后的字符串 (这是 vLLM 要求的输入格式)
            # log_print("🔍 ==== Prompt Preview (应用模板后的字符串) ====")
            # if batch_prompts_strings:
            #     log_print(f"[Prompt {start}] ----------------------------------")
            #     log_print(batch_prompts_strings[0])
            # # pdb.set_trace()

            outputs = llm.generate(batch_prompts_strings, sampling_params)
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