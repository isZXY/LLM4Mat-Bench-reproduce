import subprocess
import multiprocessing
import os

def run_task(dataset, input_type, property_name, prompt_type, model_name, data_path, results_path, min_samples):
    # 构造命令
    command = [
        "python", "../code/evaluate.py",
        "--data_path", data_path,
        "--results_path", results_path,
        "--dataset_name", dataset,
        "--input_type", input_type,
        "--property_name", property_name,
        "--prompt_type", prompt_type,
        "--min_samples", str(min_samples),
        "--model_name", model_name
    ]
    
    try:
        # 执行命令，并捕获标准输出和错误输出
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        # 如果命令成功执行，打印其输出
        print(result.stdout)
    
    except subprocess.CalledProcessError as e:
        # 捕获子进程执行时的错误
        error_type = type(e).__name__  # 错误类型
        error_message = f"Error Type: {error_type}, Command: {' '.join(command)}"
        
        # 捕获标准输出和标准错误
        error_details = f"stdout: {e.stdout}\nstderr: {e.stderr}\n"
        
        # 写入错误日志
        with open("../evaluate_error_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{dataset}|{input_type}|{property_name}|{prompt_type}|{model_name}|{error_message}|{error_details}\n")
        
        print(f"Error occurred in subprocess: {e} -- Error details written to error_log.txt")
    
    except FileNotFoundError as e:
        # 捕获文件未找到错误
        error_type = type(e).__name__
        error_message = f"File not found: {str(e)}"
        
        # 写入错误日志
        with open("../evaluate_error_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{dataset}|{input_type}|{property_name}|{prompt_type}|{model_name}|{error_type}|{error_message}\n")
        
        print(f"FileNotFoundError occurred: {e} -- Error details written to error_log.txt")
    
    except OSError as e:
        # 捕获操作系统错误（如权限问题）
        error_type = type(e).__name__
        error_message = f"OSError: {str(e)}"
        
        # 写入错误日志
        with open("../evaluate_error_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{dataset}|{input_type}|{property_name}|{prompt_type}|{model_name}|{error_type}|{error_message}\n")
        
        print(f"OSError occurred: {e} -- Error details written to error_log.txt")
    
    except Exception as e:
        # 捕获其他任何异常
        error_type = type(e).__name__
        error_message = f"Unexpected Error: {str(e)}"
        
        # 写入错误日志
        with open("../evaluate_error_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{dataset}|{input_type}|{property_name}|{prompt_type}|{model_name}|{error_type}|{error_message}\n")
        
        print(f"Unexpected error occurred: {e} -- Error details written to error_log.txt")

# 主函数
def main():
    os.chdir("/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/scripts")
    output_path = "../results/final_evaluate.csv"
    if os.path.exists(output_path): 
        os.remove(output_path) 
        print(f"{output_path} 文件已删除.")
    err_log = '../evaluate_error_log.txt'
    if os.path.exists(err_log): 
        os.remove(err_log) 
        print(f"{err_log} 文件已删除.")
    completed_tasks_file = "../completed_tasks.txt"
    data_path = '../data'
    results_path = '../results'
    min_samples = 2

    # 获取系统的 CPU 核心数
    cpu_count = multiprocessing.cpu_count()
    print(f"系统的 CPU 核心数：{cpu_count}")
    
    # 设置最大并行进程数（CPU 核心数）
    max_processes = cpu_count  # 这里是一个简单的例子，最多启动与 CPU 核心数相等的进程

    tasks = []
    
    # 读取任务并存储
    with open(completed_tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            dataset, input_type, property_name, prompt_type, model_name = line.split("|")
            tasks.append((dataset, input_type, property_name, prompt_type, model_name, data_path, results_path, min_samples))

    # 使用进程池控制并发数
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(run_task, tasks)

    print("所有任务完成。")

if __name__ == "__main__":
    main()
