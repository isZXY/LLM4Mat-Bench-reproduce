#!/usr/bin/env python3
from datetime import datetime, timezone, timedelta
import os,sys
import subprocess
import traceback

# 强制切换到脚本目录
SCRIPT_DIR = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/scripts"
os.chdir(SCRIPT_DIR)


# log配置
LOG_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/inference_run_all.log"
COMPLETED_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/completed_tasks.txt"
UNCOMPLETE_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/uncomplete_tasks.txt"

# 公共参数
DATA_PATH = "../data"
INFERENCE_SCRIPT = "../code/inference_vllm_all.py"
RESULTS_PATH = "../results/"
MAX_LEN = 4000
BATCH_SIZE = 256

# === 参数定义 ===
# 模型名称（例如不同的推理脚本或路径）
model_paths = [
    # "/public/share/model/ChemDFM-v2.0-14B",
    # "/public/share/model/Intern-S1-mini",
    # "/public/share/model/Qwen3-8B",
    # "/public/share/model/Qwen2.5-14B-Instruct",


    "/public/share/model/Qwen3-14B",
    "/public/share/model/Qwen3-4B",

    # "/public/share/model/Llama-2-7b-chat-hf",
    # "/public/share/model/Meta-Llama-3.1-8B-Instruct",
]


# 数据集名称
dataset_names = [
    "cantor_hea",
    # "gnome",
    # "hmof",
    # "jarvis_dft",
    # "jarvis_qetb",
    "mp",
    "omdb",
    "oqmd",
    # "qmof",
    # "snumat",
]

# 输入类型
input_types = [
    "formula",
    "cif_structure", 
    "description",
]

# 不同数据集对应的属性名称
property_names = {
    "cantor_hea": [
        "Ef_per_atom",
        "e_above_hull",
        "volume_per_atom",
        "e_per_atom",
    ],
    "gnome": [
        "Formation_Energy_Per_Atom",
        "Decomposition_Energy_Per_Atom",
        "Bandgap",
        "Corrected_Energy",
        "Volume",
        "Density",
    ],
    "hmof": [
        "max_co2_adsp",
        "min_co2_adsp",
        "lcd",
        "pld",
        "void_fraction",
        "surface_area_m2g",
        "surface_area_m2cm3",
    ],
    "jarvis_dft": [
        "formation_energy_peratom",
        "optb88vdw_bandgap",
        "slme",
        "spillage",
        "optb88vdw_total_energy",
        "mepsx",
        "max_efg",
        "avg_elec_mass",
        "dfpt_piezo_max_eij",
        "dfpt_piezo_max_dij",
        "dfpt_piezo_max_dielectric",
        "n-Seebeck",
        "n-powerfact",
        "p-Seebeck",
        "p-powerfact",
        "exfoliation_energy",
        "bulk_modulus_kv",
        "shear_modulus_gv",
        "mbj_bandgap",
        "ehull",
    ],
    "jarvis_qetb": [
        "energy_per_atom",
        "indir_gap",
        "f_enp",
        "final_energy",
    ],
    "mp": [
        "band_gap",
        "volume",
        "is_gap_direct",
        "formation_energy_per_atom",
        "energy_above_hull",
        "energy_per_atom",
        "is_stable",
        "density",
        "density_atomic",
        "efermi",
    ],
    "omdb": [
        "bandgap",
    ],
    "oqmd": [
        "bandgap",
        "e_form",
    ],
    "qmof": [
        "energy_total",
        "bandgap",
        "lcd",
        "pld",
    ],
    "snumat": [
        "Band_gap_HSE",
        "Band_gap_GGA",
        "Band_gap_GGA_optical",
        "Band_gap_HSE_optical",
        "Direct_or_indirect",
        "Direct_or_indirect_HSE",
        "SOC",
    ],
}


# zero-shot/few-shot
prompt_types = [
    "zero_shot", 
    "few_shot"
]

def log_message(message):
    """写入日志文件（带CST时间戳）"""
    cst = timezone(timedelta(hours=8))  # 中国标准时间 UTC+8
    timestamp = datetime.now(cst).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

if __name__ == "__main__":
    ''' before  pls run data sanity check(../utils/check_sanity.py)'''

    open(LOG_FILE, "w", encoding="utf-8").close()

    # === 载入已完成任务 ===
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
            completed_tasks = set(line.strip() for line in f if line.strip())
    else:
        completed_tasks = set()

    log_message(f"🔧 Loaded {len(completed_tasks)} completed tasks from previous runs.\n")


    # === 预计算总任务数 ===
    all_tasks = []
    for dataset_name in dataset_names:
        for input_type in input_types:
            for property_name in property_names[dataset_name]:
                for prompt_type in prompt_types:
                    for model_path in model_paths:
                        model_name = os.path.basename(model_path)
                        task_id = f"{dataset_name}|{input_type}|{property_name}|{prompt_type}|{model_name}"
                        all_tasks.append((task_id, dataset_name, input_type, property_name, prompt_type, model_path))

    total_tasks = len(all_tasks)
    log_message(f"🔧 Total tasks to run: {total_tasks}\n")

    # === 串行执行并记录日志 ===
    for idx, (task_id, dataset_name, input_type, property_name, prompt_type, model_path) in enumerate(all_tasks, start=1):
        if task_id in completed_tasks:
            log_message(f"⏩ Skipping completed task ({idx}/{total_tasks}): {task_id}")
            continue

        model_name = os.path.basename(model_path)
        progress_msg = f"[Progress: {idx}/{total_tasks}] Running: model={model_name}, dataset={dataset_name}, input={input_type}, property={property_name}, prompt={prompt_type}"
        log_message(progress_msg)


        result_dir = os.path.join(
            RESULTS_PATH,
            f"{model_name}_{dataset_name}_{input_type}_{property_name}_{prompt_type}"
        )
        os.makedirs(result_dir, exist_ok=True)

        cmd = [
            "python", INFERENCE_SCRIPT,
            "--model_path",model_path,
            "--results_path", result_dir,
            "--dataset_name", dataset_name,
            "--input_type", input_type,
            "--property_name", property_name,
            "--max_len", str(MAX_LEN),
            "--prompt_type", prompt_type,
            "--batch_size",str(BATCH_SIZE)

        ]

        log_message(f"✅ Start ({idx}/{total_tasks}): {cmd}")

        try:

            subprocess.run(
                cmd,
                check=True,
                text=True
            )

            log_message(f"✅ Finished ({idx}/{total_tasks}) Successfully.")
            with open(COMPLETED_FILE, "a", encoding="utf-8") as f:
                f.write(task_id + "\n")

        except subprocess.CalledProcessError as e:
            log_message(f"❌ Error While Running({idx}/{total_tasks}): {cmd}")
            log_message(f"Exit code: {e.returncode}")
            log_message(f"--- TRACEBACK ---\n{traceback.format_exc()}")
            with open(UNCOMPLETE_FILE, "a", encoding="utf-8") as f:
                f.write(task_id + "\n")
            log_message("Continuing to next task...\n")

        except Exception as e:
            log_message(f"⚠️ Unexpected error ({idx}/{total_tasks}): {e}")
            log_message(f"--- TRACEBACK ---\n{traceback.format_exc()}")
            log_message("Continuing to next task...\n")

    log_message("\n🎯 All tasks finished!\n")