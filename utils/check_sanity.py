#!/usr/bin/env python3
from datetime import datetime, timezone, timedelta
import os,sys
import subprocess
import traceback
import pandas as pd
# 强制切换到脚本目录
SCRIPT_DIR = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/scripts"
os.chdir(SCRIPT_DIR)


# log配置
LOG_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/inference_run_all.log"
COMPLETED_FILE = "/public/home/sjtu_zhuxuanyu/LLM4Mat-Bench/completed_tasks.txt"

# 公共参数
DATA_PATH = "../data"
INFERENCE_SCRIPT = "../code/inference_vllm_all.py"  # ✅ 统一推理入口


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



def expected_columns(props, inputs, prompts):
    """生成预期列名"""
    cols = []
    for prop in props:
        for inp in inputs:
            for prm in prompts:
                cols.append(f"{prop}_{inp}_{prm}")
        cols.append(prop)  # 原始属性列int()
       #  print('expected_columns-',cols)
    return cols


for dataset, props in property_names.items():
    csv_path = f"{DATA_PATH}/{dataset}/{dataset}_inference_prompts_data.csv"
    if not os.path.exists(csv_path):
        print(f"[{dataset}] 文件不存在: {csv_path}")
        continue

    df = pd.read_csv(csv_path, nrows=0)  # 只读取表头
    actual_cols = set(df.columns)
    expected_cols = expected_columns(props, input_types, prompt_types)
    # print(expected_cols)
    # 找出缺失和存在的列
    missing = [col for col in expected_cols if col not in actual_cols]
    present = [col for col in expected_cols if col in actual_cols]

    # 打印结果
    if missing:
        for col in missing:
            print(f"❌{dataset} 缺失列: {col}")
    else: print(f"✅{dataset} 所有列都存在")
    # for col in present:
    #     print(f"{dataset} 存在列: {col}")