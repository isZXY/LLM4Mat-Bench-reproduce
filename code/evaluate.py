import pandas as pd
import numpy as np
import json
import re
import csv
# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics
import sys, os


sys.path.append(os.path.join(os.path.dirname(__file__), './llmprop_and_matbert'))
from create_args_parser import *

def save_final_evaluate_csv(

    dataset_name,
    model_name, 
    input_type, 
    prompt_type, 
    property_name,
    record_raw_cnt=None,
    record_ex_nan_cnt=None,
    record_ex_predict_nan_cnt=None,
    record_predicted_cnt=None,
    record_roc_score=None,
    record_max=None,
    record_min=None,
    record_mae=None,
    record_rmse=None,
    record_mad=None,
    record_mad_mae_ratio=None,
    output_path="../results/final_evaluate.csv"
):
    """
    创建或更新 final_evaluate.csv 统计文件。

    表头包含：
    原始样本数, 去除target NaN样本数, 去除预测 NaN样本数, 提取预测后样本数,
    有效率（提取预测后样本数/去除target NaN样本数）, max, min, MAE↓, RMSE↓

    如果某个变量不存在或为 None，则写入 "unavailable"。
    """

    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 计算有效率（如果数据可用）
    if record_predicted_cnt is not None and record_ex_nan_cnt:
        try:
            efficiency = (record_predicted_cnt / record_ex_nan_cnt) * 100
        except ZeroDivisionError:
            efficiency = "unavailable"
    else:
        efficiency = "unavailable"

    # 统一格式化，防止None出错
    def safe_value(val):
        if val is None:
            return "unavailable"
        if isinstance(val, float):
            return f"{val:.6f}"
        return val

    # 准备一行数据
    row = [
        safe_value(dataset_name),
        safe_value(model_name),
        safe_value(input_type),
        safe_value(prompt_type),
        safe_value(property_name),
        safe_value(record_raw_cnt),
        safe_value(record_ex_nan_cnt),
        safe_value(record_ex_predict_nan_cnt),
        safe_value(record_predicted_cnt),
        f"{efficiency:.2f}%" if isinstance(efficiency, (int, float)) else efficiency,
        safe_value(record_max),
        safe_value(record_min),
        safe_value(record_mae),
        safe_value(record_rmse),
        safe_value(record_roc_score),
        safe_value(record_mad),
        safe_value(record_mad_mae_ratio),

    ]
    # 判断文件是否存在
    file_exists = os.path.exists(output_path)

    # 写入或追加
    with open(output_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "数据集",
                "模型",
                "输入类型(structure,description,formula)",
                "提示词类型(zero-shot/few-shot)",
                "属性名称",
                "原始样本数",
                "去除target NaN样本数",
                "去除预测 NaN样本数",
                "提取预测后样本数",
                "有效率（提取预测后样本数/去除target NaN样本数）",
                "max",
                "min",
                "MAE↓",
                "RMSE↓",
                "ROC-AUC↑",
                "MAD↓",
                "MAD:MAE↑"
            ])
        writer.writerow(row)

    print(f"✅ 统计结果已写入 {output_path}")


def readJSONL(input_file):
    """
    修改版：
    读取原始 JSON 列表（可能包含空字符串），
    输出一个 JSONL 风格的列表，每个元素为 {"response": "<原字符串>"}。
    """
    
    jsonl_formatted = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    jsonl_formatted.append(obj)
                elif isinstance(obj, dict) and "response" in obj:
                    jsonl_formatted.append(obj["response"])
                else:
                    jsonl_formatted.append(json.dumps(obj, ensure_ascii=False))
            except json.JSONDecodeError:
                jsonl_formatted.append(line)

    return jsonl_formatted

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

def contains_elements_and_matches(input_string, elements):
    matching_elements = [element for element in elements if element in input_string]
    return bool(matching_elements), matching_elements

def extract_values(sentence):
    chem_elts = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts']
    contains_elements, matching_elements = contains_elements_and_matches(sentence, chem_elts)
    
    # filter out chemical compound in answers
    if contains_elements:
        pattern = re.compile(rf'\b\w*{matching_elements[0]}\w*\b')
        sentence = re.sub(pattern, '', sentence)

    match_1 = re.search(r'(\d+(\.\d+)?)\s*x\s*10\s*\^*\s*(-?\d+)', sentence) # matching 2 x 10^6/2 x 10^-6values
    match_2 = re.search(r'(\d+(\.\d+)?)\s*×\s*10\s*\^*\s*(-?\d+)', sentence) # matching 2 × 10^6/2 × 10^-6 values
    match_3 = re.search(r'(\d+(\.\d+)?[eE][+-]?\d+)', sentence) # match 1e6 or 1E-08

    if match_1:
        value, _, exponent = match_1.groups()
        value = float(value)
        exponent = int(exponent)
        result = value * 10**exponent
        if result >= 100000.0 or result <= -100000.0:
            result = None

    elif match_2:
        value, _, exponent = match_2.groups()
        value = float(value)
        exponent = int(exponent)
        result = value * 10**exponent
        if result >= 100000.0 or result <= -100000.0:
            result = None
        
    elif match_3:
        notation = match_3.group()
        result = float(notation)
        if result >= 100000.0 or result <= -100000.0:
            result = None

    else:
        if "10^" in sentence:
            pattern = re.compile(r'(?<=\^)[+-]?\d+')
            match = pattern.search(sentence)
            if match:
                exponent = int(match.group())
                result = 10**exponent
                if result >= 100000.0 or result <= -100000.0:
                        result = None
            else:
                result = None
        else:
            matches = re.findall(r'(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)', sentence) #matches "1.0-3.4"->[('1.0', '3.4')]
            if len(matches) > 1:
                numbers = [float(matches[0][i]) for i in range(len(matches[0]))]
                result = np.array(numbers).mean()
            else:
                matches = re.findall(r'-?\d+\.?\d*', sentence)
                numbers = [float(number) if '.' in number else int(number) for number in matches]
                if len(numbers) > 0:
                    result = numbers[0]
                    if result >= 100000.0 or result <= -100000.0:
                        result = None
                else:
                    result = None
    return result

def extract_mp_gap_direct_predictions(sentence):
    positive_predictions = ['True','true','yes','likely','is a direct gap','Yes']
    negative_predictions = ['False','false', 'indirect']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_mp_stability_predictions(sentence):
    positive_predictions = [':stable', ':Stable',' stable', ' Stable','True','true','yes','likely}','Yes']
    negative_predictions = ['unstable', 'Unstable','False','false']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_snumat_direct_predictions(sentence):
    sentence = sentence.replace('\n\"CuCNH2S2\": \"Indirect\",\n\"Ge\": \"Direct\",\n\"TlFe2S3\": \"Indirect\",\n\"GaCl3\": \"Indirect\",\n\"(Nb2Tl5(SCl2)4)2Cl2\": \"Indirect\"', '')
    positive_predictions = [' direct', ' Direct','"direct\"','"Direct\"', ':Direct', ':direct']
    negative_predictions = [' indirect', ' Indirect', '"indirect\"', '"Indirect\"',':Indirect', ':indirect']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_snumat_direct_hse_predictions(sentence):
    sentence = sentence.replace('\n\"CuCNH2S2\": \"Indirect HSE\",\n\"Ge\": \"Direct HSE\",\n\"TlFe2S3\": \"Direct HSE\",\n\"GaCl3\": \"Indirect HSE\",\n\"(Nb2Tl5(SCl2)4)2Cl2\": \"Indirect HSE\"', '')
    positive_predictions = [' direct', ' Direct','"direct\"','"Direct\"', '\"Direct HSE\"', '\"direct HSE\"', ' direct HSE', ' Direct HSE', ':Direct', ':direct', ':Direct HSE', ':direct HSE']
    negative_predictions = [' indirect', ' Indirect', '"indirect\"', '"Indirect\"', 'indirect HSE', 'Indirect HSE',':Indirect', ':indirect', ':Indirect HSE', ':indirect HSE']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result

def extract_snumat_soc_predictions(sentence):
    positive_predictions = ['True','true','yes','likely}','Yes']
    negative_predictions = ['False','false']
    matching_positive = [prediction for prediction in positive_predictions if prediction in sentence]
    matching_negative = [prediction for prediction in negative_predictions if prediction in sentence]
    
    if bool(matching_positive):
        result = 1.0
    elif bool(matching_negative):
        result = 0.0
    else:
        result = None
    return result 

def extract_predictions(dataset_name, model_name, data_path, results_path, input_type, prompt_type, property_name, max_len):
    print(f"Results on {dataset_name}:\n")


    data = pd.read_csv(f"{data_path}/{dataset_name}/{dataset_name}_inference_prompts_data.csv")

    data_dp = data.dropna(subset=[property_name])

    folder_path = f"{model_name}_{dataset_name}_{input_type}_{property_name}_{prompt_type}"

    json_file_name = f"{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_4000.json"

    predictions = readJSONL(f"{results_path}/{folder_path}/{json_file_name}")

    record_raw_cnt=None
    record_ex_nan_cnt=None
    record_ex_predict_nan_cnt=None
    record_predicted_cnt=None
    record_roc_score=None
    record_max=None
    record_min=None
    record_mae=None
    record_rmse=None
    record_mad=None
    record_mad_mae_ratio=None

    results_df = pd.DataFrame({f'{property_name}_target': list(data_dp[property_name]), f'{property_name}_predicted': predictions})
    print(f'original results for {property_name}:', len(results_df))
    record_raw_cnt = len(results_df)
    
    results_df = results_df[~results_df.isin([np.inf, -np.inf]).any(axis=1)]
    results_df = results_df.dropna(subset=[f'{property_name}_target']).reset_index(drop=True)
    print(f'after dropping target nans:', len(results_df)) 
    record_ex_nan_cnt = len(results_df)

    results_df[f'{property_name}_predicted'] = results_df[f'{property_name}_predicted'].replace('', pd.NA)
    results_df = results_df.dropna(subset=[f'{property_name}_predicted']).reset_index(drop=True) 
    print(f'after dropping predicted nans:', len(results_df))
    record_ex_predict_nan_cnt = len(results_df)

    if dataset_name == 'mp': 
        if property_name == 'is_gap_direct':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_mp_gap_direct_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
                record_roc_score = roc_score
            else:
                print('Invalid')
        elif property_name == 'is_stable':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_mp_stability_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
                record_roc_score = roc_score
            else:
                print('Invalid')
        else:
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_values)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            print('max: ', results_df[f'{property_name}_extracted_predictions'].max())
            record_max = results_df[f'{property_name}_extracted_predictions'].max()
            print('min: ', results_df[f'{property_name}_extracted_predictions'].min())
            record_min = results_df[f'{property_name}_extracted_predictions'].max()
            if len(results_df) >= min_samples:
                y_true = list(results_df[f'{property_name}_target'])
                y_pred = list(results_df[f'{property_name}_extracted_predictions'])


                mae = metrics.mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

                y_true_mean = np.mean(y_true)
                mad = np.mean(np.abs(y_true - y_true_mean))
                mad_mae_ratio = mad / mae if mae != 0 else np.nan


                print('MAE: ', mae)
                print('RMSE: ', rmse)
                print('MAD (from ground truth): ', mad)
                print('MAD:MAE Ratio: ', mad_mae_ratio)

                record_mae = mae
                record_rmse = rmse
                record_mad = mad
                record_mad_mae_ratio = mad_mae_ratio
            else:
                print("Invalid")

    elif dataset_name == 'snumat':
        results_df = results_df.drop(results_df[results_df[f'{property_name}_target'] == 'Null'].index).reset_index(drop=True)
        results_df.loc[results_df[f'{property_name}_target'] == "Direct", f'{property_name}_target'] = 1.0
        results_df.loc[results_df[f'{property_name}_target'] == "Indirect", f'{property_name}_target'] = 0.0
        results_df[f'{property_name}_target'] = results_df[f'{property_name}_target'].astype(float)

        if property_name == 'Direct_or_indirect':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_direct_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
                record_roc_score = roc_score
            else:
                print("Invalid")
        elif property_name == 'Direct_or_indirect_HSE':
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_direct_hse_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            if len(results_df) >= min_samples:
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
                record_roc_score = roc_score
            else:
                print("Invalid")
        elif property_name == "SOC":
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_snumat_soc_predictions)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            if len(results_df) >= min_samples: 
                roc_score = metrics.roc_auc_score(list(results_df[f'{property_name}_target']), list(results_df[f'{property_name}_extracted_predictions']))
                print('ROC Score: ', roc_score)
                record_roc_score = roc_score
            else:
                print("Invalid")
        else:
            results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_values)
            results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
            print(f'after extracting predictions:', len(results_df)) 
            record_predicted_cnt = len(results_df)
            print('max: ', results_df[f'{property_name}_extracted_predictions'].max())
            record_max = results_df[f'{property_name}_extracted_predictions'].max()
            print('min: ', results_df[f'{property_name}_extracted_predictions'].min())
            record_min = results_df[f'{property_name}_extracted_predictions'].min()


            if len(results_df) >= min_samples:
                y_true = list(results_df[f'{property_name}_target'])
                y_pred = list(results_df[f'{property_name}_extracted_predictions'])


                mae = metrics.mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

                y_true_mean = np.mean(y_true)
                mad = np.mean(np.abs(y_true - y_true_mean))
                mad_mae_ratio = mad / mae if mae != 0 else np.nan


                print('MAE: ', mae)
                print('RMSE: ', rmse)
                print('MAD (from ground truth): ', mad)
                print('MAD:MAE Ratio: ', mad_mae_ratio)

                record_mae = mae
                record_rmse = rmse
                record_mad = mad
                record_mad_mae_ratio = mad_mae_ratio
            else:
                print("Invalid")
    
    else:
        results_df[f'{property_name}_extracted_predictions'] = results_df[f'{property_name}_predicted'].apply(extract_values)
        results_df = results_df.dropna(subset=[f'{property_name}_extracted_predictions']).reset_index(drop=True)
        print(f'after extracting predictions:', len(results_df)) 

        record_predicted_cnt = len(results_df)
        print('max: ', results_df[f'{property_name}_extracted_predictions'].max())
        print('min: ', results_df[f'{property_name}_extracted_predictions'].min())
        record_max = results_df[f'{property_name}_extracted_predictions'].max()
        record_min =  results_df[f'{property_name}_extracted_predictions'].min()

        if len(results_df) >= min_samples:
            y_true = list(results_df[f'{property_name}_target'])
            y_pred = list(results_df[f'{property_name}_extracted_predictions'])


            mae = metrics.mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

            y_true_mean = np.mean(y_true)
            mad = np.mean(np.abs(y_true - y_true_mean))
            mad_mae_ratio = mad / mae if mae != 0 else np.nan


            print('MAE: ', mae)
            print('RMSE: ', rmse)
            print('MAD (from ground truth): ', mad)
            print('MAD:MAE Ratio: ', mad_mae_ratio)

            record_mae = mae
            record_rmse = rmse
            record_mad = mad
            record_mad_mae_ratio = mad_mae_ratio
        else:
            print("Invalid")
    

    save_final_evaluate_csv(dataset_name, model_name, input_type, prompt_type, property_name, record_raw_cnt, record_ex_nan_cnt, record_ex_predict_nan_cnt, record_predicted_cnt, record_roc_score, record_max, record_min,record_mae,record_rmse,record_mad,record_mad_mae_ratio)
    results_df.to_csv(f"{results_path}/{folder_path}/{model_name}_test_stats_for_{property_name}_{input_type}_{prompt_type}_4000.csv", index=False)
    print('-'*50)
        
if __name__=='__main__':

    # set parameters
    args = args_parser()
    config = vars(args)
    
    dataset_name = config.get('dataset_name')
    input_type = config.get('input_type')
    prompt_type = config.get('prompt_type')
    property_name = config.get("property_name")
    model_name = config.get("model_name")
    data_path = config.get("data_path")
    results_path = config.get("results_path")
    min_samples = config.get("min_samples")

    extract_predictions(dataset_name, model_name, data_path, results_path, input_type, prompt_type, property_name, min_samples)
    print(f"Done!")