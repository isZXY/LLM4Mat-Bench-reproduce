#!/usr/bin/env bash

DATA_PATH='../data' # where LLM4Mat_Bench is saved
RESULTS_PATH='../results' # where to save the results
DATASET_NAME='cantor_hea' # any dataset name in LLM4Mat_Bench
INPUT_TYPE='formula' # other values: 'cif_structure' and 'description'
PROPERTY_NAME='e_above_hull' # any property name in $DATASET_NAME. Please check the property names associated with each dataset first
PROMPT_TYPE='zero_shot' # 'few_shot' can also be used here which let llama see five examples before it generates the answer
MIN_SAMPLES=2 # minimum number of valid outputs from llama (the default number is 10)
MODEL_NAME='Qwen3-8B'

python ../code/evaluate.py \
--data_path $DATA_PATH \
--results_path $RESULTS_PATH \
--dataset_name $DATASET_NAME \
--input_type $INPUT_TYPE \
--property_name $PROPERTY_NAME \
--prompt_type $PROMPT_TYPE \
--min_samples $MIN_SAMPLES \
--model_name $MODEL_NAME