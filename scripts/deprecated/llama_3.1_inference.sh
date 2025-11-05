#!/usr/bin/env bash

DATA_PATH='data' # where LLM4Mat_Bench is saved
RESULTS_PATH='../results_llama3.1/' # where to save the results
DATASET_NAME='mp' # any dataset name in LLM4Mat_Bench
INPUT_TYPE='formula' # other values: 'cif_structure' and 'description'
PROPERTY_NAME='band_gap' # any property name in $DATASET_NAME. Please check the property names associated with each dataset first Ef_per_atom
PROMPT_TYPE='zero_shot' # 'few_shot' can also be used here which let llama see five examples before it generates the answer
MAX_LEN=4000 # max_len and batch_size can be modified according to the available resources
BATCH_SIZE=128

python ../code/llama/llama_3.1_inference.py \
--data_path $DATA_PATH \
--results_path $RESULTS_PATH \
--dataset_name $DATASET_NAME \
--input_type $INPUT_TYPE \
--property_name $PROPERTY_NAME \
--max_len $MAX_LEN \
--prompt_type $PROMPT_TYPE \
--batch_size $BATCH_SIZE \