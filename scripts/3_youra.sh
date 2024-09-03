#!/bin/bash
# 3_youra.sh - generate tables 2 and 3
# This should be run after 2_easy.sh

# Runs youra.py with Huggingface interface
# This will generate a folder with the invoked datetime.
# The folder will contain results.
# YYYY-MM-DD-HH-MM-SS/
#   config.json - the configuration file used
#   graded_{ds_name}.json - the graded dataset
#   summary_{ds_name}.json - the summary of the graded dataset
#   summary_{ds_name}.csv - the summary of the graded in csv format
#   stat_{ds_name}.json - the statistics of the graded dataset
#   results.csv
#   mean_all.csv
#   eval.jsonl
#   results.json - the results of the predictions
python ./youra.py --config ./config/config.llama2.json --q_index -1 youra
python ./youra.py --config ./config/config.llama3.json --q_index -1 youra
python ./youra.py --config ./config/config.mistralai.json --q_index -1 youra

# Genearte vLLM input
python ./youra.py --config ./config/config.llama2.json --q_index -1 --transparent_generate False youra
python ./youra.py --config ./config/config.llama3.json --q_index -1 --transparent_generate False youra
python ./youra.py --config ./config/config.mistralai.json --q_index -1 ----transparent_generate False youra

# Run vLLM predictions and save results
# Input:
#   data/vllm_input.{model_name}.{ds_name}.json
# Output:
#   results/{model_name}.youra.{ds_name}.json
ds_names=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique")
for ds_name in "${ds_names[@]}"; do
    python pred_vllm.py --dataset "$ds_name" --dataset-path "data/vllm_input.mistralai.$ds_name.json" --model "mistralai/Mistral-7B-Instruct-v0.2" --output-json "results/mistralai.youra.$ds_name.json"
    python pred_vllm.py --dataset "$ds_name" --dataset-path "data/vllm_input.llama2.$ds_name.json" --model "meta-llama/Llama-2-7b-chat-hf" --output-json "results/llama2.youra.$ds_name.json"
    python pred_vllm.py --dataset "$ds_name" --dataset-path "data/vllm_input.llama3.$ds_name.json" --model "meta-llama/Meta-Llama-3-8B-Instruct" --output-json "results/llama3.youra.$ds_name.json"
done

# gather results
python evaluate_youra.py

echo "[INFO] Results are in results/table2.csv and results/table3.csv"

echo "[INFO] Table 2:"
cat results/table2.csv

echo "[INFO] Table 3:"
cat results/table3.csv