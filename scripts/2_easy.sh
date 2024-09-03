#!/bin/bash
# easy.sh - generates tables 4 and 5
# This should be run before 3_youra.sh, but after 1_data.sh

# Run easy algorithm for each model
# Outputs following files:
#   data/easy.llama2.json
#   data/easy.llama3.json
#   data/easy.mistralai.json
python easy.py --model_name "mistralai/Mistral-7B-Instruct-v0.2" --target_sentence_file ./data/data.target_sentences.stanza.json
python easy.py --model_name "meta-llama/Llama-2-7b-chat-hf" --target_sentence_file ./data/data.target_sentences.stanza.json
python easy.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --target_sentence_file ./data/data.target_sentences.stanza.json

# Process results
# Outputs following files:
#   results/table4.csv
#   results/table5.csv
python evaluate_easy.py easy

echo "[INFO] Results are in results/table4.csv and results/table5.csv"
echo "[INFO] You can now run 3_youra.sh"

echo "[INFO] Table 4:"
cat results/table4.csv

echo "[INFO] Table 5:"
cat results/table5.csv