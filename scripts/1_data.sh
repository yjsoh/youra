#!/bin/bash
# 1_data.sh - generate table 1

# Below script counts the number of tokens for each question in the dataset
# Generates a csv file with the count of tokens for each question
#
# Output files:
#   data/data.count_tokens.llama2.csv
#   data/data.count_tokens.llama3.csv
#   data/data.count_tokens.mistralai.csv
#
# Following files are used for paper table 1:
#   data/ntokens.llama2.csv
#   data/ntokens.llama3.csv
#   data/ntokens.mistralai.csv
python data.py  --config config/config.llama2.json count-tokens
python data.py  --config config/config.llama3.json count-tokens
python data.py  --config config/config.mistralai.json count-tokens

# Below script splits text into sentences using Stanza
# data.py script needs config file to be passed as argument
# does not need to run separately for each config file
# Output files:
#   data/data.target_sentences.stanza.json
python data.py  --config config/config.llama3.json get-target-sentences

# Below script runs the teaser experiment
# Result emphasizes how embedding-vector similarity could fall short by
# comparing the similarity of the question with a sample sentences
# No output files are generated. Just prints the results.
python data.py --config ./config/config.llama3.json teaser