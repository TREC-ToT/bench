# run: /scripts/run_bm25.sh split
# split: train or dev
# train can be used to generate negatives.

split=$1
data_path=/home/ddo/CMU/PLLM/TREC-TOT

python bm25.py --index_name bm25_0.8_1.0 --index_path ./anserini_indicies \
               --data_path $data_path \
               --param_k1 0.8 --param_b 1.0 \
               --field text --query title_text --split $split \
               --run ./runs/bm25/$split.run --run_format trec_eval --run_id baseline_bm25 \
               --negatives_out $data_path/negatives/bm25_negatives/$split-title_text-negatives.json
