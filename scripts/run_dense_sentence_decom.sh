#data_path=./data
data_path=/home/ddo/CMU/PLLM/TREC-TOT/sentence_decomp
append=sentence
split=train

python bm25.py --index_name bm25_0.8_1.1$append --index_path ./anserini_indicies \
               --data_path $data_path \
               --param_k1 0.8 --param_b 1.0 \
               --field text --query title_text --split $split \
               --run ./runs/bm25/$split$append.run --run_format trec_eval --run_id baseline_bm25_1$append \
               --negatives_out $data_path/negatives/bm25_negatives$append/$split-title_text-negatives.json

split=dev
python bm25.py --index_name bm25_0.8_1.1$append --index_path ./anserini_indicies \
               --data_path $data_path \
               --param_k1 0.8 --param_b 1.0 \
               --field text --query title_text --split $split \
               --run ./runs/bm25/$split$append.run --run_format trec_eval --run_id baseline_bm25_1$append \
               --negatives_out $data_path/negatives/bm25_negatives$append/$split-title_text-negatives.json



python train_dense.py --epochs 20 --lr 6e-05 --weight_decay 0.01 \
            --model_dir dense_models/baseline_distilbert_0$append \
            --run_id baseline_distilbert_0$append --model_or_checkpoint distilbert-base-uncased \
            --embed_size 768 --batch_size 10 --encode_batch_size 128 --data_path $data_path \
            --negatives_path $data_path/negatives/bm25_negatives$append \
            --negatives_out $data_path/negatives/baseline_distilbert_0_negatives$append \
            --query title_text \
            --device cuda

python train_dense.py --epochs 20 --lr 6e-05 --weight_decay 0.01 \
            --negatives_path $data_path/negatives/baseline_distilbert_0_negatives$append \
            --negatives_out $data_path/negatives/baseline_distilbert_1_negatives$append \
            --model_dir dense_models/baseline_distilbert$append \
            --run_id baseline_distilbert$append --model_or_checkpoint distilbert-base-uncased \
            --embed_size 768 --batch_size 10 --encode_batch_size 128 --data_path $data_path \
            --query title_text --device cuda


# After first run on Dense retrieval

# INFO:__main__:running & evaluating dev
# Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 38.78it/s]
# INFO:__main__:recip_rank  : 0.0308 (0.1310)
# INFO:__main__:P_1         : 0.0133 (0.1147)
# INFO:__main__:recall_10   : 0.0667 (0.2494)
# INFO:__main__:recall_100  : 0.1667 (0.3727)
# INFO:__main__:recall_1000 : 0.4200 (0.4936)
# INFO:__main__:ndcg_cut_10 : 0.0365 (0.1510)
# INFO:__main__:ndcg_cut_100: 0.0551 (0.1568)
# INFO:__main__:ndcg_cut_1000: 0.0856 (0.1550)



# After second run on Dense retrieval

# Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 38.96it/s]
# INFO:__main__:recip_rank  : 0.0298 (0.1236)
# INFO:__main__:P_1         : 0.0133 (0.1147)
# INFO:__main__:recall_10   : 0.0667 (0.2494)
# INFO:__main__:recall_100  : 0.2133 (0.4097)
# INFO:__main__:recall_1000 : 0.4200 (0.4936)
# INFO:__main__:ndcg_cut_10 : 0.0339 (0.1431)
# INFO:__main__:ndcg_cut_100: 0.0625 (0.1531)
# INFO:__main__:ndcg_cut_1000: 0.0879 (0.1510)
# INFO:__main__:running & evaluating train