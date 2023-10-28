data_path=./data


python train_dense.py --epochs 20 --lr 6e-05 --weight_decay 0.01 \
            --model_dir dense_models/baseline_distilbert_0 \
            --run_id baseline_distilbert_0 --model_or_checkpoint distilbert-base-uncased \
            --embed_size 768 --batch_size 10 --encode_batch_size 128 --data_path $data_path \
            --negatives_path $data_path/negatives/bm25_negatives \
            --negatives_out $data_path/negatives/distilbert_negatives \
            --query title_text \
            --device mps