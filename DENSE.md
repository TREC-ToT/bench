# Dense Retrieval Baseline

This readme contains instructions for reproducing a Dense Retrieval baseline run using [Sentence Transformers](https://www.sbert.net/)
for the TREC track on tip-of-the-tongue (ToT)  retrieval. You can view [Guidelines](https://trec-tot.github.io/guidelines) for more details..

Note that the script below trains it on both the dev1 and train splits. 

```
# step 1: train a DR model and generate negatives
python train_dense.py \
--model_dir dense_models/baseline_distilbert_0/ \
--data_path $DATA_PATH \
--encode_after_train \
--epochs 20 --loss_margin 0.75 --lr 6e-05 --n_train_negatives 5  \
--run_id baseline_distilbert_0 --device cuda --weight_decay 0.01 \
--model_or_checkpoint distilbert-base-uncased --embed_size 768 \
--encode_batch_size 128 --batch_size 10 --loss_fn triplet \
--loss_distance cosine --encode_norm \
--negatives_out distilbert_negatives  >> distilbert.log 2>&1 &

# step 2: load the new negatives and retrain
python train_dense.py \
--model_dir dense_models/baseline_distilbert/ \
--negatives_path distilbert_negatives \
--data_path $DATA_PATH \
--encode_after_train \
--epochs 20 --loss_margin 0.75 --lr 6e-05 --n_train_negatives 5  \
--run_id baseline_distilbert --device cuda --weight_decay 0.01 \
--model_or_checkpoint distilbert-base-uncased --embed_size 768 \
--encode_batch_size 128 --batch_size 10 --loss_fn triplet \
--loss_distance cosine --encode_norm  >> distilbert.log 2>&1 &
```

You can evaluate either using pytrec_eval (included in the script above), or use `trec_eval`:
```
trec_eval -m ndcg_cut.10,1000 -m recall.1000  -m recip_rank $DATA_PATH/dev1-2024/qrel.txt ./dense_models/baseline_distilbert/dev1.run
recip_rank            	all	0.0901
recall_1000           	all	0.5600
ndcg_cut_10           	all	0.1040
ndcg_cut_1000         	all	0.1665

```

  
