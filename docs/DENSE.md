# Dense Retrieval Baseline

This readme contains instructions for reproducing a Dense Retrieval baseline run using [Sentence Transformers](https://www.sbert.net/)
for the TREC track on tip-of-the-tongue (ToT)  retrieval. You can view [Guidelines](https://trec-tot.github.io/guidelines) for more details.


## Running the baseline

To run the baseline:

```
DATA_PATH= ## enter path to data ###
mkdir -p dense_models
python train_dense.py --epochs 20 --lr 6e-05 --weight_decay 0.01 \
            --model_dir dense_models/baseline_distilbert_0 \
            --run_id baseline_distilbert_0 --model_or_checkpoint distilbert-base-uncased \
            --embed_size 768 --batch_size 10 --encode_batch_size 128 --data_path $DATA_PATH \
            --negatives_out ./distilbert_negatives \
            --query title_text --device cuda 

# re-run with negatives generated from ^
python train_dense.py --epochs 20 --lr 6e-05 --weight_decay 0.01 \
            --negatives_path ./distilbert_negatives \
            --model_dir dense_models/baseline_distilbert \
            --run_id baseline_distilbert --model_or_checkpoint distilbert-base-uncased \
            --embed_size 768 --batch_size 10 --encode_batch_size 128 --data_path $DATA_PATH \
            --query title_text --device cuda 

```

You can evaluate (on the dev set) either using pytrec_eval (included in the script above), or use `trec_eval`:

```
 
./trec_eval -m ndcg  -m recall.1,10,100,1000  -m recip_rank $DATA_PATH/dev/qrel.txt ./dense_models/baseline_distilbert_0/dev.run
recip_rank            	all	0.0606
recall_1              	all	0.0267
recall_10             	all	0.1333
recall_100            	all	0.2733
recall_1000           	all	0.5333
ndcg                  	all	0.1313

# baseline run
./trec_eval -m ndcg  -m recall.1,10,100,1000  -m recip_rank $DATA_PATH/dev/qrel.txt ./dense_models/baseline_distilbert/dev.run
recip_rank            	all	0.0743
recall_1              	all	0.0400
recall_10             	all	0.1467
recall_100            	all	0.3600
recall_1000           	all	0.6600
ndcg                  	all	0.1627
```

Reproduced baselines

recip_rank  : 0.0831 (0.2316)
P_1         : 0.0533 (0.2247)
recall_10   : 0.1333 (0.3399)
recall_100  : 0.3400 (0.4737)
recall_1000 : 0.6267 (0.4837)
ndcg_cut_10 : 0.0878 (0.2481)
ndcg_cut_100: 0.1307 (0.2476)
ndcg_cut_1000: 0.1649 (0.2353)

  
