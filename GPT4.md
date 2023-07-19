# GPT-4 baseline

The GPT-4 baseline generates 20 titles at most. Each generated title was matched to titles in the corpus, with multiple matches
possible. If a generated title was matched to multiple movies, then the dense retrieval run was used to re-order these movies.


Evaluation:
```
./trec_eval -m ndcg  -m recall.1,10,100,1000  -m recip_rank $DATA_PATH/dev/qrel.txt ./runs/gpt4/dev.run
# output:
recip_rank            	all	0.2180
recall_1              	all	0.1800
recall_10             	all	0.2867
recall_100            	all	0.3200
recall_1000           	all	0.3200
ndcg                  	all	0.2407
```