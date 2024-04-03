# GPT-4 baseline

For this baseline, the following prompt was used:

```
You are an expert in movies. You are helping someone recollect a movie name that is on the tip of their tongue.
You respond to each message with a list of 20 guesses for the name of the movie being described.
**important**: you only mention the names of the movies, one per line, sorted by how likely they are the correct movie 
with the most likely correct movie first and the least likely movie last.
```

After we obtained the titles, we perform the following process: 
1. We utilize the titles of movies and their aliases, to match each generated title to a given document ID in our corpus.

2. If this is a direct match (i.e. single document associated with a given generated title), this is trivial.
 
3. Otherwise, we retrieve 5 candidates from a BM25 index, where each document in the index consists of all aliases of a movie.
 From this, we compute the levenstein distance between the generated title and the aliases. Inexact matches(~10%) are discarded. 

4. Since there are movies with the same name, we disambiguate with the dense retrieval run: if a given document (corresponding 
to a title) occurs in the run, it is ranked higher than ones that don't occur in the run.   Note that the generated run 
strictly obeys the order in which the model outputs titles, with the dense retrieval run breaking ties. Otherwise, titles
are assigned the same score.   
  
 


### Evaluation:
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


### Converting LLM output to a run

- ```pip install qwikidata```
- Create a JSONL file that matches the output in ./llm_example_runs/ex_out.jsonl
- Run: 
```
python gpt_post.py \
    --input llm_example_runs/ex_out.jsonl \
    --split train \
    --data_path ../trec-tot/datasets/TREC-TOT/public/ \
    --index_name llm_title \
    --run llm_example_runs/ex_out.run \  
    --run_id llm_example 
```
- **Recommended: ** run with aliases gathered from Wikidata (this takes a while)
```
python gpt_post.py \
    --input llm_example_runs/ex_out.jsonl \
    --split train \
    --data_path ../trec-tot/datasets/TREC-TOT/public/ \
    --index_name llm_title \
    --run llm_example_runs/ex_out.run \  
    --run_id llm_example \ 
    --gather_wikidata_aliases \
    --wikidata_cache ./wikidata_cache/ 
```