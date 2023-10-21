# Benchmarks for TREC-ToT (2023)

The following benchmarks (& runs) are available:


| Benchmark            | Runfiles | Dev-DCG | Dev-Success@1000 | Dev-MRR  |
|----------------------|----------|----------|-----------------|-------|
| [BM25](docs/BM25.md) (k1=0.8, b=1.0) |  [train](runs/bm25/train.run), [dev](runs/bm25/dev.run)     | 0.1314 |    0.4067 | 0.0881 |
| [Dense Retrieval (SBERT)](docs/DENSE.md) (Distilbert) |  [train](runs/distilbert/train.run), [dev](runs/distilbert/dev.run)  | 0.1627 |  0.6600  |  0.0743 |
| [GPT-4](docs/GPT4.md)* | [train](runs/gpt4/train.run), [dev](runs/gpt4/dev.run) | 0.2407 | 0.3200 | 0.2180 | 

*: GPT-4 generates 20 candidates at most. See [GPT4](GPT4.md) for more details.


## Initial setup 

```
## optional: create new environment using py-env virtual-env
## pyenv virtualenv 3.8.11 trec-tot-benchmarks
# install requirements 
pip install ir_datasets sentence-transformers==2.2.2 pyserini==0.20.0 pytrec_eval faiss-cpu==1.6.5 ranx==0.3.7
``` 

After downloading the files (see guidelines), set DATA_PATH to the folder which 
contains the uncompressed files s.t:
```
DATA_PATH/
  | train
  | | - queries.jsonl
  | |  - qrel.txt
  | dev 
  | | - queries.jsonl
  | | - qrel.txt
``` 


Quick test to see if data is setup properly:
```
python tot.py
```
The command above should print the correct number of train/dev queries and the number of documents 
in the corpus, along with example queries and documents.
