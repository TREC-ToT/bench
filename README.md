# Benchmarks for TREC-ToT (2023)

The following benchmarks (& runs) are available. Results are for the dev2 set.:


| Benchmark            | Runfiles | NDCG@10 | NDCG@1000 |  MRR |R@1000  |
|----------------------|----------|----------|-----------------|-------|----|
| [BM25](BM25.md) (k1=1, b=1.0) |  [runs](runs/bm25/) | 0.0657  |0.1033| 0.0590 | 0.3600|
| [Dense Retrieval (SBERT)](DENSE.md) (DR) |  [runs](runs/DR/) | 0.1040 | 0.1665   | 0.0901  | 0.5600| 


**Note**: The current repository only supports the 2024 version of the corpus/queries. 
For using the 2023 version, refer to the [2023 release](https://github.com/TREC-ToT/bench/releases/tag/2023), use `tot23.py` instead, and change the `ir_dataset` names
used by baselines inside the code.  
 
## Initial setup 

```
## optional: create new environment using py-env virtual-env
## pyenv virtualenv 3.8.11 trec-tot-benchmarks
# install requirements 
pip install ir_datasets sentence-transformers==2.2.2 pyserini==0.20.0 pytrec_eval faiss-cpu==1.6.5
``` 

### 2024
After downloading the files (see guidelines), set DATA_PATH to the folder which 
contains the uncompressed files s.t:

```
DATA_PATH/
  | train-2024
  | | - queries.jsonl
  | |  - qrel.txt
  | dev1-2024
  | | - queries.jsonl
  | | - qrel.txt
  | dev2-2024
  | | - queries.jsonl
  | | - qrel.txt
  | corpus.jsonl
```

Quick test to see if data is setup properly:
```
python tot.py
```
The command above should print the correct number of train/dev queries and the number of documents 
in the corpus, along with example queries and documents.

### 2023 


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