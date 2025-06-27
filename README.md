# Code and Baselines for TREC-ToT

The TREC-ToT dataset is integrated into [ir_datasets](https://github.com/allenai/ir_datasets) (you can install this via `pip3 install ir-datasets`) and we have baselines for [Anserini](trec25/anserini-bm25-retrieval), [PyTerrier](trec25/pyterrier-bm25-retrieval), and a [Dense Retrieval approach](lightning-dense-retrieval) that use this ir_datasets integration. The code and description for all baselines is available in [main/trec25](main/trec25).

We make the indices available for faster experimentation/modification of our baselines:

| Index | Framework |Size | md5 |
|-------|-----------|-----|-----|
|[trec-tot-2025-anserini-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip) | Anserini |1.7GB | b04afdf33519013bf08857005a6cbd88|
|[trec-tot-2025-pyterrier-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-pyterrier-index.zip) | PyTerrier | 11GB | a9a22ed35abb6cea842a7c5734987c82 |
|trec-tot-2025-dense-index|Lightning IR|**TODO**|**TODO**|

The following baselines and runs are available (more details available in [trec25/evaluation/evaluation-of-baselines.ipynb](trec25/evaluation/evaluation-of-baselines.ipynb)):

### Baselines for the training dataset:

| ir_dataset          |  Baseline                                                        | Runfiles | NDCG@10 | NDCG@1000 | R@1000  |
|---------------------|------------------------------------------------------------------|----------|-----------------|-------|----|
| trec-tot/2025/train | [BM25 (Anserini)](trec25/anserini-bm25-retrieval)                | [runs](trec25/anserini-bm25-retrieval/runs) | 0.022  | 0.055 | 0.280 |
| trec-tot/2025/train | [BM25 (PyTerrier)](trec25/pyterrier-bm25-retrieval)              | [runs](trec25/pyterrier-bm25-retrieval/runs)| 0.065 | 0.115 | 0.455 | 
| trec-tot/2025/train | [Dense Retrieval](trec25/lightning-dense-retrieval) | [runs](trec25/lightning-dense-retrieval/runs) | 0.318 | 0.373 | 0.755 |


### Baselines for the dev1 dataset:

| ir_dataset          |  Baseline                                                        | Runfiles | NDCG@10 | NDCG@1000 | R@1000  |
|---------------------|------------------------------------------------------------------|----------|-----------------|-------|----|
| trec-tot/2025/dev1 | [BM25 (Anserini)](trec25/anserini-bm25-retrieval)                | [runs](trec25/anserini-bm25-retrieval/runs) | 0.031 | 0.058 | 0.218 |
| trec-tot/2025/dev1 | [BM25 (PyTerrier)](trec25/pyterrier-bm25-retrieval)              | [runs](trec25/pyterrier-bm25-retrieval/runs)| 0.084 | 0.134 | 0.451 | 
| trec-tot/2025/dev1 | [Dense Retrieval](trec25/lightning-dense-retrieval) | [runs](trec25/lightning-dense-retrieval/runs) | 0.324 | 0.381 | 0.761 |


### Baselines for the dev2 dataset:

| ir_dataset          |  Baseline                                                        | Runfiles | NDCG@10 | NDCG@1000 | R@1000  |
|---------------------|------------------------------------------------------------------|----------|-----------------|-------|----|
| trec-tot/2025/dev2 | [BM25 (Anserini)](trec25/anserini-bm25-retrieval)                | [runs](trec25/anserini-bm25-retrieval/runs) | 0.043 | 0.072 | 0.252 |
| trec-tot/2025/dev2 | [BM25 (PyTerrier)](trec25/pyterrier-bm25-retrieval)              | [runs](trec25/pyterrier-bm25-retrieval/runs)| 0.099 | 0.143 | 0.455 | 
| trec-tot/2025/dev2 | [Dense Retrieval](trec25/lightning-dense-retrieval) | [runs](trec25/lightning-dense-retrieval/runs) | 0.020 | 0.050 | 0.245 |


### Baselines for the dev3 dataset:

| ir_dataset          |  Baseline                                                        | Runfiles | NDCG@10 | NDCG@1000 | R@1000  |
|---------------------|------------------------------------------------------------------|----------|-----------------|-------|----|
| trec-tot/2025/dev3 | [BM25 (Anserini)](trec25/anserini-bm25-retrieval)                | [runs](trec25/anserini-bm25-retrieval/runs) | 0.092 | 0.143 | 0.470 |
| trec-tot/2025/dev3 | [BM25 (PyTerrier)](trec25/pyterrier-bm25-retrieval)              | [runs](trec25/pyterrier-bm25-retrieval/runs)| 0.337 | 0.392 | 0.771 | 
| trec-tot/2025/dev3 | [Dense Retrieval](trec25/lightning-dense-retrieval) | [runs](trec25/lightning-dense-retrieval/runs) | 0.014 | 0.035 | 0.174 | 

**Note**: This repository hosts the code for processing the 2025 edition of the TREC ToT corpus in the [trec25 directory](/trec25). For processing older  versions, please refer to the [trec24 directory](/trec24) respectively the dedicated [2024 release](https://github.com/TREC-ToT/bench/releases/tag/2024) release respectively the dedicated [2023 release](https://github.com/TREC-ToT/bench/releases/tag/2023).

