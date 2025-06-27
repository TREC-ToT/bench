# Dense Retrieval Baseline for TREC-ToT 2025

This directory contains a dense retrieval baseline using the model [sbhargav/baseline-distilbert-tot24](https://huggingface.co/sbhargav/baseline-distilbert-tot24) implemented in [Lightning IR](https://github.com/webis-de/lightning-ir) for the 2025 edition of the [TREC Tip-of-the-Tongue (ToT) Track](https://trec-tot.github.io/). This baseline tracks the experiments in the [ir_metadata format](https://www.ir-metadata.org/) (including resource consumption for GPU/CPU/RAM and used energy) with the [TIREx tracker](https://github.com/tira-io/tirex-tracker).

## Existing Runs

The runs for all splits are available:

| ir_dataset          | run                                            |
|---------------------|------------------------------------------------|
| trec-tot/2025/train | [runs/train/run.txt.gz](runs/train/run.txt.gz) |
| trec-tot/2025/dev1  | [runs/dev1/run.txt.gz](runs/dev1/run.txt.gz)   |
| trec-tot/2025/dev2  | [runs/dev2/run.txt.gz](runs/dev2/run.txt.gz)   |
| trec-tot/2025/dev3  | [runs/dev3/run.txt.gz](runs/dev3/run.txt.gz)   |


## Existing Indices

A pre-built Lighnting IR index is available online so that you can make faster experimentation:

| Index | Size | md5 |
|-------|------|-----|
|[trec-tot-2025-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-index.zip) | 17GB |  |

You can download and extract this index if you want to re-run or modify this approach:

```
wget https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-index.zip
# md5 should be 
md5sum trec-tot-2025-bert-bi-encoder-index.zip
unzip trec-tot-2025-bert-bi-encoder-index.zip
```

## Retrieval with Docker

```
docker run --rm -ti -w /app -v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets -v ${PWD}:/app --entrypoint ./baseline.py mam10eks/trec-tot-lightning-ir-baseline:dev-0.0.1 --output runs/train/run.txt --index trec-tot-2025-index --dataset trec-tot/2025/train
```