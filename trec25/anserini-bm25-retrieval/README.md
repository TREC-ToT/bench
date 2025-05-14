# Attention:

**All of this is currently work in progress, we are in the steps of doing some final checks to verify that everything is ok**

# Anserini BM25 Baseline for TREC-ToT 2025

This directory contains a BM25 baseline implemented in [Anserini](https://github.com/castorini/anserini) for the 2025 edition of the [TREC Tip-of-the-Tongue (ToT) Track](https://trec-tot.github.io/). This baseline tracks the experiments in the [ir_metadata format](https://www.ir-metadata.org/) (including resource consumption for GPU/CPU/RAM and used energy) with the [TIREx tracker](https://github.com/tira-io/tirex-tracker).

## Existing Runs

The runs for all splits are available:

| ir_dataset          | run                                            |
|---------------------|------------------------------------------------|
| trec-tot/2025/train | [runs/train/run.txt.gz](runs/train/run.txt.gz) |
| trec-tot/2025/dev1  | [runs/dev1/run.txt.gz](runs/dev1/run.txt.gz)   |
| trec-tot/2025/dev2  | [runs/dev2/run.txt.gz](runs/dev2/run.txt.gz)   |
| trec-tot/2025/dev3  | [runs/dev3/run.txt.gz](runs/dev3/run.txt.gz)   |


## Existing Indices

A pre-built anserini index is available online so that you can make faster experimentation (**Attention:** we still verify the index and move it to Zenodo as soon as it is finalized.):

| Index | Size | md5 |
|-------|------|-----|
|[trec-tot-2025-anserini-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip) | 1.7GB | b04afdf33519013bf08857005a6cbd88|

You can download and extract this index if you want to re-run or modify this approach:

```
wget https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip
# md5 should be b04afdf33519013bf08857005a6cbd88
md5sum trec-tot-2025-anserini-index.zip
unzip trec-tot-2025-anserini-index.zip
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via (see the file [.devcontainer/Dockerfile.dev](.devcontainer/Dockerfile.dev) for details):

```
pip3 install -r requirements.txt
wget https://repo1.maven.org/maven2/io/anserini/anserini/1.0.0/anserini-1.0.0-fatjar.jar -O anserini.jar
```

## Retrieval

You can run retrieval for all datasets via (potentially download the index from above to speed this up):

```
./baseline.py --output runs/train/run.txt.gz --index trec-tot-2025-anserini-index --dataset trec-tot/2025/train
./baseline.py --output runs/dev1/run.txt.gz --index trec-tot-2025-anserini-index --dataset trec-tot/2025/dev1
./baseline.py --output runs/dev2/run.txt.gz --index trec-tot-2025-anserini-index --dataset trec-tot/2025/dev2
./baseline.py --output runs/dev3/run.txt.gz --index trec-tot-2025-anserini-index --dataset trec-tot/2025/dev3
```

## Retrieval with Docker

```
docker run --rm -ti -w /app -v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets -v ${PWD}:/app --entrypoint ./baseline.py mam10eks/trec-tot-anserini-baseline:dev-0.0.1 --output runs/train/run.txt --index trec-tot-2025-anserini-index --dataset trec-tot/2025/train
```
