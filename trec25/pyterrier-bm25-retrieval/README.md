# PyTerrier BM25 Baseline for TREC-ToT 2025

work in progress

## Run it locally:

```
./baseline.py --output runs/bm25/train.run.txt.gz --index trec-tot-2025-pyterrier-index --dataset trec-tot/2025/train
./baseline.py --output runs/bm25/dev1.run.txt.gz --index trec-tot-2025-pyterrier-index --dataset trec-tot/2025/dev1
./baseline.py --output runs/bm25/dev2.run.txt.gz --index trec-tot-2025-pyterrier-index --dataset trec-tot/2025/dev2
./baseline.py --output runs/bm25/dev3.run.txt.gz --index trec-tot-2025-pyterrier-index --dataset trec-tot/2025/dev3
```


## Run with docker:

```
docker run --rm -ti -w /app -v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets -v ${PWD}:/app --entrypoint ./baseline.py mam10eks/trec-tot-pyterrier-baseline:dev-0.0.1 --output runs/bm25/train.run.txt.gz --index trec-tot-2025-pyterrier-index --dataset trec-tot/2025/train
```
