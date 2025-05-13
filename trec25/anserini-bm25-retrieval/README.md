work in progress


```
docker run --rm -ti -w /app -v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets -v ${PWD}:/app --entrypoint ./baseline.py mam10eks/trec-tot-anserini-baseline:dev-0.0.1 --output runs/train/run.txt --index trec-tot-2025-anserini-index --dataset trec-tot/2025/train
```

PYTHONPATH=ir_datasets ./baeline --output runs/train/run.txt.gz --index trec-tot-2025-anserini-index --dataset trec-tot/2025/train