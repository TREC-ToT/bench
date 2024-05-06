import gzip
import json
import logging
import random
from collections import defaultdict
from itertools import islice
from typing import Dict, Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def read_jsonl(file):
    with open(file) as reader:
        jj = []
        for line in reader:
            j = json.loads(line)
            jj.append(j)

        return jj


def read_json(path, zipped=False):
    if zipped:
        with gzip.open(path, 'rt', encoding="ascii") as zipfile:
            return json.load(zipfile)
    else:
        with open(path, "r") as reader:
            return json.load(reader)


def aggregate_pytrec(results: Dict[str, Dict[str, float]], aggregate_method: str = "mean") -> Dict[str, Any]:
    if aggregate_method == "mean":
        # metric -> (mean, std)
        res_temp = aggregate_pytrec(results, "gather")
        final = {}
        for metric, values in res_temp.items():
            final[metric] = (np.mean(values), np.std(values))
        return final

    if aggregate_method == "gather":
        # metric -> [res_q1, res_q2, ...]
        final = defaultdict(list)
        for qid, qvals in results.items():
            for metric, met_val in qvals.items():
                final[metric].append(met_val)
        return final


def get_qrel(dataset, run):
    qrel = {}
    n_missing = 0
    for q in dataset.qrels_iter():
        if q.query_id not in run:
            if n_missing == 0:
                log.warning(f"Missing qids in run encountered! populating with empty result list")
            n_missing += 1
        qrel[q.query_id] = {q.doc_id: q.relevance}

    return qrel, n_missing


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def write_json(d, path, indent=None, zipped=False):
    if zipped:
        with gzip.open(path, 'wt', encoding="ascii") as zipfile:
            json.dump(d, zipfile, indent=indent)
    else:
        with open(path, "w") as writer:
            json.dump(d, writer, indent=indent)


def create_queries(dataset):
    queries = []
    n_empty = 0
    for q in dataset.queries_iter():
        text = q.query  # get_query(q, query_type)
        if text is None:
            if n_empty == 0:
                log.warning("encountered empty / null query")
            n_empty += 1
            continue
        queries.append((str(q.query_id), text))

    return queries, n_empty


def set_seed(seed, torch_deterministic=True, torch_benchmark=True):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = torch_benchmark
    np.random.seed(seed)
    random.seed(seed)
