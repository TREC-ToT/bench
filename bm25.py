import os
import json
import logging
import argparse
import subprocess

from typing import Dict, Any
from collections import defaultdict

import ir_datasets
import pytrec_eval
import numpy as np
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from src import utils
import tot

log = logging.getLogger(__name__)

METRICS = "recall_10,recall_100,recall_1000,ndcg_cut_10,ndcg_cut_100,ndcg_cut_1000,recip_rank"


def create_index(dataset, field_to_index, dest_folder, index):
    log.info(f"creating files for indexing in {dest_folder}")
    docs_folder = os.path.join(dest_folder, "docs")
    os.makedirs(docs_folder, exist_ok=True)

    with open(os.path.join(docs_folder, "docs.jsonl"), "w") as writer:
        for raw_doc in dataset.docs_iter():
            doc = {
                "id": raw_doc.doc_id,
                "contents": getattr(raw_doc, field_to_index)
            }
            writer.write(json.dumps(doc) + "\n")

    # call pyserini indexer
    cmd = f"""python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input {docs_folder} \
      --index {index} \
      --generator DefaultLuceneDocumentGenerator \
      --threads 1 \
      --storePositions --storeDocvectors --storeRaw""".split()

    try:
        subprocess.call(cmd)
    except subprocess.CalledProcessError as e:
        log.exception("Exception occurred during indexing!")
        raise ValueError(e)


def create_run(index, queries, param_k1, param_b, batch_size, n_hits, n_threads):
    run = {}
    searcher = LuceneSearcher(index)
    searcher.set_bm25(k1=param_k1, b=param_b)
    batches = list(utils.batched(queries, batch_size))
    for batch in tqdm(batches):
        batch_qids, batch_queries = [_[0] for _ in batch], [_[1] for _ in batch]
        results = searcher.batch_search(batch_queries, batch_qids, k=n_hits, threads=n_threads)
        for qid, hits in results.items():
            assert qid not in run
            run[qid] = {}
            for i in range(len(hits)):
                run[qid][hits[i].docid] = hits[i].score
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser("BM25 Run")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--split", required=True, choices={"train", "dev", "test"}, help="split to run")
    parser.add_argument("--field", required=True, help="field to index from documents")
    parser.add_argument("--index_name", required=True, help="name of index")
    parser.add_argument("--query", choices=["title", "text", "title_text"], required=True)

    parser.add_argument("--param_k1", default=0.8, type=float, help="param: k1 for BM25")
    parser.add_argument("--param_b", default=1.0, type=float, help="param: b for BM25")

    parser.add_argument("--run", default=None, help="(optional) path to save run")
    parser.add_argument("--run_format", default=None, choices={"trec_eval", "json"},
                        help="(optional) path to save run, defaults to json if json is in file name")
    parser.add_argument("--run_id", default=None, help="run id (required if run_format = trec_eval)")

    parser.add_argument("--metrics", required=False, default=METRICS,
                        help="csv - metrics to evaluate")
    parser.add_argument("--docs_path", default="./anserini_docs", help="path to store (temp) documents for indexing")
    parser.add_argument("--index_path", default="./anserini_indices", help="path to store (all) indices")
    parser.add_argument("--n_hits", default=1000, type=int, help="number of hits to retrieve")
    parser.add_argument("--n_threads", default=8, type=int, help="number of threads (eval)")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size (eval) ")
    parser.add_argument("--negatives_out", default=None,
                        help="if provided, dumps negatives for use in training other models")
    parser.add_argument("--n_negatives", default=10, type=int,
                        help="number of negatives to obtain")

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    tot.register(args.data_path)
    split = args.split

    irds_name = "trec-tot:" + split
    dataset = ir_datasets.load(irds_name)

    metrics = args.metrics.split(",")

    log.info(f"metrics: {metrics}")

    docs_path = os.path.join(args.docs_path, args.index_name)
    index = os.path.join(args.index_path, args.index_name)
    if os.path.exists(index):
        log.warning(f"Index {index} already exists!")
    else:
        log.info("Creating index!")
        create_index(dataset=dataset,
                     field_to_index=args.field,
                     dest_folder=docs_path,
                     index=index)

    log.info(f"BM25 config: k1={args.param_k1}; b={args.param_b}")

    queries, n_empty = utils.create_queries(dataset, query_type=args.query)

    log.info(f"Gathered {len(queries)} queries")
    if n_empty > 0:
        log.warning(f"Number of empty queries: {n_empty}")

    run = create_run(index=index, queries=queries, param_b=args.param_b,
                     param_k1=args.param_k1, batch_size=args.batch_size,
                     n_hits=args.n_hits, n_threads=args.n_threads)

    if dataset.has_qrels():
        qrel, n_missing = utils.get_qrel(dataset, run)
        if n_missing > 0:
            raise ValueError(f"Number of missing qids in run: {n_missing}")

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, metrics)

        eval_res = evaluator.evaluate(run)

        eval_res_agg = utils.aggregate_pytrec(eval_res, "mean")

        for metric, (mean, std) in eval_res_agg.items():
            log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")

    else:
        log.info("dataset does not have qrels. evaluation not performed!")
        eval_res_agg = None
        eval_res = None
        qrel = None

    if args.run is not None:
        run_format = args.run_format
        if run_format is None:
            run_format = "json" if "json" in args.run else "trec_eval"
        log.info(f"Saving run to {args.run} (format={run_format})")
        if run_format == "json":
            utils.write_json({
                "aggregated_result": eval_res_agg,
                "run": run,
                "result": eval_res,
                "args": vars(args)
            }, args.run, zipped=args.run.endswith(".gz"))
        else:
            run_id = args.run_id
            assert run_id is not None
            with open(args.run, "w") as writer:
                for qid, r in run.items():
                    for rank, (doc_id, score) in enumerate(sorted(r.items(), key=lambda _: -_[1])):
                        writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_id}\n")

        if args.negatives_out:
            out = {}
            for qid, hits in run.items():
                hits = sorted(hits.items(), key=lambda _: -_[1])
                negs = []
                for (doc, score) in hits:
                    if qrel[qid].get(doc, 0) > 0:
                        continue
                    if len(negs) == args.n_negatives:
                        break
                    negs.append(doc)

                out[qid] = negs

            utils.write_json(out, args.negatives_out)
