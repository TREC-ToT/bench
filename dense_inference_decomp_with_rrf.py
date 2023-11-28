import json
import argparse
import os
import pandas as pd
import logging


from tqdm import tqdm
from typing import Dict
from pyserini.search.lucene import LuceneSearcher
from pyserini.trectools import TrecRun
from pyserini.fusion import reciprocal_rank_fusion
from ranx import Qrels, Run, evaluate

from modules import llm_based_decomposition, sentence_decomposition
import tot
import ir_datasets
from src import utils
import pytrec_eval

METRICS = "P_1,recall_10,recall_100,recall_1000,ndcg_cut_10,ndcg_cut_100,ndcg_cut_1000,recip_rank"

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Path to indexes directory
    # parser.add_argument("--index_name", default="dense.index", help="name of index")
    parser.add_argument("--index_name", default="bm25_0.8_1.0", help="name of index")

    parser.add_argument("--decomposition_method", default="notllm", help="how to decompose")

    parser.add_argument("--data_path", default="/home/ddo/CMU/PLLM/TREC-TOT", help="location to dataset")

    parser.add_argument("--split", choices={"train", "dev", "test"}, default="dev", help="split to run")

    parser.add_argument("--index_path", default="./dense_index_hnsw/", help="path to store (all) indices")

    parser.add_argument("--metrics", default=METRICS, help="csv - metrics to evaluate")

    parser.add_argument("--param_k1", default=0.8, type=float, help="param: k1 for BM25")

    parser.add_argument("--param_b", default=1.0, type=float, help="param: b for BM25")

    # BM25 parameters
    parser.add_argument('--K', type=int, help='retrieve top K documents', default=1000)

    # Binary flags to enable or disable ranking methodss
    parser.add_argument('--rm3', type=str, help='enable or disable rm3', choices=['y', 'n'], default='n')
    
    # Run number
    parser.add_argument('--run_number', type=int, help='run number', default=1)

    # Output options and directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="runs_dense/")
    parser.add_argument('--load_queries', type=str, help='used previously decomposed queries', choices=['y', 'n'], default='y')
    args = parser.parse_args()

    tot.register(args.data_path)
    
    irds_name = "trec-tot:" + args.split
    dataset = ir_datasets.load(irds_name)
    if args.decomposition_method == "llm":
        if args.load_queries == 'n':
            queries_expanded = llm_based_decomposition(dataset, f"{args.data_path}/decomposed_queries")
        else:
            queries_expanded = f"{args.data_path}/decomposed_queries/llm_decomposed_queries.json"
    else:
        if args.load_queries == 'n':
            queries_expanded = sentence_decomposition(dataset, f"{args.data_path}/decomposed_queries")
        else: 
            queries_expanded = f"{args.data_path}/decomposed_queries/sentence_decomposed_queries.json"
    
    queries = json.load(open(queries_expanded))


    run_save_folder = f'{args.output_dir}DENSE-RRF-{args.split}'
    if args.decomposition_method == "llm":
        run_save_folder += f'-llm'

    run_save_folder += f'-RM3-{args.run_number}' if args.rm3 == 'y' else f'-{args.run_number}'
    run_save_folder += args.index_path.split('/')[-1]
    run_save_full = f"{run_save_folder}/{args.split}.run" 

    
    from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder, AutoQueryEncoder
    index_p  = args.index_path 
    # TODO: Put this in a args
    print(index_p)
    encoder = AutoQueryEncoder('/home/ddo/CMU/PLLM/llms-project/dense_models/baseline_distilbert/model')
    faiss_searcher = FaissSearcher(
        index_p,
        encoder
    )
    searcher = faiss_searcher
    
    # searcher = LuceneSearcher(os.path.join(args.index_path, args.index_name))
    # searcher.set_bm25(k1=args.param_k1, b=args.param_b)

    if args.rm3 == 'y':
        searcher.set_rm3()

    # Retrieve
    run_result = []

    for jj,query_id in tqdm(enumerate(queries)):
        for sintetic_query_id in queries[query_id]:
            hits = searcher.search(f'{queries[query_id][sintetic_query_id]}', k=args.K)
            sintetic_query_results = []
            for rank, hit in enumerate(hits, start=1):
                sintetic_query_results.append((query_id, 'Q0', hit.docid, rank, hit.score, f'{query_id}_{sintetic_query_id}'))
            
            if sintetic_query_results != []:
                run_result.append(TrecRun.from_list(sintetic_query_results))

    results = reciprocal_rank_fusion(run_result, depth=args.K, k=args.K)

    print(f"saving run to: {run_save_full}")
    os.makedirs(os.path.dirname(run_save_full), exist_ok=True)
    results.save_to_txt(run_save_full)

    if dataset.has_qrels():
        
        with open(run_save_full, 'r') as h:
            run_to_eval = pytrec_eval.parse_run(h)

        qrel, n_missing = utils.get_qrel(dataset, run_to_eval)
        if n_missing > 0:
            raise ValueError(f"Number of missing qids in run: {n_missing}")

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, args.metrics.split(","))

        eval_res = evaluator.evaluate(run_to_eval)

        eval_res_agg = utils.aggregate_pytrec(eval_res, "mean")

        for metric, (mean, std) in eval_res_agg.items():
            print(f"{metric:<12}: {mean:.4f} ({std:0.4f})")
    
if __name__ == '__main__':
    main()
