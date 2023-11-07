import ir_datasets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from src import data, encode, utils
import pytrec_eval
from torch import nn
import argparse
import os
import logging
from faiss import write_index, read_index
import json
import tot
import bm25

log = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("train_dense", description="Trains a dense retrieval model")

    parser.add_argument("--data_path", default="./datasets/TREC-TOT", help="location to dataset")

    parser.add_argument("--negatives_path", default="./bm25_negatives",
                        help="path to folder containing negatives ")

    parser.add_argument("--query", choices=["title", "text", "title_text"], default="title_text")

    parser.add_argument("--model_or_checkpoint", type=str, required=True, help="hf checkpoint/ path to pt-model")
    parser.add_argument("--embed_size", required=True, type=int, help="hidden size of the model")

    parser.add_argument("--epochs", type=int, required=True, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size (training)")
    parser.add_argument("--encode_batch_size", type=int, default=124, help="batch size (inference)")
    parser.add_argument("--evaluation_steps", type=int, default=-1, help="steps before evaluation is run")

    parser.add_argument("--freeze_base_model", action="store_true", default=False,
                        help="if set, freezes the base layer and trains only a projection layer on top")
    parser.add_argument("--metrics", required=False, default=bm25.METRICS, help="csv - metrics to evaluate")
    parser.add_argument("--n_hits", default=1000, type=int, help="number of hits to retrieve")

    parser.add_argument("--device", type=str, default="cuda", help="device to train /evaluate model on")

    parser.add_argument("--model_dir", type=str, help="folder to store model & runs", required=True)
    parser.add_argument("--run_id", required=True, help="run id (required if run_format = trec_eval)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives_out", default=None,
                        help="if provided, dumps negatives for use in training other models")
    parser.add_argument("--n_negatives", default=10, type=int, help="number of negatives to obtain")

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    utils.set_seed(args.seed)
    log.info(f"args: {args}")

    tot.register(args.data_path)
    metrics = args.metrics.split(",")

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    if args.freeze_base_model:
        base_model = SentenceTransformer(args.model_or_checkpoint, device=args.device)
        for param in base_model.parameters():
            param.requires_grad = False
        projection = models.Dense(args.embed_size, args.embed_size,
                                  activation_function=nn.Tanh())
        model = SentenceTransformer(modules=[base_model, projection], device=args.device)
    else:
        model = SentenceTransformer("/home/ddo/CMU/PLLM/llms-project/dense_models/baseline_distilbert/model", device="cuda")

        # model = SentenceTransformer(args.model_or_checkpoint, device=args.device)

    irds_splits = {}
    st_data = {}
    args.decomposition_method = "llm"
    if args.decomposition_method == "llm":
        queries_expanded = "/home/ddo/CMU/PLLM/TREC-TOT/decomposed_queries/llm_decomposed_queries.json"
    else:
        queries_expanded = "/home/ddo/CMU/PLLM/TREC-TOT/decomposed_queries/sentence_decomposed_queries.json"

    queries = json.load(open(queries_expanded))
    args.output_dir = "/home/ddo/CMU/PLLM/TREC-TOT/"
    run_save_folder = f'{args.output_dir}DENSE-RRF'
    if args.decomposition_method == "llm":
        run_save_folder += f'-llm'
    # splits
    args.run_number = 1
    args.rm3 = 'y'
    args.split = 'dev'
    run_save_folder += f'-RM3-{args.run_number}' if args.rm3 == 'y' else f'-{args.run_number}'
    
    run_save_full = f"{run_save_folder}/{args.split}.run"

    for split in {"train", "dev"}:
        irds_splits[split] = ir_datasets.load(f"trec-tot:{split}")

        log.info(f"loaded split {split}")
        st_data[split] = data.SBERTDataset(irds_splits[split], query_type=args.query,
                                           negatives=utils.read_json(
                                               os.path.join(args.negatives_path,
                                                            f"{split}-{args.query}-negatives.json")))

    log.info(f"training model for {args.epochs} epochs")
    train_dataloader = DataLoader(st_data["train"], shuffle=True, batch_size=args.batch_size)

    args.loss_fn = "mnrl"
    if args.loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        raise NotImplementedError(args.loss_fn)

    val_evaluator = data.get_ir_evaluator(st_data["dev"], name=f"dev",
                                          mrr_at_k=[1000],
                                          ndcg_at_k=[10, 1000],
                                          corpus_chunk_size=args.encode_batch_size)

    optimizer_params = {
        "lr": args.lr
    }

    # Tune the model
    """model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluation_steps=args.evaluation_steps,
              output_path=os.path.join(model_dir, "model"),
              evaluator=val_evaluator,
              epochs=args.epochs,
              warmup_steps=args.warmup_steps,
              optimizer_params=optimizer_params,
              weight_decay=args.weight_decay,
              save_best_model=True)"""

    log.info("encoding corpus with model")
    embed_size = args.embed_size
    # index, (idx_to_docid, docid_to_idx) = encode.encode_dataset_faiss(model, embedding_size=embed_size,
    #                                                                   dataset=irds_splits["train"],
    #                                                                   device=args.device,
    #                                                                   encode_batch_size=args.encode_batch_size)
    index = read_index( "dense.index")
    idx_to_docid = json.load(open("idx_to_docid.json","r"))
    docid_to_idx = json.load(open("docid_to_idx.json","r"))
    runs = {}
    eval_res_agg = {}
    eval_res = {}

    try:
        log.info("attempting to load test set")
        # plug in the test set
        irds_splits["test"] = ir_datasets.load(f"trec-tot:test")
        log.info("success!")
    except KeyError:
        log.info("couldn't find test set!")
        pass

    split_qrels = {}
    model.eval()

    qids = []
    # queries = []
    # for query in queries:
    #     queries.append(utils.get_query(query, query_type))
    #     qids.append(query.query_id)
    import torch
    eval_batch_size = 64
    device = 'cpu'
    # with torch.no_grad():
    #     query_embeddings = model.encode(queries, batch_size=eval_batch_size, show_progress_bar=True,
    #                                     convert_to_numpy=True, device=device)

    # top_k = 10
    # scores, raw_doc_ids = index.search(query_embeddings, k=top_k)
    # run = {}
    # for qid, sc, rdoc_ids in zip(qids, scores, raw_doc_ids):
    #     run[qid] = {}
    #     for s, rdid in zip(sc, rdoc_ids):
    #         if rdid == -1:
    #             log.warning(f"invalid doc ids!")
    #             continue
    #         run[qid][idx_to_docid[rdid]] = float(s)

    # return run
    # for split, dataset in irds_splits.items():
    #     log.info(f"running & evaluating {split}")

    #     run = encode.create_run_faiss(model=model,
    #                                   dataset=dataset,
    #                                   query_type=args.query, device=args.device,
    #                                   eval_batch_size=args.encode_batch_size,
    #                                   index=index, idx_to_docid=idx_to_docid,
    #                                   docid_to_idx=docid_to_idx,
    #                                   top_k=args.n_hits)
    #     runs[split] = run

    #     if dataset.has_qrels():
    #         qrel, n_missing = utils.get_qrel(dataset, run)
    #         split_qrels[split] = qrel
    #         evaluator = pytrec_eval.RelevanceEvaluator(
    #             qrel, metrics)

    #         eval_res[split] = evaluator.evaluate(run)
    #         eval_res_agg[split] = utils.aggregate_pytrec(eval_res[split], "mean")

    #         for metric, (mean, std) in eval_res_agg[split].items():
    #             log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")

    utils.write_json({
        "aggregated_result": eval_res_agg,
        "run": runs,
        "result": eval_res,
        "args": vars(args)
    }, os.path.join(model_dir, "out.gz"), zipped=True)

    run_id = args.run_id
    assert run_id is not None

    for split, run in runs.items():
        run_path = os.path.join(model_dir, f"{split}.run")
        with open(run_path, "w") as writer:
            for qid, r in run.items():
                for rank, (doc_id, score) in enumerate(sorted(r.items(), key=lambda _: -_[1])):
                    writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_id}\n")

    if args.negatives_out:
        log.info(f"writing negatives to folder: {args.negatives_out}")
        os.makedirs(args.negatives_out, exist_ok=True)
        out = {}

        for split, run in runs.items():
            if split == "test":
                continue
            negatives_path = os.path.join(args.negatives_out, f"{split}-{args.query}-negatives.json")
            qrel = split_qrels[split]
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
            utils.write_json(out, negatives_path)
