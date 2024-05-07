import argparse
import logging
import os

import ir_datasets
import pytrec_eval
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.losses import TripletDistanceMetric, SiameseDistanceMetric
from torch import nn
from torch.utils.data import DataLoader

import bm25
import tot
from src import data, encode, utils

log = logging.getLogger(__name__)

OUT_TYPES = {
    "mnrl": "triplet",
    "triplet": "triplet",
    "contrastive": "contrastive",
    "online_contrastive": "contrastive"
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser("train_dense", description="Trains a dense retrieval model")

    parser.add_argument("--data_path", default="./datasets/TREC-ToT2024/", help="location to dataset")

    parser.add_argument("--negatives_path", default="./bm25_negatives",
                        help="path to folder containing negatives ")

    parser.add_argument("--model_or_checkpoint", type=str, required=True, help="hf checkpoint/ path to pt-model")
    parser.add_argument("--embed_size", required=True, type=int, help="hidden size of the model")

    parser.add_argument("--epochs", type=int, required=True, help="number of epochs to train")
    parser.add_argument("--loss_fn", type=str, required=True, help="loss function")
    parser.add_argument("--loss_distance", type=str, default=None, help="distance function for loss [only some losses]")
    parser.add_argument("--loss_margin", type=str, default=None, help="margin for loss [only some losses]")
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
    parser.add_argument("--n_train_negatives", default=30, type=int,
                        help="number of negatives to use during training")
    parser.add_argument("--n_negatives", default=30, type=int, help="number of negatives to obtain")
    parser.add_argument("--encode_after_train", action="store_true", default=False, help="encode & run after training ")
    parser.add_argument("--encode_norm", action="store_true", default=False, help="normalize embeds")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s')

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
        model = SentenceTransformer(args.model_or_checkpoint, device=args.device)

    # load the negatives
    negatives = {}
    for split in {"train-2024", "dev1-2024", "dev2-2024"}:
        neg_name = split.split("-")[0]
        negatives[split] = utils.read_json(os.path.join(args.negatives_path, f"{neg_name}-negatives.json"))

    out_type = OUT_TYPES[args.loss_fn]
    log.info(f"output type for datasets: {out_type}, loss={args.loss_fn}")
    irds_splits = {}
    st_data = {}
    # splits
    for split in {"train-2024", "dev1-2024", "dev2-2024"}:
        irds_splits[split] = ir_datasets.load(f"trec-tot:{split}")
        log.info(f"loaded split {split}")
        st_data[split] = data.SBERTDataset(irds_splits[split],
                                           negatives=negatives[split],
                                           out_type=out_type,
                                           n_negatives=args.n_train_negatives)

    # create a new dataset with train + dev1
    train_data = data.SBERTDatasets(
        [irds_splits["train-2024"], irds_splits["dev1-2024"]],
        [negatives["train-2024"], negatives["dev1-2024"]],
        out_type=out_type,
        n_negatives=args.n_train_negatives
    )

    log.info(f"training model for {args.epochs} epochs [train_len={len(train_data)}]")
    train_dataloader = DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=args.batch_size)

    args.loss_fn = "mnrl"
    if args.loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    elif args.loss_fn == "triplet":
        assert args.loss_margin is not None
        loss_distances = {
            "cosine": TripletDistanceMetric.COSINE,
            "euclidean": TripletDistanceMetric.EUCLIDEAN
        }
        assert args.loss_distance is not None and args.loss_distance in loss_distances
        train_loss = losses.TripletLoss(model=model,
                                        distance_metric=args.loss_distance,
                                        triplet_margin=args.loss_margin)
    elif args.loss_fn == "contrastive":
        assert args.loss_margin is not None
        loss_distances = {
            "cosine": SiameseDistanceMetric.COSINE,
            "euclidean": SiameseDistanceMetric.EUCLIDEAN
        }
        assert args.loss_distance is not None and args.loss_distance in loss_distances
        train_loss = losses.ContrastiveLoss(model=model,
                                            distance_metric=args.loss_distance,
                                            margin=args.loss_margin)
    elif args.loss_fn == "online_contrastive":
        assert args.loss_margin is not None
        loss_distances = {
            "cosine": SiameseDistanceMetric.COSINE,
            "euclidean": SiameseDistanceMetric.EUCLIDEAN
        }
        assert args.loss_distance is not None and args.loss_distance in loss_distances
        train_loss = losses.OnlineContrastiveLoss(model=model,
                                                  distance_metric=args.loss_distance,
                                                  margin=args.loss_margin)
    else:
        raise NotImplementedError(args.loss_fn)

    val_evaluator = data.get_ir_evaluator(st_data["dev2-2024"], name=f"dev2",
                                          mrr_at_k=[1000],
                                          ndcg_at_k=[10, 1000],
                                          corpus_chunk_size=args.encode_batch_size)

    optimizer_params = {
        "lr": args.lr
    }

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluation_steps=args.evaluation_steps,
              output_path=os.path.join(model_dir, "model"),
              evaluator=val_evaluator,
              epochs=args.epochs,
              warmup_steps=args.warmup_steps,
              optimizer_params=optimizer_params,
              weight_decay=args.weight_decay,
              save_best_model=True)

    if args.encode_after_train:
        run_id = args.run_id
        assert run_id is not None
        try:
            log.info("attempting to load test set")
            # plug in the test set
            irds_splits["test"] = ir_datasets.load(f"trec-tot:test-2024")
            log.info("success!")
        except KeyError:
            log.info("couldn't find test set!")
        log.info("encoding corpus with model")
        embed_size = args.embed_size
        index, (idx_to_docid, docid_to_idx) = encode.encode_dataset_faiss(model, embedding_size=embed_size,
                                                                          dataset=irds_splits["train-2024"],
                                                                          device=args.device,
                                                                          encode_batch_size=args.encode_batch_size,
                                                                          normalize_embeddings=args.encode_norm)

        runs = {}
        eval_res_agg = {}
        eval_res = {}
        split_qrels = {}
        for split, dataset in irds_splits.items():
            log.info(f"running & evaluating {split}")

            run = encode.create_run_faiss(model=model,
                                          dataset=dataset,
                                          device=args.device,
                                          eval_batch_size=args.encode_batch_size,
                                          index=index, idx_to_docid=idx_to_docid,
                                          docid_to_idx=docid_to_idx,
                                          top_k=args.n_hits)
            runs[split] = run

            if dataset.has_qrels():
                qrel, n_missing = utils.get_qrel(dataset, run)
                split_qrels[split] = qrel
                evaluator = pytrec_eval.RelevanceEvaluator(
                    qrel, metrics)

                eval_res[split] = evaluator.evaluate(run)
                eval_res_agg[split] = utils.aggregate_pytrec(eval_res[split], "mean")

                for metric, (mean, std) in eval_res_agg[split].items():
                    log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")

        utils.write_json({
            "aggregated_result": eval_res_agg,
            "run": runs,
            "result": eval_res,
            "args": vars(args)
        }, os.path.join(model_dir, "out.gz"), zipped=True)

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
                if "test" in split:
                    continue
                negatives_path = os.path.join(args.negatives_out, f"{split}-negatives.json")
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
    else:
        log.info("encode_after_train not set. complete!")
