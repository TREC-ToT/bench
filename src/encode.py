import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses

import ir_datasets

from src import data, utils
import logging

log = logging.getLogger(__name__)


def encode_dataset_faiss(model: SentenceTransformer, embedding_size: int, dataset: ir_datasets.Dataset, device,
                         encode_batch_size):
    doc_ids, documents = data.get_documents(dataset)

    idx_to_docid = {}
    docid_to_idx = {}
    for idx, doc_id in enumerate(doc_ids):
        idx_to_docid[idx] = doc_id
        docid_to_idx[doc_id] = idx

    model.eval()

    with torch.no_grad():
        embeddings = model.encode(documents,
                                  batch_size=encode_batch_size,
                                  show_progress_bar=True, device=device,
                                  convert_to_numpy=True)

    index = faiss.IndexFlatIP(embedding_size)
    indexwmap = faiss.IndexIDMap(index)
    indexwmap.add_with_ids(embeddings, np.arange(len(doc_ids)))

    return indexwmap, (idx_to_docid, docid_to_idx)


def create_run_faiss(model: SentenceTransformer, dataset: ir_datasets.Dataset, query_type, device, eval_batch_size,
                     index: faiss.IndexIDMap, idx_to_docid, docid_to_idx, top_k):
    model.eval()

    qids = []
    queries = []
    for query in dataset.queries_iter():
        queries.append(utils.get_query(query, query_type))
        qids.append(query.query_id)

    with torch.no_grad():
        query_embeddings = model.encode(queries, batch_size=eval_batch_size, show_progress_bar=True,
                                        convert_to_numpy=True, device=device)

    scores, raw_doc_ids = index.search(query_embeddings, k=top_k)
    run = {}
    for qid, sc, rdoc_ids in zip(qids, scores, raw_doc_ids):
        run[qid] = {}
        for s, rdid in zip(sc, rdoc_ids):
            if rdid == -1:
                log.warning(f"invalid doc ids!")
                continue
            run[qid][idx_to_docid[rdid]] = float(s)

    return run


def create_qrel(dataset, run=None):
    qrel = {}
    n_missing = 0
    for q in dataset.qrels_iter():
        if run and q.query_id not in run:
            n_missing += 1
        qrel[q.query_id] = {q.doc_id: q.relevance}

    return qrel, n_missing
