import logging

import faiss
import ir_datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm import tqdm, trange

from src import data

log = logging.getLogger(__name__)


def encode_dataset_faiss(model: SentenceTransformer, embedding_size: int, dataset: ir_datasets.Dataset, device,
                          encode_batch_size, normalize_embeddings=False):
    log.info("loading docs")
    doc_ids, documents = data.get_documents(dataset)

    idx_to_docid = {}
    docid_to_idx = {}
    for idx, doc_id in tqdm(enumerate(doc_ids), desc="pre-encode indexing"):
        idx_to_docid[idx] = doc_id
        docid_to_idx[doc_id] = idx
    log.info("loading docs complete")

    model = model.eval().to(device)

    all_embeddings = []
    for start_index in trange(0, len(documents), encode_batch_size, desc="Batches"):
        sentences_batch = documents[start_index:start_index + encode_batch_size]
        features = batch_to_device(model.tokenize(sentences_batch), device)

        with torch.no_grad():
            out_features = model.forward(features)
            embeddings = out_features["sentence_embedding"]
            embeddings = embeddings.detach()
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.cpu().numpy()

            all_embeddings.extend(embeddings)

    all_embeddings = np.asarray(all_embeddings)
    index = faiss.IndexFlatIP(embedding_size)
    indexwmap = faiss.IndexIDMap(index)
    indexwmap.add_with_ids(all_embeddings, np.arange(len(doc_ids)))
    return indexwmap, (idx_to_docid, docid_to_idx)


def create_run_faiss(model: SentenceTransformer, dataset: ir_datasets.Dataset, device, eval_batch_size,
                     index: faiss.IndexIDMap, idx_to_docid, docid_to_idx, top_k):
    model.eval()

    qids = []
    queries = []
    for query in dataset.queries_iter():
        queries.append(query.query)
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
