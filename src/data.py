import logging
import os
import random

import ir_datasets
from sentence_transformers import InputExample, evaluation
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

log = logging.getLogger(__name__)


class SBERTDataset(Dataset):
    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, dataset: ir_datasets.Dataset, negatives, out_type, n_negatives):
        self.d = dataset
        self.out_type = out_type
        self.n_negatives = n_negatives

        # qid -> doc_id
        self.pos = {}
        # qid -> set of negatives
        self.neg = {}

        self.qid_to_query = {}
        for q in self.d.queries_iter():
            self.qid_to_query[q.query_id] = q

        docstore = self.d.docs_store()

        ## these are for IREvaluator
        # qid -> query
        self.queries = {}
        # qid -> set with relevant pids (size=1 in our case)
        self.ire_rel_docs = {}
        # pid -> passage
        self.corpus = {}

        # qrel: qid -> doc_id
        self.qrel = {}

        for qrel in self.d.qrels_iter():
            self.pos[qrel.query_id] = qrel.doc_id

            query = self.qid_to_query[qrel.query_id]
            text = query.query

            pos_doc = docstore.get(qrel.doc_id).text

            self.queries[qrel.query_id] = text
            self.qrel[qrel.query_id] = qrel.doc_id
            self.ire_rel_docs[qrel.query_id] = {qrel.doc_id}
            self.corpus[qrel.doc_id] = pos_doc

            self.neg[qrel.query_id] = []
            for neg_docid in negatives[qrel.query_id][:self.n_negatives]:
                neg_doc = docstore.get(neg_docid).text
                self.corpus[neg_docid] = neg_doc
                self.neg[qrel.query_id].append(neg_doc)

        if self.out_type == "triplet":
            self.idx_to_qid = {i: qid for (i, qid) in enumerate(self.qid_to_query)}
        elif self.out_type == "contrastive":
            # idx -> [(qid1, pos_1, +), (qid1, neg_1, -), (qid1, neg_2, -), ...]
            self.idx_to_qid = {}
            for qid in self.qid_to_query:
                self.idx_to_qid[len(self.idx_to_qid)] = (qid, self.corpus[self.qrel[qid]], self.POS_LABEL)
                for n in self.neg[qid]:
                    self.idx_to_qid[len(self.idx_to_qid)] = (qid, n, self.NEG_LABEL)
        else:
            raise NotImplementedError(self.out_type)

    def __getitem__(self, index) -> T_co:
        if self.out_type == "triplet":
            qid = self.idx_to_qid[index]
            query_text = self.queries[qid]
            pos_text = self.corpus[self.qrel[qid]]
            neg_text = random.choice(self.neg[qid])
            return InputExample(texts=[query_text, pos_text, neg_text])
        elif self.out_type == "contrastive":
            qid, text, label = self.idx_to_qid[index]
            return InputExample(texts=[self.queries[qid], text], label=label)
        else:
            raise NotImplementedError(self.out_type)

    def __len__(self):
        return len(self.idx_to_qid)


class SBERTDatasets(Dataset):
    """
    Wrapper for multiple SBERTDatasets
    """

    def __init__(self, datasets, negatives, out_type, n_negatives):
        self.n_datasets = len(datasets)
        self.datasets = [SBERTDataset(datasets[i], negatives[i],
                                      out_type, n_negatives) for i in range(self.n_datasets)]

        # idx -> (dataset, idx)
        self.idx_to_dset_idx = {}
        curr_idx = 0
        for d in self.datasets:
            for didx in range(len(d)):
                self.idx_to_dset_idx[curr_idx] = (d, didx)
                curr_idx += 1
        assert len(self) == curr_idx

    def __getitem__(self, index) -> T_co:
        dset, didx = self.idx_to_dset_idx[index]
        return dset[didx]

    def __len__(self):
        return sum(len(_) for _ in self.datasets)


def get_documents(dataset: ir_datasets.Dataset):
    documents = []
    doc_ids = []

    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        documents.append(doc.text)

    return doc_ids, documents


class IrEvaluator(evaluation.InformationRetrievalEvaluator):
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch,
                                                                                                                 steps)
        else:
            out_txt = ":"

        log.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]['accuracy@k'][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]['precision@k'][k])
                    output_data.append(scores[name]['recall@k'][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]['ndcg@k'][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]['map@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['ndcg@k'][max(self.ndcg_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]['ndcg@k'][max(self.ndcg_at_k)]


def get_ir_evaluator(dataset: SBERTDataset, name,
                     mrr_at_k=None,
                     ndcg_at_k=None,
                     corpus_chunk_size=100,
                     show_progress_bar=True):
    return IrEvaluator(dataset.queries, dataset.corpus, dataset.ire_rel_docs,
                       show_progress_bar=show_progress_bar,
                       accuracy_at_k=[1000],
                       precision_recall_at_k=[1000],
                       map_at_k=[1000],
                       mrr_at_k=mrr_at_k,
                       ndcg_at_k=ndcg_at_k,
                       corpus_chunk_size=corpus_chunk_size,
                       name=name)
