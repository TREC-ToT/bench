import argparse
import os
import logging
from collections import Counter, defaultdict

import tot
import re
from bm25 import create_index, METRICS
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from thefuzz import fuzz, process
import pytrec_eval
import qwikidata
from datetime import datetime
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem

from src import utils
import json
import subprocess
import pandas as pd

log = logging.getLogger("gpt_post")


def create_title_index(dataset, dest_folder, index, gather_wikidata_aliases, wikidata_cache):
    log.info(f"creating files for indexing in {dest_folder}")
    docs_folder = os.path.join(dest_folder, "docs")
    os.makedirs(docs_folder, exist_ok=True)

    # get aliases
    aliases = {}

    if gather_wikidata_aliases:
        wikicache = WikiCache(wikidata_cache)

    log.info(f"gather_wikidata_aliases: {gather_wikidata_aliases}")
    for raw_doc in tqdm(dataset.docs_iter(), desc="gathering aliases"):
        aliases[raw_doc.doc_id] = {raw_doc.page_title}

        if gather_wikidata_aliases:
            went = wikicache.get(raw_doc.wikidata_id)
            if went:
                al = went["aliases"]
                if "en" in al:
                    for a in al["en"]:
                        aliases[raw_doc.doc_id].add(a["value"])

        # remove braces and add to aliases
        no_br = set()
        for _ in aliases[raw_doc.doc_id]:
            no_br.add(remove_braces(_))
        aliases[raw_doc.doc_id].update(no_br)

    with open(os.path.join(docs_folder, "docs.jsonl"), "w") as writer:
        for raw_doc in dataset.docs_iter():
            doc = {
                "id": raw_doc.doc_id,
                "contents": "\n".join(aliases[raw_doc.doc_id])
            }
            writer.write(json.dumps(doc) + "\n")

    # call pyserini indexer
    cmd = f"""python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input {docs_folder} \
      --index {index} \
      --generator DefaultLuceneDocumentGenerator \
      --keepStopwords \
      --stemmer none \
      --threads 1 \
      --storeRaw""".split()

    try:
        subprocess.call(cmd)
    except subprocess.CalledProcessError as e:
        log.exception("Exception occurred during indexing!")
        raise ValueError(e)

    return aliases


def remove_braces(text):
    return re.sub("[\(].*?[\)]", "", text).strip()


def remove_non_alpha(text):
    return re.sub(r'[\W\s]', ' ', text)


def resolve(title, matched_title, title_to_doc_id, aliases, scorer, assert_perfect_score=False):
    gen = []
    for doc_id in title_to_doc_id[matched_title]:
        # pick the best match
        best_match, score = process.extractOne(title, aliases[doc_id], scorer=scorer)

        if assert_perfect_score:
            # perfect match, this *has* to happen
            assert score == 100

        if score == 100:
            score = 101
        gen.append((best_match, doc_id, score))

    return gen


class WikiCache:
    def __init__(self, location="./wikidata_cache"):
        self.loc = location
        os.makedirs(location, exist_ok=True)

    def get(self, wid):
        cache_path = os.path.join(self.loc, wid)
        if os.path.exists(cache_path):
            return utils.read_json(cache_path)["entity"]

        try:
            ent = get_entity_dict_from_api(wid)
        except qwikidata.linked_data_interface.LdiResponseNotOk as e:
            log.exception(f"unable to find {wid}, skipping!")
            return None

        utils.write_json({
            "entity": ent,
            "retrieved_on": datetime.now().isoformat()}
            , cache_path)
        return ent

    def exists(self, wid):
        cache_path = os.path.join(self.loc, wid)
        if os.path.exists(cache_path):
            return True, utils.read_json(cache_path)["retrieved_on"]
        return False, None


if __name__ == '__main__':

    parser = argparse.ArgumentParser("gpt_post", description="post process outputfrom GPT, and compute run")
    parser.add_argument("--input", required=True, help="output from GPT (json)")

    parser.add_argument("--split", required=True, choices={"train", "dev", "test"}, help="corresponding split")
    parser.add_argument("--gather_wikidata_aliases", action="store_true", default=False,
                        help="if set, gathers aliases from Wikidata (recommended, takes time)")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--index_name", required=True, help="name of index")
    parser.add_argument("--run", required=True, help="path to save run")
    parser.add_argument("--run_format", default=None, choices={"trec_eval"})
    parser.add_argument("--run_id", required=True, help="run id (required if run_format = trec_eval)")
    parser.add_argument("--ref_run", default=None, help="if provided, this run is used to break ties")

    parser.add_argument("--metrics", required=False, default="recall_1,recall_10,recall_20,ndcg_cut_20,recip_rank",
                        help="csv - metrics to evaluate")
    parser.add_argument("--docs_path", default="./anserini_title_docs",
                        help="path to store (temp) documents for indexing")
    parser.add_argument("--index_path", default="./anserini_title_indices", help="path to store (all) indices")
    parser.add_argument("--param_k1", default=0.8, type=float, help="param: k1 for BM25")
    parser.add_argument("--param_b", default=1.0, type=float, help="param: b for BM25")
    parser.add_argument("--n_threads", default=8, type=int, help="number of threads (eval)")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size (eval) ")
    # /Users/sam/workspaces/tot-uncert/wikidata_cache
    parser.add_argument("--wikidata_cache", required=False, type=str)
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    if args.gather_wikidata_aliases:
        assert args.wikidata_cache is not None, "provide --wikidata_cache"
        os.makedirs(args.wikidata_cache, exist_ok=True)

    tot.register(args.data_path)
    split = args.split

    irds_name = "trec-tot:" + split
    dataset = ir_datasets.load(irds_name)

    args = parser.parse_args()
    docs_path = os.path.join(args.docs_path, args.index_name)
    index = os.path.join(args.index_path, args.index_name)
    AL_PATH = "./aliases.json"
    if not os.path.exists(index):
        log.info("Creating index!")
        aliases = create_title_index(dataset=dataset,
                                     dest_folder=docs_path,
                                     index=index,
                                     gather_wikidata_aliases=args.gather_wikidata_aliases,
                                     wikidata_cache=args.wikidata_cache)
        aliases = {k: list(v) for (k, v) in aliases.items()}
        utils.write_json(aliases, AL_PATH)
    else:
        aliases = utils.read_json(AL_PATH)
        log.info("index already created. loaded aliases")

    # title -> set{doc_id}
    title_to_doc_id = {}
    for doc_id, titles in tqdm(aliases.items(), leave=False):
        for al in aliases[doc_id]:
            if al in title_to_doc_id:
                title_to_doc_id[al].update([doc_id])
            else:
                title_to_doc_id[al] = {doc_id}

    # for d in dataset.docs_iter():
    #     doc_ids_to_title[d.doc_id] = d.page_title
    #     title_to_doc_id[d.page_title] = d.doc_id

    queries = utils.read_jsonl(args.input)

    searcher = LuceneSearcher(index)

    titles = []
    for query in queries:
        # gather the titles
        titles.extend(query["gpt_queries"])
    # dedup
    titles = list(set(titles))
    log.info(f"performing search on title index for {len(titles)} titles")

    # title -> [(title, doc_id, score))
    # score == 101 if it's a perfect match
    gen_title_to_doc_ids = {}
    matches = Counter()
    unmatched = set()
    unmatched_props = {}
    MIN_SCORE = 100
    BM25_K = 5

    scorer = fuzz.ratio
    for title in tqdm(titles):
        if title in title_to_doc_id:
            gen_title_to_doc_ids[title] = resolve(title=title,
                                                  matched_title=title,
                                                  title_to_doc_id=title_to_doc_id,
                                                  aliases=aliases,
                                                  scorer=scorer,
                                                  assert_perfect_score=True)

            if len(gen_title_to_doc_ids[title]) == 1:
                matches["exact_1"] += 1
            else:
                matches["exact_n"] += 1
        else:

            # no exact match, perform retrieval, followed by matching
            res = searcher.search(title, k=BM25_K)

            choices = []
            for _ in res:
                # get the closest alias
                best_match, score = process.extractOne(title, aliases[_.docid])
                choices.append(best_match)

            matched = process.extractOne(title, choices)
            if matched is None:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices
                }
                continue

            matched_title, score = matched
            if score >= MIN_SCORE:
                gen_title_to_doc_ids[title] = resolve(title=title,
                                                      matched_title=matched_title,
                                                      title_to_doc_id=title_to_doc_id,
                                                      aliases=aliases,
                                                      scorer=scorer,
                                                      assert_perfect_score=False)

                if len(gen_title_to_doc_ids[title]) == 1:
                    matches["inexact_1"] += 1
                    print(title, gen_title_to_doc_ids[title])
                else:
                    matches["inexact_n"] += 1

            ## try again after removing braces and non alpha numeric characters

            # we need to retain the original titles for mapping it back
            nobr2br = {remove_non_alpha(remove_braces(_)): _ for _ in choices}
            choices_nobr = list(nobr2br.keys())
            matched_nobr = process.extractOne(remove_non_alpha(remove_braces(title)), choices_nobr)
            if matched_nobr is None:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices,
                    "choices_nobr": choices_nobr,
                    "matched": matched
                }
                continue

            matched_title_nobr, score_nobr = matched_nobr
            if score_nobr >= MIN_SCORE:
                matched_org_title = nobr2br[matched_title_nobr]

                gen_title_to_doc_ids[title] = resolve(title=title,
                                                      matched_title=matched_org_title,
                                                      title_to_doc_id=title_to_doc_id,
                                                      aliases=aliases,
                                                      scorer=scorer,
                                                      assert_perfect_score=False)

            else:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices,
                    "choices_nobr": choices_nobr,
                    "matched": matched,
                    "matched_nobr": matched_nobr
                }
                continue

    print(matches)
    print(f"unmatched: {len(unmatched)}")
    rows = []

    for title in unmatched:
        row = {
            "title": title,
            "choices": ";".join(unmatched_props[title]["choices"]),
            "matched": unmatched_props[title].get("matched"),
            "matched_nobr": unmatched_props[title].get("matched_nobr"),
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(f"unmatched_{args.split}.csv", index=False)

    if args.ref_run:
        log.info(f"using reference run: {args.ref_run}")
        ref_run = defaultdict(dict)
        with open(args.ref_run) as reader:
            for line in reader:
                qid, _, doc_id, _, score, _ = line.split()
                ref_run[qid][doc_id] = float(score)
    else:
        log.info("no reference run provided!")
        ref_run = None

    # qid -> doc_id -> relevance
    run = {}
    # create run
    for query in queries:
        qid = query["id"]
        run[qid] = {}
        ranks = range(len(query["gpt_queries"]), 0, -1)
        for rank, title in zip(ranks, query["gpt_queries"]):
            gen_titles = gen_title_to_doc_ids.get(title, [])
            # no matches! :(
            if len(gen_titles) == 0:
                continue
            # single match, no problem!
            elif len(gen_titles) == 1:
                matched_title, doc_id, score = gen_titles[0]
                run[qid][doc_id] = float(rank)
            # if reference run isn't provided, assign the same
            # rank to each matched title
            elif ref_run is None:
                # assign same rank
                for (matched_title, doc_id, score) in gen_titles:
                    run[qid][doc_id] = float(rank)
            # otherwise re-order based on reference run
            else:
                ref_scores = {}
                rank = float(rank)
                for (matched_title, doc_id, score) in gen_titles:
                    if doc_id in ref_run[qid]:
                        ref_scores[doc_id] = ref_run[qid][doc_id]
                    else:
                        # those without ref scores get score = rank
                        run[qid][doc_id] = rank

                # those with ref scores gets score from (rank+step to rank+step*len(ref_scores))
                step = 1 / (len(ref_scores) + 1)
                for srank, (doc_id, _) in enumerate(sorted(ref_scores.items(), key=lambda _: _[1])):
                    rank += step
                    run[qid][doc_id] = rank

    if dataset.has_qrels():
        qrel, n_missing = utils.get_qrel(dataset, run)
        metrics = args.metrics.split(",")

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, metrics)
        eval_res = evaluator.evaluate(run)

        eval_res_agg = utils.aggregate_pytrec(eval_res, "mean")

        for metric, (mean, std) in eval_res_agg.items():
            log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")
    else:
        log.info("dataset has no qrels. no eval performed!")

    # write run file
    run_id = args.run_id
    with open(args.run, "w") as writer:
        for qid, r in run.items():
            for rank, (doc_id, score) in enumerate(sorted(r.items(), key=lambda _: -_[1])):
                writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{float(score)}\t{run_id}\n")
