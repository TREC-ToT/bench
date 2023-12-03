import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from tqdm import tqdm
import os
import time
import pytrec_eval
import random
from collections import defaultdict
import numpy as np

base = './data'
GPT_MODEL = "gpt-4-1106-preview"
openai.api_key = ""
METRICS = "P_1,recall_10,recall_100,recall_1000,ndcg_cut_10,ndcg_cut_100,ndcg_cut_1000,recip_rank"
first_stage_retriever = "distilbert_pretrained_reddit_tomt_warmedup_finetuned_2"
run_to_rerank = f"/user/home/jcoelho/trec-tot/llms-project/dense_models/{first_stage_retriever}/dev.run"
qrels_path = '/user/home/jcoelho/trec-tot/llms-project/data/dev/qrel.txt'

shuffle_top_100 = True

def aggregate_pytrec(results, aggregate_method: str = "mean"):
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

######### INITIAL RESULTS

print("##### INITIAL RESULT")
with open(qrels_path, 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(f_qrel)

with open(run_to_rerank, 'r') as f_run:
    run = pytrec_eval.parse_run(f_run)

evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, METRICS.split(","))

eval_res = evaluator.evaluate(run)

eval_res_agg = aggregate_pytrec(eval_res, "mean")

for metric, (mean, std) in eval_res_agg.items():
    print(f"{metric:<12}: {mean:.4f} ({std:0.4f})")

##########################


top_100 = {}
n=100

docs_needed = []
with open(run_to_rerank, 'r') as h:
    for line in h:
        qid, _, did, _, _, _ = line.split()
        qid = str(qid)
        did = str(did)
        if qid not in top_100:
            top_100[qid] = [did]
            docs_needed.append(did)
        else:
            if len(top_100[qid]) < n:
                top_100[qid].append(did)
                docs_needed.append(did)

if shuffle_top_100:
    print("shuffling top-100")
    for qid in top_100:
        random.shuffle(top_100[qid])

jsonObjdata = pd.read_json(f"{base}/corpus.jsonl", lines=True)
ans = []
count=0
corpus = dict()
for index, row in jsonObjdata.iterrows():
    if str(row["doc_id"]) in docs_needed:
        corpus[str(row["doc_id"])] = row["page_title"]

del jsonObjdata

jsonObj = pd.read_json(path_or_buf=f"{base}/dev/queries.jsonl", lines=True)
ans = []
actual = []
count=0

def make_openai_request(query, retries=2):
    for attempt in range(retries + 1):
        try:
            response = openai.ChatCompletion.create(
                messages=[
                    {'role': 'system', 'content': 'You help someone re-rank a list of movies with respect to relevancy towards a description.'},
                    {'role': 'user', 'content': query},
                ],
                model=GPT_MODEL,
                temperature=0,
            )
            ranked_list = response['choices'][0]['message']['content']
            
            int_list = [int(part.strip("[] ")) for part in ranked_list.split(">") if part.strip("[] ").isdigit()]

            if any(number > 99 or number < 0 for number in int_list) or len(int_list) > 100 or len(int_list) != len(set(int_list)):

                if attempt < retries:
                    print("going for retry")
                    continue
                    
                else:
                    int_list = None
 
            return int_list
            
        except Exception as e:
            print("going for retry")
            time.sleep(2)

    return None

done = 0
todo =  150

reranked_run_path = f"./runs/{GPT_MODEL}-rerank-shuffled-{first_stage_retriever}"
if not os.path.exists(reranked_run_path):
    os.makedirs(reranked_run_path)

with open(f"{reranked_run_path}/dev.run", 'w') as out_file:
    for index,row in tqdm(jsonObj.iterrows(), total=todo):
        if done == todo:
            break 
        query_id = str(row['id'])

        query_top_100 = top_100[query_id][:100]

        small_id_mapper = {str(idx): str(x) for idx, x in enumerate(query_top_100)}
        inv_mapper = {v:k for k,v in small_id_mapper.items()}

        query = f"I will provide you with {n} movies, each indicated by a numerical identifier between [], e.g., [1], [2]. Rank the moves based on their relevance to the user description: {row['text']}.\n\n\n"
        
        for doc in query_top_100:
            doc_title = corpus[str(doc)]
            small_id = inv_mapper[str(doc)]
            query += f"[{small_id}] {doc_title}\n"

        query += f"\nUser Description:\n{row['text']}\n"
        query += f"Rank the {n} movies above based on their relevance to the search query. Use your best knowledge about the movie given their titles. All the movies should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain."

        int_list = make_openai_request(query)

        if not int_list:
            print("received default int_list")
            int_list = list(small_id_mapper.keys())
        
        doc_id_list = [small_id_mapper[str(small_id)] for small_id in int_list]

        for pos, rd in enumerate(doc_id_list):
            out_file.write(f"{query_id}\tQ0\t{rd}\t{pos}\t{1/(pos+1)}\tgpt4-rerank-distilbert-top100\n")
        
        done += 1


print("##### RESULT AFTER RR")

with open(qrels_path, 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(f_qrel)

with open(f"{reranked_run_path}/dev.run", 'r') as f_run:
    run = pytrec_eval.parse_run(f_run)

evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, METRICS.split(","))

eval_res = evaluator.evaluate(run)

eval_res_agg = aggregate_pytrec(eval_res, "mean")

for metric, (mean, std) in eval_res_agg.items():
    print(f"{metric:<12}: {mean:.4f} ({std:0.4f})")