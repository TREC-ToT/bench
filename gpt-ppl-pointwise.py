import openai
import numpy as np


import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from tqdm import tqdm
import os
import time
base = './data'


GPT_MODEL = "text-davinci-003"
openai.api_key = ""

def perplexity(text):
    """Compute the perplexity of the provided text."""
    completion = openai.Completion.create(
        model=GPT_MODEL,
        prompt=text,
        logprobs=0,
        max_tokens=0,
        temperature=1.0,
        echo=True)
    token_logprobs = completion['choices'][0]['logprobs']['token_logprobs']
    nll = np.mean([i for i in token_logprobs if i is not None])
    ppl = np.exp(-nll)
    return ppl



top_100 = {}
n=100
run_to_rerank = "./runs/distilbert/dev.run"
docs_needed = []
returned = []
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

#os.makedirs(f"./runs/{GPT_MODEL}rerank-distilbert/")

jsonObjdata = pd.read_json(f"/Users/aprameya/Downloads/corpus.jsonl", lines=True)
ans = []
count=0
corpus = dict()
for index, row in jsonObjdata.iterrows():
    if str(row["doc_id"]) in docs_needed:
        corpus[str(row["doc_id"])] = row["page_title"]

del jsonObjdata

jsonObj = pd.read_json(f"/Users/aprameya/Desktop/llms-project/data/queries.jsonl", lines=True)
ans = []
actual = []
count=0




done = 0
todo =  150

# os.makedirs(f"./runs/{GPT_MODEL}-rerank-distilbert/")
# print(top_100)

with open(f"./runs/gpt-4-rerank-distilbert/dev_pointwise_GPT-ppl.run", 'a') as out_file:
    for index,row in tqdm(jsonObj.iterrows(), total=todo):
        if done == todo:
            break 
        query_id = str(row['id'])

        query_top_100 = top_100[query_id][:100]

        small_id_mapper = {str(idx): str(x) for idx, x in enumerate(query_top_100)}
        inv_mapper = {v:k for k,v in small_id_mapper.items()}
        id_title_mapper = {corpus[str(x)]:str(x) for x in query_top_100}



        for doc in query_top_100:
            doc_title = corpus[str(doc)]
            
            # query = f"I will provide you with a single movie. Give  a score for the movie in the range of 1 through 10 upto 3 digits of precision based on its relevance to the user description. A score of 1 indicates that the movie title is not relevant to the user description, a score of 10 indicates that it is the correct match between title and description.\n\n\n"
            query = f"\nUser Description:\n{row['text']}\n"
            # query += f"Give a score to the movie movie based on its relevance to the search query. Use your best knowledge about the movie given only its title. The output format should be movie name::score. Only respond with the name and score do not say any word or explain."
            movie_string = " The title of the movie is: {}".format(doc_title)
            movie,score = doc_title,perplexity(query+movie_string)
            print(movie,score)
            returned.append((float(score),movie))

        returned.sort()
        print(id_title_mapper)

        for pos, rd in enumerate(returned):
            print(rd)
            out_file.write(f"{query_id} Q0 {id_title_mapper[rd[1]]} {pos} {1/(pos+1)} gpt3-rerank-distilbert-top100-pointwiseppl\n")
        returned = []
        
        done += 1