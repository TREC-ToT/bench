import openai
import json
import re
import os

from tenacity import retry, stop_after_attempt, wait_random_exponential

# Implement multiple ways of decomposing queries
# Each one should assume the input as the base query dictionary, and output a json with the format:
# {<query_id>: {<decomposed_query_1>: text, ..., <decomposed_query_n>: text}, ...}

GPT_MODEL = "gpt-4"
openai.api_key = os.environ["OPENAI_API_KEY"]
pattern = re.compile(r"\d\.\s(.*)")
prompt = """
Title: {}
\nOriginal Query: {}
\nDecomposed Query: 
"""


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate(content):
    """
    """
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system',
             'content': 'You are a utility that decomposes complex user-formed descriptions of movies into smaller'
                        ' independent sentences that aid in movie name retrieval. Incorporate as much historical'
                        ' information in each sentence and lengthen them by appending synonyms and words at the'
                        ' end that may resemble the movie title and description and help in retrieval.'},
            {'role': 'user', 'content': content},
        ],
        model='gpt-4',
        temperature=0)
    return response['choices'][0]['message']['content']


def llm_based_decomposition(dataset, outpath):

    decomposed_queries = {}
    for q in dataset.queries_iter():
        original_query = []
        for annotation in q.sentence_annotations:
            if not annotation["labels"]["social"]:
                original_query.append(annotation["text"])
        original_query = " ".join(original_query)

        decomposed = generate(prompt.format(q.title, original_query))
        # decomposed = generate(prompt.format(q.text))
        subqueries = {}
        for i, sentence in enumerate(decomposed.split("\n")):
            try:
                sentence = pattern.match(sentence).groups()[0]
            except Exception as e:
                pass
            subqueries[i+1] = f"{q.title}. {sentence}"

        decomposed_queries[q.query_id] = subqueries

    queries_file = f"{outpath}/llm_decomposed_queries_no_social.json"

    with open(queries_file, 'w', encoding="utf-8") as fp:
        json.dump(decomposed_queries, fp)

    return queries_file


def sentence_decomposition(dataset, outpath):
    """Baseline decomposition method.
        Each query is decomposed into its sentences as provided in the data.
        Title of orignal query prepended to each decomposed subquery.
    """
    decomposed_queries = {}
    for q in dataset.queries_iter():
        subqueries = {}
        for sentence in q.sentence_annotations:
            subqueries[sentence['id']] = f"{q.title}. {sentence['text']}"
        
        decomposed_queries[q.query_id] = subqueries
    
    queries_file = f"{outpath}/sentence_decomposed_queries.json"

    with open(queries_file, 'w', encoding="utf-8") as fp:
        json.dump(decomposed_queries, fp)

    return queries_file

