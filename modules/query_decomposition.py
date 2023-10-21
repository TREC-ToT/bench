import json


# Implement multiple ways of decomposing queries
# Each one should assume the input as the base query dictionary, and output a json with the format:
# {<query_id>: {<decomposed_query_1>: text, ..., <decomposed_query_n>: text}, ...}


def llm_based_decomposition():
    raise NotImplementedError()

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

