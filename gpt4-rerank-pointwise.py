# l = [(9.5, 'Apartment Eight'), (8.5, 'The Last Berliner'), (2.5, 'Hide and Seek (2014 film)'), (9.5, 'Block B (film)'), (2.5, 'Nousukausi'), (7.5, 'Highrise (documentary)'), (9.5, 'The New Tenants'), (3.5, 'The Other Barrio'), (9.5, 'Sunday (1997 film)'), (8.5, "Squatter's Delight"), (7.5, 'Master: A Building in Copacabana'), (2.5, 'Sous le même toit'), (8.5, 'Evixion'), (8.5, 'Exhibition (film)'), (1.0, 'Steve (film)'), (3.5, 'Battle for Brooklyn'), (2.5, 'Amexicano'), (9.5, 'Three Floors'), (1.0, 'Rent Control (1984 film)'), (2.5, 'Miracle on 42nd Street'), (9.5, 'The Cambridge Squatter'), (7.5, 'Everything Strange and New'), (7.5, 'Tom White (film)'), (7.5, 'Owners (film)'), (3.125, '8 Guys'), (1.0, 'Motherhood (2009 film)'), (7.5, 'The Edukators'), (9.5, '111 First Street (film)'), (2.5, 'Desperate Shareholders'), (2.5, 'Venussian Tabutasco'), (8.5, 'Behind Locked Doors (1991 film)'), (8.5, 'Autumn Ball'), (7.5, 'Boystown (film)'), (9.5, 'Unmade Beds (1997 film)'), (2.5, 'Steakhouse (film)'), (9.5, 'The Man Next Door (2010 film)'), (7.5, 'Accidence (film)'), (8.5, 'Chelsea Walls'), (7.5, 'Williamsburg (film)'), (1.0, 'Gypo (film)'), (7.5, 'The Exchange (2011 film)'), (8.5, 'Roshmia'), (8.5, 'Quality of Life (film)'), (8.5, 'Silent Nights (film)'), (8.5, 'Snapshots for Henry'), (9.5, 'La Soledad (film)'), (7.5, 'New Brooklyn'), (7.5, 'Gordonia (film)'), (9.5, 'The Architect (2006 film)'), (7.5, 'A Montreal Girl'), (2.5, 'The Roommates Party'), (8.5, 'Los Angeles Plays Itself'), (8.5, 'Meanwhile (2011 film)'), (7.5, 'Bedrooms (film)'), (9.5, 'At Home by Myself...With You'), (9.5, 'Food and Shelter'), (8.5, 'Beautiful Youth'), (9.5, 'The Street: A Film with the Homeless'), (1.0, 'Bonobo (2018 film)'), (7.5, 'Kill the Poor (film)'), (7.5, 'Tumbok'), (7.5, 'Temporary Family'), (3.5, 'Class Divide (film)'), (8.5, 'Something Like Happiness'), (8.5, 'Abroad'), (1.0, 'Flag Wars'), (9.5, "In Vanda's Room"), (7.5, 'Ara (film)'), (7.5, 'Breaking Upwards'), (9.5, 'The Tenants (2005 film)'), (1.0, 'Oben ohne (TV series)'), (1.0, 'Dilli (film)'), (8.5, 'Oblique (film)'), (7.5, 'FL 19,99'), (8.5, 'Quiet City (film)'), (7.5, 'See You in Hell, My Darling'), (7.5, 'Picture Paris'), (7.5, 'Metro cuadrado'), (2.5, 'Gendernauts'), (6.5, 'Havana, from on High'), (7.5, 'Life in Flight'), (8.5, 'Me and Her'), (7.5, 'Pulse: A Stomp Odyssey'), (6.5, 'Columbus Circle (film)'), (1.0, 'Hampstead (film)'), (7.5, 'Please Give'), (9.5, '24 City'), (8.5, 'The Lollipop Generation'), (8.5, 'Movement and Location'), (9.5, 'Rent Control (2005 film)'), (8.5, 'Down There (film)'), (7.5, '5 Flights Up'), (7.5, 'The Lips'), (8.5, 'The Earth Belongs to No One'), (9.5, 'On the Fringe (film)'), (7.5, 'For a Moment, Freedom'), (7.5, 'Dark Days (film)'), (9.5, 'Dream Home'), (9.5, 'Autumn in March'), (9.5, 'Cageman')]
# l.sort(reverse=True)
# for p,r in enumerate(l):
#     print(p)
#     print(r)

import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from tqdm import tqdm
import os
import time
base = './data'
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = ""

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

with open(f"./runs/gpt-4-rerank-distilbert/dev_pointwise_GPT-3-prompt.run", 'a') as out_file:
    for index,row in tqdm(jsonObj.iterrows(), total=todo):
        if done == todo:
            break 
        query_id = str(row['id'])

        query_top_100 = top_100[query_id][:100]

        small_id_mapper = {str(idx): str(x) for idx, x in enumerate(query_top_100)}
        inv_mapper = {v:k for k,v in small_id_mapper.items()}
        id_title_mapper = {corpus[str(x)]:str(x) for x in query_top_100}

        # print(small_id_mapper)
        # print(inv_mapper)


        for doc in query_top_100:
            # print(doc)
            # break
            doc_title = corpus[str(doc)]
            # small_id = inv_mapper[str(doc)]
            # query += f"[{small_id}] {doc_title}\n"
            # print(query)

            
            query = f"I will provide you with a single movie. Give  a score for the movie in the range of 1 through 10 upto 3 digits of precision based on its relevance to the user description. A score of 1 indicates that the movie title is not relevant to the user description, a score of 10 indicates that it is the correct match between title and description.\n\n\n"
            query += f"\nUser Description:\n{row['text']}\n"
            query += f"Give a score to the movie movie based on its relevance to the search query. Use your best knowledge about the movie given only its title. The output format should be movie name::score. Only respond with the name and score do not say any word or explain."
            movie_string = " The title of the movie is: {}".format(doc_title)
            # print(query+movie_string)
            response = openai.ChatCompletion.create(
                messages=[
                    {'role': 'system', 'content': 'You help someone give a rating for a movie with respect to relevancy towards a description.'},
                    {'role': 'user', 'content': query+movie_string},
                ],
                model=GPT_MODEL,
                temperature=0,
            )
            time.sleep(2)
            ranked_list = response['choices'][0]['message']['content']
            # print(ranked_list)
            movie,score = ranked_list.split("::")
            # print(movie,id_title_mapper[movie])
            returned.append((float(score),movie))



        # try:
        #     int_list = [int(part.strip("[] ")) for part in ranked_list.split(">") if part.strip("[] ").isdigit()]
        #     if any(number > 99 or number < 0 fo[r number in int_list):
        #         print(f"bad list for query {query_id}")
        #         int_list = list(small_id_mapper.keys())
        #     if len(int_list) > 100:
        #         print(f"bad list (too big) for query  {query_id}")
        #         int_list = list(small_id_mapper.keys())

        # except Exception as e:
        #     print(f"bad list for query {query_id}")
        #     int_list = list(small_id_mapper.keys())
        returned.sort(reverse=True)
        # doc_id_list = [small_id_mapper[str(small_id)] for small_id in int_list]


        for pos, rd in enumerate(returned):
            out_file.write(f"{query_id} Q0 {id_title_mapper[rd[1]]} {pos} {1/(pos+1)} gpt3-rerank-distilbert-top100-pointwiseprompt\n")
        returned = []
        
        done += 1