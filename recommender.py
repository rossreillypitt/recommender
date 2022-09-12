import unicodedata
import pandas as pd
import json
import sys #left in for now, will likely need eventually if the next step is cli
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from bs4 import BeautifulSoup
import numpy as np
import math

#if len(sys.argv) < 2:
sys.argv = ["recommender.py", 81]

selected_dataset = sys.argv[1]
print(selected_dataset)
# this is just to make it run for now. The number is 
# the index of a dataset in the list that's returned by
# the vectorizers. You can change this to test another dataset's recs.
# for now, can be any int from  - 329 inclusive. 

# pulls data from site, puts it in a dataframe
raw_data = []
response = requests.get("https://data.wprdc.org/data.json")
raw_data.append(json.loads(response.content))
raw_data_dict = raw_data[0]
raw_data_pd = pd.DataFrame.from_dict(raw_data_dict)

# isolates title, desc, and keywords from dataset, puts them in a dataframe
staging_list = []
for n in range(len(raw_data_pd)):
    staging_dict = {}
    staging_dict['title'] = raw_data_pd.loc[n]["dataset"]["title"]
    staging_dict.update({'description' : raw_data_pd.loc[n]["dataset"]["description"]})
    try:
        staging_dict.update({'keyword' : raw_data_pd.loc[n]["dataset"]["keyword"]})
    except KeyError: 
        pass
    staging_list.append(staging_dict)
data_df = pd.DataFrame(staging_list)

# sorts dataframe by title and resets index
data_sorted = data_df.sort_values(by=['title'])
data_sorted = data_sorted.reset_index(drop=True)

#removes technical keywords that were intended for internal use only 
for x in range(len(data_sorted["keyword"])):
    try:
        if "_etl" in data_sorted.loc[x]['keyword']:
            data_sorted.loc[x]['keyword'].remove("_etl")
        if "_jupyter" in data_sorted.loc[x]['keyword']:
            data_sorted.loc[x]['keyword'].remove("_jupyter")
        data_sorted.loc[x]['keyword'] = ' '.join(data_sorted.loc[x]['keyword'])
    except TypeError:
        data_sorted.loc[x]['keyword'] = ' '

# removes html, python escape characters, unicode from descriptions, 
sorted_desc = []
for n in range(len(data_sorted)):
    soup=BeautifulSoup(data_sorted.loc[n]["description"], 'html.parser')
    text = soup.get_text()
    text = text.replace("\n", "")
    text = text.replace("\r", " ")
    text = unicodedata.normalize('NFKD', text)
    sorted_desc.append(text)

# converts keyword list items into string to be read by vectorizers
sorted_keywords = []
for item in (data_sorted['keyword']):
    try:
        sorted_keywords.append(item)
    except TypeError: 
        sorted_keywords.append(' ')

sorted_title_str = []
for item in data_sorted["title"]:
    sorted_title_str.append(item)

#the following code is a bit circular. This is mostly for future use -- 
#it puts the selected dataset in the form of a string because I assume
#that'll be what is grabbed from the site when this is used. It then finds
#the index of the dataset with the same title. This will likely have to be tweaked.
selected_dataset = data_sorted.loc[selected_dataset]["title"]
for x in range(len(data_sorted["title"])):
    if selected_dataset == data_sorted.loc[x]["title"]:
        selected_dataset = x


column_transform = ColumnTransformer(
    [('title_output', TfidfVectorizer(stop_words="english", max_df=.9), 'title'),
     ('desc_output', TfidfVectorizer(stop_words="english", max_df=.9), 'description'),
     ('keyword_output', TfidfVectorizer(stop_words="english", max_df=.9), 'keyword'),])

column_transform.fit(data_sorted)
tfidf_matrix = column_transform.transform(data_sorted).toarray()

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
new_array = cosine_sim[selected_dataset]
final_array = np.argsort(new_array)[::-1]

def recommendation_list(final_array, new_array):
    x = 0
    list_x = []
    while len(list_x) < 5: 
        if final_array[x] == selected_dataset:
            x += 1
        else: 
            list_x.append([final_array[x], data_sorted.loc[final_array[x]]["title"], f'-- Score: {round(new_array[final_array[x]], 3)}'])
            x += 1
    for item in list_x:
        print(item[0], item[1], item[2])
    return list_x

#prints name, keyword, and description of the selected dataset
print("Title: ", data_sorted.loc[selected_dataset]["title"])
print("Keywords: ", data_sorted.loc[selected_dataset]["keyword"])
print("Description: ", data_sorted.loc[selected_dataset]["description"])

print("\nRelated Datasets and Scores")
count_keyw = recommendation_list(final_array, new_array)
