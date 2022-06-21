import unicodedata
import pandas as pd
import json
import sys #left in for now, will likely need eventually if the next step is cli
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import numpy as np
import math

if sys.argv < 2:
    sys.argv = ["recommender.py", 42]

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
for item in data_sorted["keyword"]:
    try:
        if "_etl" in item:
            item.remove("_etl")
        if "_jupyter" in item:
            item.remove("_jupyter")
    except TypeError:
        pass

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
for n in range(len(data_sorted)):
    try:
        sorted_keywords.append(' '.join(data_sorted.loc[n]["keyword"]))
    except TypeError: 
        sorted_keywords.append('')

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
        
# initializes tfidf vectorizer for descriptions and performs cosine similarity
tfidf = TfidfVectorizer(stop_words="english", max_df=.9)
tfidfmatrix = tfidf.fit_transform(sorted_desc)
cosine_sim = cosine_similarity(tfidfmatrix, tfidfmatrix)

# initializes count vectorizer for descriptions and performs cosine similarity
count = CountVectorizer(max_df = .9, stop_words="english")
count_matrix = count.fit_transform(sorted_desc)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# initializes tfidf vectorizer for keywords and performs cosine similarity
tfidf2 = TfidfVectorizer(stop_words="english", max_df=.9)
tfidfmatrix2 = tfidf2.fit_transform(sorted_keywords)
cosine_sim3 = cosine_similarity(tfidfmatrix2, tfidfmatrix2)

# initializes count vectorizer for keywords and performs cosine similarity
count2 = CountVectorizer(max_df = .9, stop_words="english")
count_matrix2 = count2.fit_transform(sorted_keywords)
cosine_sim4 = cosine_similarity(count_matrix2, count_matrix2)

# grabs result of cosine similarity for selected dataset, places results in new array, sorts it
new_array = cosine_sim[selected_dataset]
final_array = np.argsort(new_array)[::-1]
new_array2 = cosine_sim2[selected_dataset]
final_array2 = np.argsort(new_array2)[::-1]
new_array3 = cosine_sim3[selected_dataset]
final_array3 = np.argsort(new_array3)[::-1]
new_array4 = cosine_sim4[selected_dataset]
final_array4 = np.argsort(new_array4)[::-1]

# everything from here to the end is output related.
# mostly, it's temporary and just to check work/compare vectorizer results. 
# some may be used in the final code? 

#prints name, keyword, and description of the selected dataset
print("Title: ", data_sorted.loc[selected_dataset]["title"])
print("Keywords: ", data_sorted.loc[selected_dataset]["keyword"])
print("Description: ", data_sorted.loc[selected_dataset]["description"])


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
        
print("\nTFIDF -- Description")
tfidf_desc = recommendation_list(final_array, new_array)
    
print("\nCount -- Description")
count_desc = recommendation_list(final_array2, new_array2)
    

print("\nDatasets in both desc. recommender results")
for item_a in tfidf_desc:
    for item_b in count_desc:
        if item_a[1] == item_b[1]:
            print(item_a[0], item_a[1])

print("\nTFIDF -- Keywords")
tfidf_keyw = recommendation_list(final_array3, new_array3)

print("\nCount -- Keywords")
count_keyw = recommendation_list(final_array4, new_array4)

print("\nDatasets in both keyword recommender results")
for item_a in tfidf_keyw:
    for item_b in count_keyw:
        if item_a[1] == item_b[1]:
            print(item_a[0], item_a[1])
