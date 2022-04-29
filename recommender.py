#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

selected_dataset = 15 
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
final_array = np.argsort(new_array[::-1])
new_array2 = cosine_sim2[selected_dataset]
final_array2 = np.argsort(new_array2[::-1])
new_array3 = cosine_sim3[selected_dataset]
final_array3 = np.argsort(new_array3[::-1])
new_array4 = cosine_sim4[selected_dataset]
final_array4 = np.argsort(new_array4[::-1])

# everything from here to the end is output related.
# mostly, it's temporary and just to check work/compare vectorizer results. 
# some may be used in the final code? 

#prints name, keyword, and description of the selected dataset
print("Title: ", data_sorted.loc[selected_dataset]["title"])
print("Keywords: ", data_sorted.loc[selected_dataset]["keyword"])
print("Description: ", data_sorted.loc[selected_dataset]["description"])

print("\nTFIDF -- Description")
for n in range(5):
    print(final_array[n], data_sorted.loc[final_array[n]]["title"])
    
print("\nCount -- Description")
for n in range(5):
    print(final_array2[n], data_sorted.loc[final_array2[n]]["title"])

print("\nDatasets in both desc. recommender results")
for n in final_array[:5]:
    if n in final_array2[:5]:
        print(n, data_sorted.loc[n]["title"])

print("\nTFIDF -- Keywords")
for n in range(5):
    print(final_array3[n], data_sorted.loc[final_array3[n]]["title"])

print("\nCount -- Keywords")
for n in range(5):
    print(final_array4[n], data_sorted.loc[final_array4[n]]["title"])

print("\nDatasets in both keyword recommender results")
for n in final_array3[:5]:
    if n in final_array4[:5]:
        print(n, data_sorted.loc[n]["title"])

