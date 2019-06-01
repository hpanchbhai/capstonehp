
from flask import Flask, render_template, request, jsonify
import aiml as aiml
import os
#HP1
kernel = aiml.Kernel()


#Reference : https://github.com/parulnith/Building-a-Simple-Chatbot-in-Python-using-NLTK/blob/master/chatbot.py

import nltk
import warnings
import pandas as pd
import numpy as np
import scipy.sparse
import re
from capstone1 import LemNormalize #Answer_id, df_answers

warnings.filterwarnings("ignore")

import pickle
#nltk.download() # for downloading packages

import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import os
import psycopg2
import pandas as pd

l1 = pickle.load( open( "tfidf.dat", "rb" ) )
l2 = pickle.load( open( "tfidf_2.dat", "rb" ) )


Answer_id = l1[4]


conn = psycopg2.connect(user = "postgres",
                               password = "root",
                               host = "localhost",
                               port = "5432",
                               database = "postgres")
df_answers = pd.read_sql_query("SELECT * FROM capstone.answers",conn)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
	return [lemmer.lemmatize(token) for token in tokens]
	
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
	return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
# Checking for greetings

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']





def weight(user_response):
    last_query1=(user_response[-1].split(" "))
    last_query=[]
    for word in last_query1:
        if word not in stopwords:
            last_query.append(word)
    query_words_df=pd.DataFrame(columns=last_query,data=np.full(shape=(1,len(last_query)),fill_value=1,dtype=np.int))
    i=len(user_response)-1
    j=2
    while i:
        temp=user_response[i-1].split(" ")
        for word in temp:
            if word in query_words_df.columns:
                if word not in stopwords:
                    query_words_df[word]=query_words_df[word]+(1/j)
            else:
                if word not in stopwords:
                    query_words_df[word]=1/j
        i=i-1
        j=j+1
    return query_words_df



def user_response_matrix(user_response):
    l1 = pickle.load( open( "tfidf.dat", "rb" ) )
    names = [x.encode('UTF8') for x in l1[1]]
    TfidfVec_response = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    user_response_new=nltk.sent_tokenize(user_response)
    tfidf_response = TfidfVec_response.fit_transform(user_response_new)
    query_words_df=pd.DataFrame(columns=TfidfVec_response.get_feature_names(), data=tfidf_response.toarray())
    print(query_words_df)
    
    df_all=pd.DataFrame(np.zeros((1,len(names))),columns=names)
    for col in query_words_df.columns:
        for col2 in df_all.columns:
            if col==col2:
                df_all[col2]=query_words_df[col]
    tfidf_response=scipy.sparse.csr_matrix(df_all)
    print("TFIDF response")
    print(tfidf_response)
    return tfidf_response          

def user_response_matrix_secondary(user_response):
    l2 = pickle.load( open( "tfidf_2.dat", "rb" ) )
    TfidfVec_response_2 = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    user_response_new_2=nltk.sent_tokenize(user_response)
    tfidf_response_2 = TfidfVec_response_2.fit_transform(user_response_new_2)
    query_words_df_2=pd.DataFrame(columns=TfidfVec_response_2.get_feature_names(), data=tfidf_response_2.toarray())
    
    df_all_2=pd.DataFrame(np.zeros((1,len(l2[1]))),columns=l2[1])
    for col in query_words_df_2.columns:
        for col2 in df_all_2.columns:
            if col==col2:
                df_all_2[col2]=query_words_df_2[col]
    tfidf_response_2=scipy.sparse.csr_matrix(df_all_2)
    return tfidf_response_2          

def response(user_response):
    #query_words_df=weight(user_response)
    robo_response=[  ]
    tfidf_response_matrix=user_response_matrix(user_response)
    vals = cosine_similarity(tfidf_response_matrix, l1[-1])
    idx=vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    print("Cosine similarity : ")
    print(flat) 
    req_tfidf = flat[-3:]
    print("req_tfidf")
    print (req_tfidf)
    print('Type of TFIDF : '+str(type(req_tfidf)))

    #for i in req_tfidf:
    if(req_tfidf[-1]<0.5):
        tfidf_response_matrix_2=user_response_matrix_secondary(user_response)
        vals_2 = cosine_similarity(tfidf_response_matrix_2, l2[-1])
        idx_2=vals_2.argsort()[0][-3:]
        flat_2 = vals_2.flatten()
        flat_2.sort()
        print("Cosine similarity : ")
        print(flat_2)      
        req_tfidf_2 = flat_2[-3:]
        j=2
        #for ii in req_tfidf_2:
        if(req_tfidf_2[-1]<0.4):
            robo_response.append("I am sorry! I don't understand you. Please enter more relevant keywords")
            robo_response.append(None)
            robo_response.append(3)
                    #return robo_response
        else:
            while(j>-1):
                robo_response.append(l2[3][idx_2[j]])
                robo_response.append(l2[4][idx_2[j]])
                robo_response.append(2)
                j=j-1   
            print(robo_response)
        return (robo_response)
    else:
        ans_id=Answer_id[idx]
        print('ANS ID ==> '+str(ans_id))
        robo_response.append(df_answers[df_answers['answer_id']==ans_id]['answer'].iloc[0])
        robo_response.append(df_answers[df_answers['answer_id']==ans_id]['url'].iloc[0])
        robo_response.append(1)           
        return robo_response
