#Reference : https://github.com/parulnith/Building-a-Simple-Chatbot-in-Python-using-NLTK/blob/master/chatbot.py

import nltk
import warnings
import pandas as pd
import numpy as np
import scipy.sparse
import re

warnings.filterwarnings("ignore")

import pickle
#nltk.download() # for downloading packages

import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import io
import codecs
import os
import pymysql
import pandas as pd
import psycopg2
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


pd.set_option('display.max_columns', 50000)

#SQL connection for Questions

conn = psycopg2.connect(user = "postgres",
                               password = "root",
                               host = "localhost",
                               port = "5432",
                               database = "postgres")
df_questions = pd.read_sql_query("SELECT * FROM capstone.questions",conn)
df_answers = pd.read_sql_query("SELECT * FROM capstone.answers",conn)

#print(df_questions.head())

#print(df_answers.head())

df_questions=df_questions.sort_values(by='question_id')
sentence_tokens = []
sentence =[]
Answer_id=[]

sw = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

for i in range (0,len(df_questions)):
    raw=df_questions.iloc[i]['question']
    sent_tokens = nltk.sent_tokenize(raw)
    sentence_tokens.extend(sent_tokens)
    sentence.extend(sent_tokens)
    Answer_id.append(df_questions.iloc[i]['ans_id'])
    #print("i : "+str(i)+"Question id: "+str(df_questions.iloc[i]['question_id'])+" Answer_id : "+str(df_questions.iloc[i]['ans_id']))

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=sw)
tfidf = TfidfVec.fit_transform(sentence_tokens)

df_sentence=pd.DataFrame(data=tfidf.toarray(),columns=TfidfVec.get_feature_names())
tfidf_matrix=scipy.sparse.csr_matrix(df_sentence)


pickle.dump([tfidf,TfidfVec.get_feature_names(),TfidfVec,sentence,Answer_id,tfidf_matrix], open( "tfidf.dat", "wb" ) )

# SECONDARY TFIDF

filenames=[]
for i in range(1,1244):
    filenames.append(str(i)+".txt")

sentence_tokens_2 = []
sentence_2 =[]
url_2=[]


for i in range (0,len(filenames)):
   # n="C:\github\hp\\flask-chatbot\html_files\\"+str(filenames[i])
    n="https://github.com/hpanchbhai/capstonehp/tree/master/html_files/"+str(filenames[i])
    f = io.open(n, "r", encoding='utf-8')
    raw2=f.read()
    result=re.search('Source Link: (.*)\n\nSkip to main content',raw2)
    raw3=raw2.replace("*","").replace("#","")
    raw4=re.sub(r'\([^()]*\)','',raw3)
    raw5=re.sub(r'\[.*]','',raw4)
    raw6 = " ".join(re.split("\s+", raw5, flags=re.UNICODE))
    sent_tokens_2 = nltk.sent_tokenize(raw6)
    sentence_tokens_2.extend(sent_tokens_2)
    sentence_2.extend(sent_tokens_2)
    url_2.extend([result.group(1) for j in range(len(sent_tokens_2))])


TfidfVec_2 = TfidfVectorizer(tokenizer=LemNormalize, stop_words=sw)
tfidf_2 = TfidfVec_2.fit_transform(sentence_tokens_2)

df_sentence_2=pd.DataFrame(data=tfidf_2.toarray(),columns=TfidfVec_2.get_feature_names())
tfidf_matrix_2=scipy.sparse.csr_matrix(df_sentence_2)


pickle.dump([tfidf_2,TfidfVec_2.get_feature_names(),TfidfVec_2,sentence_2,url_2,tfidf_matrix_2], open( "tfidf_2.dat", "wb" ) )
