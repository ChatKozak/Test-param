#!/usr/bin/env python
# coding: utf-8

import subprocess

# Command to install the datasets package using pip
command = "pip install datasets"

# Use subprocess to run the command
subprocess.call(command, shell=True)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import re
import nltk
import gensim
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from gensim.models import Word2Vec
from datasets import get_dataset_split_names
get_dataset_split_names("squad")

from datasets import load_dataset
dataset_train = load_dataset("squad", split="train")
dataset_test = load_dataset("squad", split="validation")
print(dataset_train)
print(dataset_train[0])

df_train = pd.DataFrame(dataset_train)

print(df_train)

df2 = pd.json_normalize(df_train['answers'])
print(df2)


df_train= pd.concat([df_train, df2], axis=1)
df_train.drop(["answers"],axis=1,inplace=True)
print(df_train)


df_train['text'] = [''.join(map(str, l)) for l in df_train['text']]
df_train['answer_start'] = [''.join(map(str, l)) for l in df_train['answer_start']]
print(df_train)

nltk.download('stopwords')
STOP_WORDS = nltk.corpus.stopwords.words("english")


def clean_sentence(val):
    # remove chars that are not letters or numbers, downcase, then remove stop words
    regex = re.compile('([^\s\w]|_)+')
    try:
      sentence = regex.sub('', val).lower()
      sentence = sentence.split(" ")
    except :
      sentence = val
    try :
      for word in list(sentence):
        try:
          if word in STOP_WORDS:
            sentence.remove(word)
        except:
          pass
      sentence = " ".join(sentence)
    except :
      pass

    return sentence


def clean_dataframe(data):
    # drop nans, then apply 'clean_sentence' function to 'context', 'question', 'text', 'answer_start'
    data = data.dropna(how="any")
    for col in ['context', 'question', 'text']:
      data[col] = data[col].apply(clean_sentence)
    return data


data_train= clean_dataframe(df_train)
print(data_train.head(5))


def build_corpus(data):
  #"Creates a list of lists containing words from each sentence"
    corpus = []
    for col in  ['context', 'question', 'text']:
      for sentence in data[col].iteritems():
        word_list = sentence[1].split(" ")
        corpus.append(word_list)
    return corpus

corpus = build_corpus(data_train)
print(len(corpus))


df_test = pd.DataFrame(dataset_test)
df2 = pd.json_normalize(df_test['answers'])
df_test = pd.concat([df_test, df2], axis=1)
df_test.drop(['answers'],axis=1 , inplace=True)


df_test['text'] = [l[0] for l in df_test['text']]
df_test['answer_start'] = [l[0] for l in df_test['answer_start']]
df_test['text'] = df_test['text'].astype(str)

print(df_test)


print(df_test.info())

data_test = clean_dataframe(df_test)
print(data_test)

print("test 1 :")
tfidf_vectorizer_question = TfidfVectorizer()
X_train_question = tfidf_vectorizer_question.fit_transform(data_train['question'])
X_test_question = tfidf_vectorizer_question.transform(data_test['question'])

print("test 2 :")

tfidf_vectorizer_context = TfidfVectorizer()
X_train_context = tfidf_vectorizer_context.fit_transform(data_train['context'])
X_test_context = tfidf_vectorizer_context.transform(data_test['context'])

print("test 3 :")

tfidf_vectorizer_title = TfidfVectorizer()
X_train_title = tfidf_vectorizer_title.fit_transform(data_train['context'])
X_test_title = tfidf_vectorizer_title.transform(data_test['context'])

print("train :")
X_train_combined = hstack([X_train_question, X_train_context,X_train_title])
X_test_combined = hstack([X_test_question, X_test_context,X_test_title])


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_combined, data_train['text'])


predictions = clf.predict(X_test_combined)

print(predictions)

# Calculate accuracy
accuracy = accuracy_score(data_test['text'], predictions)
print('Accuracy:',accuracy)


plt.figure(figsize=(6, 4))
plt.bar(['Random Forest'], [accuracy])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

