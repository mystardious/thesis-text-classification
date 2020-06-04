# Import required packages
import logging

import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gensim
import random
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

# List of stopwords
all_stopwords = set(stopwords.words('english'))

pos_data = open("allergies.txt", "r").read().split("\n|||\n")
pos_document = []
pos_p = []
for text in pos_data:
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    pos_p.append((' '.join(text_tokens_ns), 'allergies'))
    pos_document.append((text_tokens_ns, 'allergies'))

neg_data = open("social_history.txt", "r").read().split("\n|||\n")
neg_document = []
neg_p = []
for text in neg_data:
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    neg_p.append((' '.join(text_tokens_ns), 'social_history'))
    neg_document.append((text_tokens_ns, 'social_history'))

neu_data = open("family_history.txt", "r").read().split("\n|||\n")
neu_document = []
neu_p = []
for text in neu_data:
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    neu_p.append((' '.join(text_tokens_ns), 'family_history'))
    neu_document.append((text_tokens_ns, 'family_history'))

ext_data = open("family_history.txt", "r").read().split("\n|||\n")
ext_document = []
ext_p = []
for text in ext_data:
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    ext_p.append((' '.join(text_tokens_ns), 'history_illness'))
    ext_document.append((text_tokens_ns, 'history_illness'))

df = pos_p + neg_p + ext_p + neu_p
random.shuffle(df)
df = pd.DataFrame(df, names=['Data', 'Category'])

X = df[0]
y = df[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

test_tokenized = X_test.apply(lambda r: w2v_tokenize_text(r['post'appapp]), axis=1).values
train_tokenized = X_train.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(n_jobs=1, C=1e5)),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
