# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec, doc2vec
from sklearn import utils
import re

stopwords = set(stopwords.words('english'))

def process_data(fileName):
    category = fileName.replace('.txt', '').replace('data/', '')
    data = open(fileName, 'r').read().split("\n|||\n")
    data_original = []
    data_tokenized = []
    for text in data:
        text = text.lower()
        text = re.sub("\[.*?\]", "", text)
        text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
        text_tokens_ns = [word for word in text_tokens if not word in stopwords]
        data_original.append((' '.join(text_tokens_ns), category))
        data_tokenized.append((text_tokens_ns, category))
    return data, data_original, data_tokenized


allergies_data, allergies_original, allergies_tokenized = process_data('data/allergies.txt')
family_history_data, family_history_original, family_history_tokenized = process_data('data/family_history.txt')
history_illness_data, history_illness_original, history_illness_tokenized = process_data('data/history_illness.txt')
social_history_data, social_history_original, social_history_tokenized = process_data('data/social_history.txt')

mixed_original = allergies_original + family_history_original + history_illness_original + social_history_original
mixed_tokenized = allergies_tokenized + family_history_tokenized + history_illness_tokenized + social_history_tokenized

random.shuffle(mixed_original)
mixed_original = pd.DataFrame(mixed_original)
mixed_original.columns = ['Data', 'Category']

def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled


X_train, X_test, y_train, y_test = train_test_split(mixed_original.Data, mixed_original.Category, random_state=0,
                                                    test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_data)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

model_dbow.save_word2vec_format('model.bin', binary=True)


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors


train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_vectors_dbow, y_train)
logreg = logreg.fit(train_vectors_dbow, y_train)
y_pred = logreg.predict(test_vectors_dbow)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))