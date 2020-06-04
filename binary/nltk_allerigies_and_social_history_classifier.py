import argparse
import nltk
import random
import re
import pickle

from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-in", "--inputFile", required=True, help="Specify file you want to parse.")
args = vars(ap.parse_args())

# Set corpus size
array_size = 2000

# List of stopwords
all_stopwords = set(stopwords.words('english'))

# Load and prepare test input
test_data = open(args['inputFile'], "r").read().split("\n|||\n")
test_document = []
for text in test_data:
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    test_document.append((text_tokens_ns, 'neutral'))

# for item in test_document:
#     print(item)

# Load and prepare dataset
pos_data = open("allergies.txt", "r").read().split("\n|||\n")
pos_document = []
for text in pos_data:
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    pos_document.append((text_tokens_ns, 'allergies'))

neg_data = open("social_history.txt", "r").read().split("\n|||\n")
neg_document = []
for text in neg_data:
    text = re.sub("\[.*?\]", "", text)
    text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
    text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
    neg_document.append((text_tokens_ns, 'social_history'))

documents = pos_document + neg_document
random.shuffle(documents)

# print(pos_document)

words = []
# Make one big list of words combined
for (d, c) in pos_document:
    for word in d:
        words.append(word)
for (d, c) in neg_document:
    for word in d:
        words.append(word)

# Define feature extractor
all_words = nltk.FreqDist(w.lower() for w in words)
all_words.plot(30, cumulative=False)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

input_file_results = [classifier.classify(document_features(d)) for (d, c) in test_document]
output_file = list(zip(input_file_results, test_data))

with open('output_file.txt', 'w') as filehandle:
    for (d, c) in output_file:
        filehandle.write("%s, %s\n" % (d, c))
