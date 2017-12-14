from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
from functools import lru_cache

character_n_grams_dict = dict()
word_n_grams_dict = dict()


@lru_cache(maxsize=128)
def character_n_grams(n, document, corpus):
    hashed_corpus = hash_corpus(corpus)
    if (n, hashed_corpus) in character_n_grams_dict:
        vectorizer = character_n_grams_dict[(n, hashed_corpus)]
    else:
        vectorizer = vectorize_character_n_grams(corpus, n)
        character_n_grams_dict[(n, hashed_corpus)] = vectorizer
    matrix = vectorizer.transform([document])
    assert matrix.max() != 0
    return matrix

@lru_cache(maxsize=256)
def vectorize_character_n_grams(corpus, n):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n))
    vectorizer = vectorizer.fit(corpus)
    return vectorizer


@lru_cache(maxsize=256)
def hash_corpus(corpus):
    return hash(str(corpus))


@lru_cache(maxsize=128)
def word_n_grams(n, document, corpus, max_df=1.0, stop_words=None):
    hashed_corpus = hash(str(corpus))
    if (n, max_df, stop_words, hashed_corpus) in character_n_grams_dict:
        vectorizer = word_n_grams_dict[(n, max_df, stop_words, hashed_corpus)]
    else:
        vectorizer = vectorize_word_n_grams(corpus, max_df, n, stop_words)
        word_n_grams_dict[(n, max_df, stop_words, hashed_corpus)] = vectorizer
    return vectorizer.transform([document])


@lru_cache(maxsize=256)
def vectorize_word_n_grams(corpus, max_df, n, stop_words):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(n, n), stop_words=stop_words, max_df=max_df).fit(
        corpus)
    return vectorizer


@lru_cache(maxsize=1024)
def avg_stdev_words_per_sentence(document):
    document_splitted = re.split('. |! |\? ', document)
    # TODO: Maybe remove newlines
    words_per_sentence = []
    for sentence in document_splitted:
        words_per_sentence.append(len(sentence.split()))
    return [[np.average(words_per_sentence)], [np.std(words_per_sentence)]]


@lru_cache(maxsize=1024)
def vocabulary_diversity(document):
    document = "".join(c for c in document if c not in ':;?!.,()')
    document_splitted = document.split()
    counter = Counter(document_splitted)
    return [[len(list(counter)) / len(document_splitted)]]


@lru_cache(maxsize=1024)
def avg_marks(document):
    document_splitted = document.split('.')
    number_of_sentences = len(document_splitted)
    marks = "".join(c for c in document if c in ',;:()!?')
    counter = Counter(marks)
    representation = []
    for element in ',;:()!?':
        try:
            representation.append([counter[element]/number_of_sentences])
        except KeyError:
            representation.append([0])

    return representation


@lru_cache(maxsize=1024)
def concatenation(document):
    features = avg_stdev_words_per_sentence(document)
    features.append(vocabulary_diversity(document)[0])
    features.extend(avg_marks(document))
    return np.reshape(features, (np.shape(features)[0], 1)).tolist()


def load_feature_dict(folder, name):
    if os.path.exists(os.path.join(folder, str(name) + '-char.pickle')):
        with open(os.path.join(folder, str(name) + '-char.pickle'), 'r') as file:
            global character_n_grams_dict
            character_n_grams_dict = pickle.load(file)
    if os.path.exists(os.path.join(folder, str(name) + '-word.pickle')):
        with open(os.path.join(folder, str(name) + '-word.pickle'), 'r') as file:
            global word_n_grams_dict
            word_n_grams_dict = pickle.load(file)


def write_feature_dict(folder, name):
    if not os.path.exists(os.path.join(folder, str(name) + '-char.pickle')):
        with open(os.path.join(folder, str(name) + '-char.pickle'), 'wb') as file:
            pickle.dump(character_n_grams_dict, file)
    if not os.path.exists(os.path.join(folder, str(name) + '-word.pickle')):
        with open(os.path.join(folder, str(name) + '-word.pickle'), 'wb') as file:
            pickle.dump(word_n_grams_dict, file)
