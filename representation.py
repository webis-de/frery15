from collections import Counter
import re
import numpy as np


def character_n_grams(n, document, corpus):
    return -1


def word_n_grams(n, document, corpus, with_30_percent_most_frequent=True, with_stop_words=True):
    return -1


def avg_stdev_words_per_sentence(document):
    document_splitted = re.split('. |! |\? ', document)
    # TODO: Maybe remove newlines
    words_per_sentence = []
    for sentence in document_splitted:
        words_per_sentence.append(len(sentence.split()))
    return [np.average(words_per_sentence), np.std(words_per_sentence)]


def vocubalary_diversity(document):
    document = "".join(c for c in document if c not in ':;?!.,()')
    document_splitted = document.split()
    counter = Counter(document_splitted)
    return len(list(counter)) / len(document_splitted)


def avg_marks(document):
    document_splitted = document.split('.')
    number_of_sentences = len(document_splitted)
    marks = "".join(c for c in document if c in ',;:()!?')
    counter = Counter(marks)
    representation = []
    for element in list(counter):
        representation.append(counter[element]/number_of_sentences)
    return representation


def concatenation(document):
    features = avg_stdev_words_per_sentence(document)
    features.extend(vocubalary_diversity(document))
    features.extend(avg_marks(document))
    return features
