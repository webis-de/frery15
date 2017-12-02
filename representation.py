#def representation(document):
#    raise NotImplementedError('Please use an implemented representation')

def character_n_grams(n, document, corpus):
    return -1

def word_n_grams(n, document, corpus, with_30_percent_most_frequent=True, with_stop_words=True):
    return -1

def avg_stdev_words_per_sentence(document):
    return -1

def vocubalary_diversity(document):
    return -1

def avg_marks(document):
    return -1

def concatenation(document):
    features = avg_stdev_words_per_sentence(document)
    features.extend(vocubalary_diversity(document))
    features.extend(avg_marks(document))
    return features