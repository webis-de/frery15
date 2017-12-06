from representation import character_n_grams, word_n_grams, avg_stdev_words_per_sentence,vocabulary_diversity, \
    avg_marks, concatenation


def representation_space1(document, corpus):
    return character_n_grams(n=8, document=document, corpus=corpus)


def representation_space2(document, corpus):
    return character_n_grams(n=3, document=document, corpus=corpus)


def representation_space3(document, corpus):
    return word_n_grams(n=2, document=document, corpus=corpus)


def representation_space4(document, corpus):
    return word_n_grams(n=1, document=document, corpus=corpus, max_df=0.7)


# TODO: Would need stop words for each language
def representation_space5(document, corpus):
    return word_n_grams(n=1, document=document, corpus=corpus, stop_words='english')


def representation_space6(document):
    return avg_stdev_words_per_sentence(document)


def representation_space7(document):
    return vocabulary_diversity(document)


def representation_space8(document):
    return avg_marks(document)


def representation_space678(document):
    return concatenation(document)
