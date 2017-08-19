import os
import urllib
import zipfile
from itertools import chain
import nltk

train_corpora_url = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-14/pan14-data/pan14-authorship-verification-training-corpus-2014-04-22.zip'
train_corpora_dir = 'pan14-authorship-verification-training-corpus-2014-04-22'
data_dir = 'data'


def may_download_training(url, prefix_dir, dir):
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    if not os.path.exists(prefix_dir + '/' + dir):
        zip_file = prefix_dir + '/' + dir + '.zip'
        filename, headers = urllib.urlretrieve(url, zip_file)

        assert os.path.exists(zip_file)
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(prefix_dir)

        assert os.path.exists(prefix_dir + '/' + dir)
        os.remove(zip_file)


def may_unzip_corpus(dir_zips, data_dir, train_corpora_dir):
    for _, _, files in os.walk(dir_zips):
        for file in files:
            if file.endswith(".zip") and not os.path.exists(data_dir + '/' + train_corpora_dir + '/' + file[:-4]):
                with zipfile.ZipFile(data_dir + '/' + train_corpora_dir + '/' + file, "r") as z:
                    z.extractall(data_dir + '/' + train_corpora_dir + '/')

                # Unzipped file has an other name than zip file
                # assert os.path.exists(data_dir+'/'+train_corpora_dir+'/'+file[:-4])
                os.remove(data_dir + '/' + train_corpora_dir + '/' + file)


# From https://stackoverflow.com/questions/22428020/how-to-extract-character-ngram-from-sentences-python?noredirect=1&lq=1
def word_to_char_ngrams(word, n=3):
    """ Convert word into character ngrams. """
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def text_to_char_ngrams(text, n=3):
    """ Convert sentences into character ngrams. """
    return list(chain(*[word_to_char_ngrams(i, n) for i in text.lower().split()]))


def text_to_word_bigrams(text):
    words = nltk.word_tokenize(text)
    bigrams = nltk.bigrams(words)
    return bigrams


def text_to_word_unigrams(text):
    words = nltk.word_tokenize(text)
    return words


# TODO: Phrases: word per sentence mean and standard deviation
# TODO: Vocabulary diversity: total number of different terms divided by the total number of occurrences of words
# TODO: Punctuation: average of punctuation marks per sentence characters: "," ";" ":" "(" ")" "!" "?"

def main():
    may_download_training(train_corpora_url, data_dir, train_corpora_dir)
    may_unzip_corpus(data_dir + '/' + train_corpora_dir, data_dir, train_corpora_dir)


if __name__ == '__main__':
    main()
