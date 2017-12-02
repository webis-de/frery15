import os
import urllib
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import yaml
import copy
import sys
import numpy
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import pairwise
from sklearn.svm import SVC

train_corpora_url = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-14/pan14-data/pan14-authorship-verification-training-corpus-2014-04-22.zip'
train_corpora_dir = 'pan14-authorship-verification-training-corpus-2014-04-22'
data_dir = 'data'


def cosine_similarity(vector1, vector2):
    return pairwise.cosine_similarity(vector1, vector2)


def correlation_coefficient(vector1, vector2):
    # TODO: Search method for correlation coefficient
    return 0


class FeatureWrapper(object):
    def __init__(self, name, genre, language):
        self.name = name
        self.genre = genre
        self.language = language

        self.unknown_text = None
        self.label = None

        self.count = []
        self.mean = []
        self.count_tot = []

    def set_unknown_text(self, unknown_text):
        self.unknown_text = unknown_text

    def get_unknown_text(self):
        return self.unknown_text

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

    def set_count(self, count, number_representation_state):
        self.count.insert(number_representation_state, count)

    def get_count(self):
        return self.count

    def set_mean(self, mean, number_representation_state):
        self.mean.insert(number_representation_state, mean)

    def get_mean(self):
        return self.mean

    def set_count_tot(self, count_tot, number_representation_state):
        self.count_tot.insert(number_representation_state, count_tot)

    def get_count_tot(self):
        return self.count_tot


class TfidfRepresentationSpace(object):
    def __init__(self, space, analyzer=None, ngram_range=0, stopwords=None, max_df=1.0, similarity=cosine_similarity):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.stopwords = stopwords
        self.max_df = max_df
        self.vectorizer = None
        self.document_matrix = None
        self.label = None
        self.name = None
        self.corpus = None
        self.unknown_text = None
        self.language = None
        self.genre = None
        self.similarity = similarity
        self.mean = None
        self.count = None
        self.space = space

    def set_corpus(self, corpus):
        self.corpus = corpus

    def set_unknown_text(self, text):
        self.unknown_text = text

    def set_language(self, language):
        self.language = language

    def set_genre(self, genre):
        self.genre = genre

    def set_label(self, label):
        self.label = label

    def set_name(self, name):
        self.name = name

    def get_vectorizer(self):
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(analyzer=self.analyzer,
                                              ngram_range=self.ngram_range, stop_words=self.stopwords).fit(self.corpus)
        return self.vectorizer

    # def get_document_matrix(self):
    #    assert self.corpus is not None

    #    if self.document_matrix is None:
    #        self.document_matrix = self.get_vectorizer().fit_transform(self.corpus)
    #    return self.document_matrix

    def abstract_similarity(self, document1, document2):
        return self.similarity(self.get_vectorizer().transform([document1], True),
                               self.get_vectorizer().transform([document2], True))

    def get_count(self):
        if self.count is None:
            count = 0

            # Check for each document if the similarity with the unknown document is lower than the similarity to all other documents
            for document in self.corpus:
                min_incorpus_similarity = sys.maxsize
                for similarity_document in self.corpus:
                    if document == similarity_document:
                        pass
                    min_incorpus_similarity = min(min_incorpus_similarity,
                                                  self.abstract_similarity(document, similarity_document))
                if self.abstract_similarity(document, self.unknown_text) < min_incorpus_similarity:
                    count += 1
            self.count = count

        return self.count

    def get_mean(self):
        if self.mean is None:
            added_similarities = 0
            number_of_documents = 0
            for document in self.corpus:
                added_similarities += self.abstract_similarity(document, self.unknown_text)
                number_of_documents += 1
            # TODO: Investigate why this is a list in a list
            self.mean = (added_similarities / number_of_documents)[0][0]

        return self.mean

        # TODO: Implement TOT_count (added count over all representation spaces)

    def get_space(self):
        return self.space

    def get_unknown_text(self):
        return self.unknown_text

    def get_label(self):
        return self.label


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


# TODO: Phrases: word per sentence mean and standard deviation
# TODO: Vocabulary diversity: total number of different terms divided by the total number of occurrences of words
# TODO: Punctuation: average of punctuation marks per sentence characters: "," ";" ":" "(" ")" "!" "?"

def set_labels(representationSpaces):
    for dirname in os.listdir(data_dir + '/' + train_corpora_dir):
        if dirname == '.DS_Store':
            continue
        with open(data_dir + '/' + train_corpora_dir + '/' + dirname + '/' + 'truth.json') as truth_data:

            truth = yaml.load(truth_data)
            for problem in truth['problems']:
                for representationSpace in representationSpaces:
                    if representationSpace.name == problem['name'] \
                            and representationSpace.genre == problem['genre'] \
                            and representationSpace.language == problem['language']:
                        if problem['answer'] == 'Y':
                            representationSpace.set_label(True)
                        elif problem['answer'] == 'N':
                            representationSpace.set_label(False)
                        else:
                            raise Exception('Answer isn\'t Y or N')


def load_text_corpus(representationSpaces):
    representationSpacesWithCorpus = []
    for representationSpace in representationSpaces:
        for dirname in os.listdir(data_dir + '/' + train_corpora_dir):
            if not os.path.isdir(data_dir + '/' + train_corpora_dir + '/' + dirname):
                continue
            if dirname == train_corpora_dir or dirname == '.DS_Store':
                continue
            with open(data_dir + '/' + train_corpora_dir + '/' + dirname + '/' + 'contents.json') as json_data:
                contents = yaml.load(json_data)
                print(contents)
                for problem in contents['problems']:
                    unknown = open(
                        data_dir + '/' + train_corpora_dir + '/' + dirname + '/' + problem + '/' + 'unknown.txt',
                        'r').read()
                    corpus = []
                    # TODO: should also be replaced with os.listdir
                    for _, _, files in os.walk(
                                                                                    data_dir + '/' + train_corpora_dir + '/' + dirname + '/' + problem + '/'):
                        for file in files:
                            if file.endswith(".txt") and not file == 'unknown.txt':
                                corpus.append(open(
                                    data_dir + '/' + train_corpora_dir + '/' + dirname + '/' + problem + '/' + file,
                                    'r').read())
                    representationSpaceCopy = copy.deepcopy(representationSpace)
                    representationSpaceCopy.set_language(contents['language'])
                    representationSpaceCopy.set_genre(contents['genre'])
                    representationSpaceCopy.set_name(problem)
                    representationSpaceCopy.set_unknown_text(unknown)
                    representationSpaceCopy.set_corpus(corpus)

                    representationSpacesWithCorpus.append(representationSpaceCopy)
                    corpus = []
    return representationSpacesWithCorpus


def build_representation_space():
    representationSpaces = []
    # create TfidfRepresentationSpace objects for each combination from the paper
    representationSpaces.append(TfidfRepresentationSpace(analyzer='char', ngram_range=(3, 3), space=2))
    representationSpaces.append(TfidfRepresentationSpace(analyzer='char', ngram_range=(8, 8), space=1))
    representationSpaces.append(TfidfRepresentationSpace(analyzer='char_wb', ngram_range=(3, 3), space=9))
    representationSpaces.append(TfidfRepresentationSpace(analyzer='char_wb', ngram_range=(8, 8), space=10))

    representationSpaces.append(TfidfRepresentationSpace(analyzer='word', ngram_range=(2, 2), space=3))
    # TODO: Stopwords should be language dependent
    representationSpaces.append(
        TfidfRepresentationSpace(analyzer='word', ngram_range=(1, 1), stopwords='english', space=5))
    representationSpaces.append(TfidfRepresentationSpace(analyzer='word', ngram_range=(1, 1), max_df=0.7, space=4))
    # TODO: Set correct similarity method
    return representationSpaces


def build_features(representationSpaces):
    # Collect all valid tupels (genre,name)
    genre_name_language = []
    for representationSpace in representationSpaces:
        genre_name_language.append((representationSpace.genre, representationSpace.name, representationSpace.language))
    problems = set(genre_name_language)
    featureWrappers = []
    for problem in problems:
        (genre, name, language) = problem
        featureWrappers.append(FeatureWrapper(name, genre, language))

    for representationSpace in representationSpaces:
        for featureWrapper in featureWrappers:
            if representationSpace.genre == featureWrapper.genre and representationSpace.name == featureWrapper.name:
                assert representationSpace.get_space() is not None

                count = representationSpace.get_count()
                if count is None:
                    count = 0
                mean = representationSpace.get_mean()
                if mean is None:
                    mean = 0
                # if mean is not 0:
                #    mean = mean[0]

                featureWrapper.set_count(count, representationSpace.get_space())
                featureWrapper.set_mean(mean, representationSpace.get_space())
                # TODO: featureWrapper.set_tot_count(representationSpace.get_mean(), representationSpace.get_space())
                # TODO: Sets label and unknown text multiple times
                featureWrapper.set_label(representationSpace.get_label())
                featureWrapper.set_unknown_text(representationSpace.get_unknown_text())
    return featureWrappers


def main():
    print('May download training data...')
    may_download_training(train_corpora_url, data_dir, train_corpora_dir)
    print('May unzip corpus...')
    may_unzip_corpus(data_dir + '/' + train_corpora_dir, data_dir, train_corpora_dir)

    print('Build representation spaces...')
    representationSpaces = build_representation_space()
    print('Load text corpora...')
    representationSpaces = load_text_corpus(representationSpaces)
    set_labels(representationSpaces)
    print('Build features...')
    featureWrappers = build_features(representationSpaces)

    X = []
    Y = []

    for featureWrapper in featureWrappers:
        mergedlist = []
        mergedlist.extend(featureWrapper.get_count())
        mergedlist.extend(featureWrapper.get_mean())
        X.append(mergedlist)
        Y.append(featureWrapper.get_label())
    print('Use all data for one model:')
    train_evaluate_model(X, Y)

    for (language, genre) in [('english', 'novels'), ('english', 'essays'), ('dutch', 'reviews'), ('dutch', 'essay'),
                              ('spanish', 'articles'), ('greek', 'articles')]:
        print('Use corpus ', language, ' ', genre)
        for featureWrapper in featureWrappers:
            if featureWrapper.language is language and featureWrapper.genre is genre:
                mergedlist = []
                mergedlist.extend(featureWrapper.get_count())
                mergedlist.extend(featureWrapper.get_mean())
                X.append(mergedlist)
                Y.append(featureWrapper.get_label())
        train_evaluate_model(X, Y)


def train_evaluate_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    decision_tree_result = 0
    svm_result = 0
    iterations = 10
    divisor_iterations = iterations

    while iterations > 0:
        print('Calculating iteration ', iterations, '...')
        # print('Fit decision tree classifier...')
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        # print('Predict test samples...')
        Y_predicted = clf.predict(X_test)
        decision_tree_result += metrics.roc_auc_score(Y_test, Y_predicted)
        # print('Fit svm classifier...')
        svm = SVC()
        svm = svm.fit(X_train, Y_train)
        # print('Predict test samples...')
        Y_predicted = svm.predict(X_test)
        svm_result += metrics.roc_auc_score(Y_test, Y_predicted)
        iterations -= 1

    print('decision tree classifier result: ', decision_tree_result / divisor_iterations)
    print('svm classifier result: ', svm_result / divisor_iterations)


if __name__ == '__main__':
    main()
