from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from preprocessing import may_download_data, may_unzip_corpus, load_text_corpora
from dissimilarity_counter_method import dissimilarity_counter_method
from representation_spaces import *
from similarity_measures import cosine_similarity, correlation_coefficient, euclidean_distance
from representation import load_feature_dict, write_feature_dict
from features import count, mean
import numpy as np
from copy import deepcopy
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import jsonhandler
import pickle
import os
import sys


train_corpora_url = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-14/pan14-data/pan14-authorship-verification-training-corpus-2014-04-22.zip'
train_corpora_dir = 'pan14-authorship-verification-training-corpus-2014-04-22'
test_corpora_url = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-14/pan14-data/pan14-authorship-verification-test-corpus2-2014-04-22.zip'
test_corpora_dir = 'pan14-authorship-verification-test-corpus2-2014-04-22'
data_dir = 'data'
features_dict_folder = 'features_dict'

attribution_dataset_data_dir = '../authorship-attribution'
attribution_dataset_dirs = ['pan11-authorship-attribution-training-dataset-small-2015-10-20',
                            'pan11-authorship-attribution-training-dataset-large-2015-10-20',
                            'pan12-authorship-attribution-training-dataset-problem-a-2015-10-20',
                            'pan12-authorship-attribution-training-dataset-problem-b-2015-10-20',
                            'pan12-authorship-attribution-training-dataset-problem-c-2015-10-20',
                            'pan12-authorship-attribution-training-dataset-problem-d-2015-10-20',
                            'pan12-authorship-attribution-training-dataset-problem-i-2015-10-20',
                            'pan12-authorship-attribution-training-dataset-problem-j-2015-10-20',
                            'stamatatos06-authorship-attribution-training-dataset-c10-2015-10-20']


def training_k_fold():
    may_download_data(train_corpora_url, data_dir, train_corpora_dir)
    may_unzip_corpus(data_dir + '/' + train_corpora_dir, data_dir, train_corpora_dir)

    corpora = load_text_corpora(data_dir, train_corpora_dir)
    corpora_hash = hash_corpora(corpora)

    load_feature_dict(features_dict_folder, corpora_hash)

    for corpus in corpora:
        print('Start next corpus')
        corpus_each_problem_as_one_text = corpus_as_one_text(corpus)

        for similarity_measure in [cosine_similarity, correlation_coefficient, euclidean_distance]:
            X, Y = calculate_features_in_representation_space(corpus, similarity_measure, corpus_each_problem_as_one_text)
            print('Start cross-validation')
            for classifier in [DecisionTreeClassifier(), SVC(kernel='rbf'), SVC(kernel='linear')]:
                print("roc auc for classifier " + classifier.__class__.__name__)
                scores_roc = cross_val_score(classifier, X, Y, scoring='roc_auc', cv=10, n_jobs=-1)
                print(scores_roc)
                print('Average: ' + str(np.average(scores_roc)))
                print('Standard deviation: ' + str(np.std(scores_roc)))

    write_feature_dict(features_dict_folder, corpora_hash)


def training_test():
    # training data
    may_download_data(train_corpora_url, data_dir, train_corpora_dir)
    may_unzip_corpus(data_dir + '/' + train_corpora_dir, data_dir, train_corpora_dir)

    # test data
    # TODO: Doesn't lie in the correct folder
    may_download_data(test_corpora_url, data_dir, test_corpora_dir)
    may_unzip_corpus(data_dir + '/' + test_corpora_dir, data_dir, test_corpora_dir)

    train_corpora = load_text_corpora(data_dir, train_corpora_dir)
    test_corpora = load_text_corpora(data_dir, test_corpora_dir)

    complete_corpora = deepcopy(train_corpora)
    complete_corpora.extend(test_corpora)
    corpora_hash = hash_corpora(complete_corpora)

    load_feature_dict(features_dict_folder, corpora_hash)

    for train_corpus, test_corpus in zip(train_corpora, test_corpora):
        print('Start next corpus')

        train_corpus_each_problem_as_one_text = corpus_as_one_text(train_corpus)
        test_corpus_each_problem_as_one_text = corpus_as_one_text(test_corpus)

        for similarity_measure in [cosine_similarity, correlation_coefficient, euclidean_distance]:
            X_train, Y_train = calculate_features_in_representation_space(train_corpus, similarity_measure,
                                                                          train_corpus_each_problem_as_one_text)
            X_test, Y_test = calculate_features_in_representation_space(test_corpus, similarity_measure,
                                                                          test_corpus_each_problem_as_one_text)
            print('Start training and test')
            for classifier in [DecisionTreeClassifier(), SVC(kernel='rbf'), SVC(kernel='linear')]:
                for metric in [roc_auc_score, accuracy_score]:
                    clf = classifier.fit(X_train, Y_train)
                    predicted_labels = classifier.predict(X_test)
                    print(metric.__name__)
                    print(classifier.__class__.__name__)
                    print(metric(Y_test, predicted_labels))

    write_feature_dict(features_dict_folder, corpora_hash)


def do_attribution():
    #for dataset in attribution_dataset_dirs[2:]:#attribution_dataset_dirs:
    corpus_name = sys.argv[1]
    print('Load attribution data')
    corpus = load_attribution_data(corpus_name)
    dataset = attribution_dataset_data_dir + '/' + corpus_name

    #load_feature_dict(features_dict_folder, corpora_hash)

    for similarity_measure in [cosine_similarity, correlation_coefficient, euclidean_distance]:
        if not os.path.exists(os.path.join(attribution_dataset_data_dir, dataset, 'X.pickle')):
            X, Y = calculate_attribution_features_in_representation_space(corpus, similarity_measure, dataset)
            for (content, filename) in [(X, 'X.pickle'), (Y, 'Y.pickle')]:
                dfile = open(os.path.join(attribution_dataset_data_dir, dataset, filename), "wb")
                pickle.dump(content, dfile, protocol=pickle.HIGHEST_PROTOCOL)
                dfile.close()
        else:
            X = pickle.load(os.path.join(attribution_dataset_data_dir, dataset, 'X.pickle'))
            Y = pickle.load(os.path.join(attribution_dataset_data_dir, dataset, 'Y.pickle'))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        print('Start training and test')
        for classifier in [DecisionTreeClassifier(), SVC(kernel='rbf'), SVC(kernel='linear')]:
            for metric in [roc_auc_score, accuracy_score]:
                clf = classifier.fit(X_train, Y_train)
                predicted_labels = classifier.predict(X_test)
                print(metric.__name__)
                print(classifier.__class__.__name__)
                try:
                    print(metric(Y_test, predicted_labels))
                except ValueError as e:
                    print(e)

    #write_feature_dict(features_dict_folder, corpora_hash)


def calculate_attribution_features_in_representation_space(corpus, similarity_measure, dataset):
    jsonhandler.reset_state()
    jsonhandler.loadJson(dataset)

    print('Using similarity measure ' + similarity_measure.__name__)
    X = []
    Y = []

    args = []
    for element in corpus:
        args.append((similarity_measure, element))

    pool = Pool(processes=cpu_count()-2)
    results = pool.map_async(calculate_features, args).get()
    for (features, label) in results:
        X.append(features)
        Y.append(label)
    print("Calculated features")
    pool.close()
    pool.join()
    jsonhandler.reset_state()
    return X, Y


def calculate_features(args):
    (similarity_measure, (known_documents, unknown, label)) = args
    counts = []
    means = []
    features = []
    for representation_space in [representation_space1, representation_space2, representation_space3,
                                 representation_space4, representation_space5, representation_space6,
                                 representation_space7, representation_space8, representation_space678]:
        known_documents_in_representation_space = []
        for known_document in known_documents:
            (author, textfile) = known_document
            known_documents_in_representation_space.append(
                jsonhandler.loadTransformedTrainingText(author, textfile, representation_space.__name__))
        (author, textfile) = unknown
        unknown_document_in_representation_space = jsonhandler.loadTransformedTrainingText(author, textfile,
                                                                                           representation_space.__name__)
        threshold = None
        # print(dissimilarity_counter_method(known_documents_in_representation_space,
        #                                   unknown_document_in_representation_space, threshold=threshold,
        #                                   similarity_measure=similarity_measure))
        counts.append(count(known_documents_in_representation_space, unknown_document_in_representation_space,
                            similarity_measure))
        means.append(mean(known_documents_in_representation_space, unknown_document_in_representation_space,
                          similarity_measure))
    features.extend(counts)
    features.extend(means)
    features.append(np.average(counts))
    # Convert NaNs in features to 0
    features = np.array(features)
    where_are_NaNs = np.isnan(features)
    features[where_are_NaNs] = 0

    #jsonhandler.reset_state()

    return features, label


def calculate_features_in_representation_space(corpus, similarity_measure, corpus_each_problem_as_one_text):
    print('Using similarity measure ' + similarity_measure.__name__)
    X = []
    Y = []
    for [known_documents, unknown, label] in corpus:
        counts = []
        means = []
        features = []

        for representation_space in [lambda document: representation_space1(document, corpus_each_problem_as_one_text),
                                     lambda document: representation_space2(document, corpus_each_problem_as_one_text),
                                     lambda document: representation_space3(document, corpus_each_problem_as_one_text),
                                     lambda document: representation_space4(document, corpus_each_problem_as_one_text),
                                     lambda document: representation_space5(document, corpus_each_problem_as_one_text),
                                     lambda document: representation_space6(document),
                                     lambda document: representation_space7(document),
                                     lambda document: representation_space8(document),
                                     lambda document: representation_space678(document)]:
            known_documents_in_representation_space = []
            for known_document in known_documents:
                known_documents_in_representation_space.append(representation_space(known_document))
            unknown_document_in_representation_space = representation_space(unknown)
            threshold = None
            # print(dissimilarity_counter_method(known_documents_in_representation_space,
            #                                   unknown_document_in_representation_space, threshold=threshold,
            #                                   similarity_measure=similarity_measure))
            counts.append(count(known_documents_in_representation_space, unknown_document_in_representation_space,
                                similarity_measure))
            means.append(mean(known_documents_in_representation_space, unknown_document_in_representation_space,
                              similarity_measure))

        features.extend(counts)
        features.extend(means)
        features.append(np.average(counts))

        # Convert NaNs in features to 0
        features = np.array(features)
        where_are_NaNs = np.isnan(features)
        features[where_are_NaNs] = 0

        X.append(features)
        Y.append(label)
    return X, Y


def hash_corpora(corpora):
    corpora_as_one_text = []
    for corpus in corpora:
        corpora_as_one_text.append(corpus_as_one_text(corpus))
    return hash(str(corpora_as_one_text))


def corpus_as_one_text(corpus):
    corpus_each_problem_as_one_text = []
    for problem in corpus:
        [known_documents, unknown, _] = problem
        corpus_each_problem_as_one_text.append(unknown)
        for known_document in known_documents:
            corpus_each_problem_as_one_text.append(known_document)
    return corpus_each_problem_as_one_text


def load_attribution_data(corpus_name):
    dataset = attribution_dataset_data_dir + '/' + corpus_name

    if not os.path.exists(os.path.join('corpora_texts', dataset)):
        if not os.path.exists('corpora_texts'):
            os.makedirs('corpora_texts')
        candidates = jsonhandler.candidates
        unknowns = jsonhandler.unknowns
        jsonhandler.loadJson(dataset)
        jsonhandler.loadTraining()
        corpus = []
        for author in candidates:
            for other_author in candidates:
                if author == other_author:
                    continue
                for unknown_text in jsonhandler.trainings[other_author]:
                    data_sample = []
                    known_documents = []
                    for known_document in jsonhandler.trainings[author]:
                        if known_document != unknown_text:
                            known_documents.append((author, known_document))
                    data_sample.append(known_documents)
                    data_sample.append((other_author, unknown_text))
                    data_sample.append(False)
                    corpus.append((data_sample))
            for unknown in jsonhandler.trainings[author]:
                data_sample = []
                known_documents = []
                for known_document in jsonhandler.trainings[author]:
                    if unknown != known_document:
                        known_documents.append((author, known_document))
                data_sample.append(known_documents)
                data_sample.append((author, unknown))
                data_sample.append(True)
                corpus.append((data_sample))
        # Another run of the program could have written the corpus
        if not os.path.exists(os.path.join('corpora_texts', dataset)):
            with open(os.path.join('corpora_texts', corpus_name), 'wb') as pickle_file:
                pickle.dump(corpus, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join('corpora_texts', corpus_name), 'rb') as pickle_file:
            corpus = pickle.load(pickle_file)
    return corpus


if __name__ == '__main__':
    do_attribution()
