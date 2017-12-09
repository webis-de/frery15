from preprocessing import may_download_training, may_unzip_corpus, load_text_corpora
from dissimilarity_counter_method import dissimilarity_counter_method
from representation_spaces import *
from similarity_measures import cosine_similarity, correlation_coefficient, euclidean_distance
from features import count, mean
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

train_corpora_url = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-14/pan14-data/pan14-authorship-verification-training-corpus-2014-04-22.zip'
train_corpora_dir = 'pan14-authorship-verification-training-corpus-2014-04-22'
data_dir = 'data'

def main():
    may_download_training(train_corpora_url, data_dir, train_corpora_dir)
    may_unzip_corpus(data_dir + '/' + train_corpora_dir, data_dir, train_corpora_dir)

    corpora = load_text_corpora(data_dir, train_corpora_dir)

    for corpus in corpora:
        print('Start next corpus')
        corpus_appended = []
        for problem in corpus:
            [known_documents, unknown, _] = problem
            corpus_appended.append(unknown)
            for known_document in known_documents:
                corpus_appended.append(known_document)

        for similarity_measure in [cosine_similarity, correlation_coefficient, euclidean_distance]:
            print('Using similarity measure ' + similarity_measure.__name__)

            X = []
            Y = []
            for [known_documents, unknown, label] in corpus:
                counts = []
                means = []
                features = []

                for representation_space in [lambda document: representation_space1(document, corpus_appended),
                                             lambda document: representation_space2(document, corpus_appended),
                                             lambda document: representation_space3(document, corpus_appended),
                                             lambda document: representation_space4(document, corpus_appended),
                                             lambda document: representation_space5(document, corpus_appended),
                                             lambda document: representation_space6(document),
                                             lambda document: representation_space7(document),
                                             lambda document: representation_space8(document),
                                             lambda document: representation_space678(document)]:
                    known_documents_in_representation_space = []
                    for known_document in known_documents:
                        known_documents_in_representation_space.append(representation_space(known_document))
                    unknown_document_in_representation_space = representation_space(unknown)
                    threshold = None
                    #print(dissimilarity_counter_method(known_documents_in_representation_space,
                    #                                   unknown_document_in_representation_space, threshold=threshold,
                    #                                   similarity_measure=similarity_measure))
                    counts.append(count(known_documents_in_representation_space, unknown_document_in_representation_space, similarity_measure))
                    means.append(mean(known_documents_in_representation_space, unknown_document_in_representation_space, similarity_measure))

                features.extend(counts)
                features.extend(means)
                features.append(np.average(counts))

                # Convert NaNs in features to 0
                features = np.array(features)
                where_are_NaNs = np.isnan(features)
                features[where_are_NaNs] = 0

                X.append(features)
                Y.append(label)
            print('Start cross-validation')
            for classifier in [DecisionTreeClassifier(), SVC(kernel='rbf'), SVC(kernel='linear')]:
                print("roc auc for classifier " + classifier.__class__.__name__)
                scores_roc = cross_val_score(classifier, X, Y, scoring='roc_auc', cv=10, n_jobs=-1)
                print(scores_roc)
                print('Average: ' + str(np.average(scores_roc)))
                print('Standard deviation: ' + str(np.std(scores_roc)))

if __name__ == '__main__':
    main()
