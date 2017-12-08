import numpy as np


# set_of_known_documents_space and unknown_document_space in representation space
def dissimilarity_counter_method(set_of_known_documents_space, unknown_document_space, similarity_measure, threshold=None):
    if threshold is None:
        threshold = len(set_of_known_documents_space)/2
    count = 0
    for known_document in set_of_known_documents_space:
        smin = 1
        for other_known_document in set_of_known_documents_space:
            if id(known_document) == id(other_known_document):
                pass
            similarity = similarity_measure(known_document, other_known_document)
            if smin > similarity:
                smin = similarity
        if similarity_measure(unknown_document_space, known_document) > smin:
            count += 1
    if count > threshold:
        return True
    else:
        return False


# Returns a random decision if there is no majority
def dissimilarity_counter_method_voting(sets_of_known_documents_space, unknown_documents_space, threshold, similarity_measure):
    known = 0
    unknown = 0
    for (set_of_known_documents, unknown_document) in zip(sets_of_known_documents_space, unknown_documents_space):
        result = dissimilarity_counter_method(set_of_known_documents, unknown_document,threshold, similarity_measure)
        if result is True:
            known += 1
        else:
            unknown += 1
    if known > unknown:
        return True
    if unknown > known:
        return False
    else:
        print('Returning random decision because there is no majority')
        if np.random.random() < 0.5:
            return True
        else:
            return False