import sys
import numpy as np

def count(known_documents, unknown_document, similarity):
    """

    :param known_documents: All known documents given in the representation space
    :param unknown_document: Unknown document given in the representation space
    :param similarity
    :return:
    """
    count = 0

    # Check for each document if the similarity with the unknown document is lower than the similarity to all other documents
    for known_document in known_documents:
        min_incorpus_similarity = sys.maxsize
        for other_known_document in known_documents:
            if id(known_document) == id(other_known_document):
                pass
            min_incorpus_similarity = min(min_incorpus_similarity,
                                                  similarity(known_document, other_known_document))
        if min_incorpus_similarity < similarity(known_document, unknown_document):
            count += 1
    if len(known_documents) != 0:
        count *= 1/len(known_documents)

    return count


def mean(known_documents, unknown_document, similarity):
    similarities = []
    for known_document in known_documents:
        similarities.append(similarity(known_document, unknown_document))
    return np.average(similarities)


# TODO: Implement it
def tot_count():
    return -1
