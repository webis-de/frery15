def dissimilarity_counter_method(set_of_known_documents, unknown_document, threshold, similarity_measure):
    count = 0
    for known_document in set_of_known_documents:
        smin = 1
        for other_known_document in set_of_known_documents:
            if known_document == other_known_document:
                pass
            similarity = similarity_measure(known_document, other_known_document)
            if smin > similarity:
                smin = similarity
        if similarity_measure(unknown_document, known_document) > smin:
            count += 1
    if count > threshold:
        return True
    else:
        return False
