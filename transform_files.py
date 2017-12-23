import os
import jsonhandler
from multiprocessing import Pool
from representation_spaces import *
from main import attribution_dataset_data_dir
import sys


def transform_data():
    corpus_name = sys.argv[1]
    dataset = attribution_dataset_data_dir + '/' + corpus_name

    pool = Pool(processes=30)

    if not os.path.exists(os.path.join('corpora_texts', dataset)):
        if not os.path.exists('corpora_texts'):
            os.makedirs('corpora_texts')
        candidates = jsonhandler.candidates
        unknowns = jsonhandler.unknowns
        jsonhandler.loadJson(dataset)
        jsonhandler.loadTraining()
        corpus = []
        for author in candidates:
            for file in jsonhandler.trainings[author]:
                args = (author, file)
                pool.map_async(transform_write_text,args,"test")
        # TODO: Also for unknown texts


def transform_write_text(arg):
    author, file = arg
    text = jsonhandler.getTrainingText(author, file)
    for representation_space in [#lambda document: representation_space1(document, corpus_each_problem_as_one_text),
                                     #lambda document: representation_space2(document, corpus_each_problem_as_one_text),
                                     #lambda document: representation_space3(document, corpus_each_problem_as_one_text),
                                     #lambda document: representation_space4(document, corpus_each_problem_as_one_text),
                                     #lambda document: representation_space5(document, corpus_each_problem_as_one_text),
                                     lambda document: representation_space6(document),
                                     lambda document: representation_space7(document),
                                     lambda document: representation_space8(document),
                                     lambda document: representation_space678(document)]:
        if not jsonhandler.existTransformedTrainingText(author, file, representation_space.__name__):
            content = representation_space(text)
            jsonhandler.pickleTransformedTrainingText(author, file, content, representation_space.__name__)
        #else:
        #    content = jsonhandler.loadTransformedTrainingText(author, file, representation_space.__name__)
