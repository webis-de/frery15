import os
import jsonhandler
from multiprocessing import Pool
from representation_spaces import *
from main import attribution_dataset_data_dir
import sys
import pickle
import time

def transform_data():
    corpus_name = sys.argv[1]
    assert corpus_name != ''
    dataset = attribution_dataset_data_dir + '/' + corpus_name

    pool = Pool(processes=30)

    #if not os.path.exists(os.path.join('corpora_texts', dataset)):
    #    if not os.path.exists('corpora_texts'):
    #        os.makedirs('corpora_texts')
    candidates = jsonhandler.candidates
    unknowns = jsonhandler.unknowns
    jsonhandler.loadJson(dataset)
    jsonhandler.loadTraining()

    corpus = []
    for author in candidates:
        for file in jsonhandler.trainings[author]:
            corpus.append(jsonhandler.getTrainingText(author, file))
    for unknown in unknowns:
        corpus.append(jsonhandler.getUnknownText(unknown))

    corpus_file = open(os.path.join(dataset,'all_text_files.pickle'), "wb")
    pickle.dump(corpus, corpus_file, protocol=pickle.HIGHEST_PROTOCOL)
    corpus_file.close()

    args = []
    for author in candidates:
        for file in jsonhandler.trainings[author]:
            args.append((author, file, dataset))
    # TODO: Also for unknown texts
    author = 'unknown'
    for file in unknowns:
        args.append((author, file, dataset))
    result = pool.map_async(transform_write_text, args).get()
    #result.wait()
    pool.close()
    pool.join()



def transform_write_text(arg):
    (author, file, dataset) = arg
    jsonhandler.loadJson(dataset)
    if author != 'unknown':
        text = jsonhandler.getTrainingText(author, file)
    else:
        text = jsonhandler.getUnknownText(file)
    corpus_file = open(os.path.join(dataset, 'all_text_files.pickle'), "rb")
    corpus = pickle.load(corpus_file)
    corpus_file.close()
    for representation_space in [representation_space6,
                                 representation_space7,
                                 representation_space8,
                                 representation_space678]:
        if not jsonhandler.existTransformedTrainingText(author, file, representation_space.__name__):
            print('File does not exist')
            content = representation_space(text)
            jsonhandler.pickleTransformedTrainingText(author, file, content, representation_space.__name__)
        else:
            print('File already exists')
            content = jsonhandler.loadTransformedTrainingText(author, file, representation_space.__name__)

    for representation_space in [representation_space1, representation_space2, representation_space3,
                                 representation_space4, representation_space5]:
        if not jsonhandler.existTransformedTrainingText(author, file, representation_space.__name__):
            print('File does not exist')
            content = representation_space(text, corpus)
            jsonhandler.pickleTransformedTrainingText(author, file, content, representation_space.__name__)
        else:
            print('File already exists')
            content = jsonhandler.loadTransformedTrainingText(author, file, representation_space.__name__)
    if not os.path.exists(os.path.join(dataset, author, file+'_'+representation_space1.__name__+'.pickle')):
        raise FileNotFoundError("Ahh")

    jsonhandler.encoding = ""
    jsonhandler.language = ""
    jsonhandler.corpusdir = ""
    jsonhandler.upath = ""
    jsonhandler.candidates = []
    jsonhandler.unknowns = []
    jsonhandler.trainings = {}
    jsonhandler.trueAuthors = []

transform_data()
