import os
import jsonhandler
from multiprocessing import Pool, cpu_count
from representation_spaces import *
from main import cores_to_leave_over, pickle_files_dir, attribution_dataset_data_dir, transformationdir
import sys
import pickle
import time

def transform_data():
    corpus_name = sys.argv[1]
    assert corpus_name != ''

    pool = Pool(processes=cpu_count()-cores_to_leave_over)

    #if not os.path.exists(os.path.join('corpora_texts', dataset)):
    #    if not os.path.exists('corpora_texts'):
    #        os.makedirs('corpora_texts')
    candidates = jsonhandler.candidates
    unknowns = jsonhandler.unknowns
    jsonhandler.loadJson(os.path.join(attribution_dataset_data_dir, corpus_name))
    jsonhandler.loadTraining()

    corpus = []
    for author in candidates:
        for file in jsonhandler.trainings[author]:
            corpus.append(jsonhandler.getTrainingText(author, file))
    for unknown in unknowns:
        corpus.append(jsonhandler.getUnknownText(unknown))

    if not os.path.exists(os.path.join(pickle_files_dir, corpus_name)):
        os.makedirs(os.path.join(pickle_files_dir, corpus_name))
    corpus_file = open(os.path.join(pickle_files_dir, corpus_name, 'all_text_files.pickle'), "wb")
    pickle.dump(corpus, corpus_file, protocol=pickle.HIGHEST_PROTOCOL)
    corpus_file.close()

    args = []
    for author in candidates:
        for file in jsonhandler.trainings[author]:
            args.append((author, file, pickle_files_dir, attribution_dataset_data_dir, corpus_name))
    # TODO: Also for unknown texts
    author = 'unknown'
    for file in unknowns:
        args.append((author, file, pickle_files_dir, attribution_dataset_data_dir, corpus_name))
    result = pool.map_async(transform_write_text, args).get()
    #result.wait()
    pool.close()
    pool.join()



def transform_write_text(arg):
    (author, file, pickle_files_dir, attribution_dataset_data_dir, corpus_name) = arg
    jsonhandler.loadJson(os.path.join(attribution_dataset_data_dir, corpus_name))
    if author != 'unknown':
        text = jsonhandler.getTrainingText(author, file)
    else:
        text = jsonhandler.getUnknownText(file)
    corpus_file = open(os.path.join(pickle_files_dir, corpus_name, 'all_text_files.pickle'), "rb")
    corpus = pickle.load(corpus_file)
    corpus_file.close()
    for representation_space in [representation_space6,
                                 representation_space7,
                                 representation_space8,
                                 representation_space678]:
        if not jsonhandler.existTransformedTrainingText(author, file, representation_space.__name__,
                                                        transformationdir = transformationdir):
            print('File does not exist')
            content = representation_space(text)
            jsonhandler.pickleTransformedTrainingText(author, file, content, representation_space.__name__,
                                                      transformationdir = transformationdir)
        else:
            print('File already exists')

    for representation_space in [representation_space1, representation_space2, representation_space3,
                                 representation_space4, representation_space5]:
        if not jsonhandler.existTransformedTrainingText(author, file, representation_space.__name__,
                                                        transformationdir = transformationdir):
            print('File does not exist')
            content = representation_space(text, corpus)
            jsonhandler.pickleTransformedTrainingText(author, file, content, representation_space.__name__,
                                                      transformationdir = transformationdir)
        else:
            print('File already exists')

    jsonhandler.reset_state()
