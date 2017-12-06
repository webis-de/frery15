import os
import urllib
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import yaml
import copy
import sys

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

#TODO: Change for new requirements
def load_text_corpus(representationSpaces):
    corpora = []
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
