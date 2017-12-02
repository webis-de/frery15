from sklearn.metrics import pairwise
from scipy.stats.stats import pearsonr
import numpy as np

def similarity(vector1, vector2):
    raise NotImplementedError('Please use an implemented distance method')

def cosine_similarity(vector1, vector2):
    return pairwise.cosine_similarity(vector1, vector2)


def correlation_coefficient(vector1, vector2):
    return pearsonr(vector1, vector2)[0]


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)
