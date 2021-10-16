"""
This script defines the class that can be used for Actionable Feature Tweaking.
"""
__author__ = "Subhas K. Ghosh"
__copyright__ = "Copyright (C) 2021 GTM.ai"
__version__ = "1.0"
from scipy import spatial
from scipy.stats import *
import numpy as np

def __normalize_vector(v):
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm == 1:
        return v
    return v / norm

def __count_matches(u, v):
    return len([i for i, j in zip(u, v) if i == j])

def __count_unmatches(u, v):
    return len([i for i, j in zip(u, v) if i != j])

def unmatched_component_rate(u, v):
    return round(__count_unmatches(u, v) / float(len(u)), 5)

def euclidean_distance(u, v):
    u = __normalize_vector(u)
    v = __normalize_vector(v)
    return round(np.linalg.norm(u - v), 5)

def cosine_distance(u, v):
    return round(spatial.distance.cosine(u, v), 5)

def jaccard_distance(u, v):
    return round(spatial.distance.jaccard(u,v), 5)

def pearson_correlation_distance(u, v):
    u = u.flatten().tolist()
    v = v.flatten().tolist()
    rho = stats.pearsonr(u, v)[0]
    rho_d = 1 - rho
    return round(rho_d, 5)

def chebyshev_distance(u, v):
    return round(spatial.distance.chebyshev(u, v), 5)

def canberra_distance(u, v):
    return round(spatial.distance.canberra(u, v), 5)
