"""
cost_functions.py

Created by Gabriele Tolomei on 2016-06-29.
Copyright (c) 2016 Yahoo! Labs. All rights reserved.
"""
from scipy import spatial
from scipy.stats import *
import numpy as np
import pandas as pd
from core.logmanager import get_logger

# get the root logger
logger = get_logger('cost_function')

######################################## Normalize vector ################


def __normalize_vector(v):
    """
    This function checks wheter the given input vector is already normalized (i.e. if its L-2 norm is 1).
    If it does then the function just returns the input vector as it is, otherwise it normalizes it before
    returning it.
    Firstly, however, it also transforms the vector into a numpy.array if needed.

    Args:
        v (any sequence): the input vector

    Returns:
        v' (numpy.array): the normalized vector v' as a numpy.array
    """

    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm == 1:
        return v
    return v / norm

##########################################################################

# Count matches between two vect


def __count_matches(u, v):
    """
    This function returns the number of matching elements between two input vectors u and v.
    (Assumption: len(u) = len(v))

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        int: the number of matching elements between u and v
    """
    return len([i for i, j in zip(u, v) if i == j])

##########################################################################

# Count unmatches between two ve


def __count_unmatches(u, v):
    """
    This function returns the number of unmatching elements between two input vectors u and v.
    (Assumption: len(u) = len(v))

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        int: the number of unmatching elements between u and v
    """
    return len([i for i, j in zip(u, v) if i != j])

##########################################################################

# Compute the Unmatched Component Rate (UCR) between two


def unmatched_component_rate(u, v):
    """
    This function measures the distance between two input vectors u and v in terms of the ratio of
    different (i.e. unmatching) components, assuming len(u) = len(v)

    Example:
    u = [1, 2, 3, 4, 5, 4, 2, 2, 7, 10]
    v = [9, 2, 7, 6, 5, 4, 7, 6, 11, 7]
    len(u) = len(v) = 10
    unmatched_components = {i | u[i] != v[i]} = {0, 2, 3, 6, 7, 8, 9}
    unmatched_component_rate = |unmatched_components|/len(u) = 7/10 = 0.7

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the ratio of unmatched components between u and v, normalized by the length of the vectors
    """
    return round(__count_unmatches(u, v) / float(len(u)), 5)

##########################################################################

# Compute the Euclidean Distance between tw


def euclidean_distance(u, v):
    """
    This function returns the euclidean distance between two vectors u and v.

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the euclidean distance between u and v computed as the L-2 norm of the vector
                resulting from the difference (u - v)
    """

    u = __normalize_vector(u)
    v = __normalize_vector(v)
    return round(np.linalg.norm(u - v), 5)

##########################################################################

# Compute the Cosine Distance between two v


def cosine_distance(u, v):
    """
    This function returns the cosine distance between two vectors u and v.
    Invariant with respect to the magnitude of the vectors (i.e. scaling)

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the cosine distance between u and v
    """
    return round(spatial.distance.cosine(u, v), 5)

##########################################################################

# Compute the Jaccard Distance between two


def jaccard_distance(u, v):
    """
    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the Jaccard distance between u and v
    """
    return round(spatial.distance.jaccard(u,v), 5)

##########################################################################

# Compute the Pearson Correlation distance


def pearson_correlation_distance(u, v):
    """
    This function computes the distance between two input vectors u and v
    in terms of their Pearson's correlation coefficient.
    This coefficient is invariant to the magnitude of the vectors (i.e. scaling)
    and also to adding any constant to all elements

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the Pearson's correlation distance between u and v

    """
    u = u.flatten().tolist()
    v = v.flatten().tolist()
    rho = stats.pearsonr(u, v)[0]
    rho_d = 1 - rho
    return round(rho_d, 5)