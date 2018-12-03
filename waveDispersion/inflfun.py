import numpy as np
import numpy.linalg as la
import math


def gaussian_infl_fun(xi, horizon):
    """
    returns a gaussian influence function

    :xi: bond vector
    :horizon: peridynamic horizon 
    :returns: omega

    """

    omega = np.exp(-(la.norm(xi,2)**2/(horizon**2)))

    return omega

def unit_infl_fun(xi, horizon):
    """
    returns a uniform unit influnce function

    """

    return 1.0

def parabolic_decay(xi, horion):
    """
    returns a parabolic decay influence field function

    adapted from peridigm++ code
    """
    scaled_dist = la.norm(xi,2)/horizon

    if scaled_dist < 0.5:
        return 1.0
    else:
        return (-4.0*scaled_dist**2 + 4.0*scaled_dist) 

