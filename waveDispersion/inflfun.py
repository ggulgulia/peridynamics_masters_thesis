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


def omega1(domain, horizon):
    """
    returns a unit influence function

    :domain: TODO
    :horizon: TODO
    :returns: TODO

    """
    return np.ones(len(domain), dtype=float)

def omega2(domain, horizon, elem_cent=0.0):
    """
    returns an inverted parabolic decay function (higher weights to long range forces) for an peridynamic centroid located around zero 

    input:
    ------
        domain: np.array of discritized domain
        horizon: float value
    returns:
    --------
        see omega2 of equation 13
    """

    return ((domain-elem_cent)/horizon)**2

def omega3(domain, horizon, p=1.0, elem_cent=0.0):
    """
    returns a gaussian decay influence function
    centered around zero
    for a 1D domain 
    see omega3 in equation 13
    TODO : Debug to get correct results
    """
    N = 1.0 #1 for 1d, 2 for 2D and so on
    omega0 = 3.14159**(-0.5*N)*((p/horizon)**N)
    bnd_vct = domain - elem_cent 
    return omega0*np.exp(-((p*bnd_vct/horizon)**2))
