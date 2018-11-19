"""
this file is for managing the calculation of influence function 
for peridynamic function
"""
from helper import *

def gaussian_influence_function(bnd_len, horizon):
    """TODO: Docstring for gaussian_influence_function.
    :returns: TODO

    """
    return -np.exp(-(bnd_len**2/horizon**2))


def parabolic_influence_func(bnd_len, horizon):
    """TODO: Docstring for parabolic_influence_func.
    :returns: TODO

    """
    pass

def cubic_influence_func(bnd_len, horizon):
    """
    TODO: complete implementation
    """
    pass


def linear_influence_fun(bnd_len, horizon):
    """
    TODO: complete implementation
    """
    pass
