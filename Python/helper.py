import math
import numpy.linalg as la
import numpy as np


"""
this file defines some of the auxillary vector operations
that are repeatedly used in peridynamic modelling
"""

def vect_diff(vect1, vect2):
    """
    input:
    ------
        vect1, vect2 : np.array double
            two vectors of arbitrary length
    
    returns: the difference of two vectors
    -------
    """
    if type(vect1) == list:
        vect1 = np.array(vect1, copy=True, dtype = float)
        vect2 = np.array(vect2, copy=True, dtype = float)

        return (vect1 - vect2).tolist()

    else :
        return vect1 - vect2
        

def vect_sum(vect1, vect2):
    """
    input:
    ------
        vect1, vect2 : array double
            two vectors of arbitrary length
            both vectors are either list or 
            numpy array 
    
    returns: the sum of two vectors
            return type is same as input type 
    -------
    """
    if type(vect1) == list:
        vect1 = np.array(vect1, copy=True, dtype = float)
        vect2 = np.array(vect2, copy=True, dtype = float)

        return (vect1 + vect2).tolist()

    else :
        return vect1 + vect2

def mod(vector):
    """
    returns the modulus (mathematically second norm) 
    of the vector of arbitrary dimension
    
    input
    -----
        vector:
    output
    ------
        scalar, second norm of the vector 

    """
    if type(vector) == list:
        vector = np.array(vector, copy=True, dtype = float)
        return la.norm(vector,2)

    else:
        return la.norm(vector,2)


def compute_distance(coord1, coord2):
    """
    TODO : compute distance for vectors of arbitrary lengths
    
    computes the (abs value) of distance
    between the two coordinates 'coord1' and 'coord2'
    (for 2D only)
    
    input:
    ------
        coord1, coord2 : list of floats
            two coordinates in a 2D space
    returns:
    --------
        distance float:
            distance between coord1 and coord2
    """
    return  math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def unit_vect(vector):
    """
    for a given input vector, the function
    returns its corresponding unit vector 

    input:
    -----
    vector: np.array() or list of doubles
        vector of arbitrary dimension

    output
    ------
    unit vector corresponding to input vector
    output vector has same type(np.array/list) as input vector 
    """
    
    if type(vector) == list:
        return (vector/mod(vector)).tolist()

    else:
        return vector/mod(vector)
