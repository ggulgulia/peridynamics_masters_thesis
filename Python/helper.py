import math
import numpy as np

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
