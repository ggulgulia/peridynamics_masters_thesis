import meshpy.triangle as tri
from mesh_tools import *
from meshpy.geometry import bounding_box
import math


def peridym_set_horizon(mesh, horizon=None):

    """TODO: Docstring for peridym_set_horizon 
    :returns: horizon : float , global for the entire domain

    """
    el = get_edge_lengths(mesh)
    max_len = max(el)
    if horizon is None:
        horizon = 5*max_len
        
    p = np.array(mesh.points)
    corner_min = np.min(p, axis=0)
    corner_max = np.max(p, axis=0)

    lx_max = abs(corner_min[0] - corner_max[0])
    ly_max = abs(corner_min[1] - corner_max[1])

    if (horizon > 0.2*min(lx_max, ly_max)):
        refine_factor = math.ceil(horizon/(0.2*min(ly_max, lx_max)))
        print("cells are too coarse for peridynamic simulation\n",
                    "refining the cells with refine factor of %i\n"
                    %refine_factor)

        mesh = uniform_refine_triangles(mesh, refine_factor)
        return  peridym_set_horizon(mesh)
    else:
        return mesh, horizon

def compute_distance(coord1, coord2):
    """
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



def peridym_compute_neighbors(mesh, horizon):
    """TODO: Docstring for peridym_compute_neighbors.

    input
    -----
    mesh : meshpy.triangle mesh
    horizon : folat , length of peridynamic horizon

    returns
    -------
        neighbor_list: np.array
            list of peridynamic neighbors for the 
            given peridynamic horizon
    """
    print("computing the neighbor list of the mesh for horizon size of %f" %horizon)
    neighbor_list = []

    elems = np.array(mesh.elements)
    points = np.array(mesh.points)
    elem_centroid = get_elem_centroid(mesh)

    for i in range(len(elems)):
        temp = elem_centroid.copy()
        temp.remove(elem_centroid[i])
        curr_dist = 0.0
        curr_neighbor_list = []
        for j in range(len(temp)):
            curr_dist = compute_distance(elem_centroid[i][0:2], temp[j][0:2])
            if curr_dist <= horizon : 
                curr_neighbor_list.append(temp[j][2]) # appending the element ID to neighbor_list

        neighbor_list.append(curr_neighbor_list)

    return neighbor_list
