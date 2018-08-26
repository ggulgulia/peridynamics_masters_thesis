import meshpy.triangle as tri
from mesh_tools import *
from helper import *
import influence_function_manager as ifm
from meshpy.geometry import bounding_box


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


def peridym_compute_neighbors(mesh, horizon):
    """
    given a mesh and a horizon, this function
    rturns for each node in the mesh the 
    neighborhood list where each node has all
    the neighboring elements that fall within
    the horizon (horizon is a float value)

    input
    -----
    mesh : meshpy.triangle mesh
    horizon : folat , length of peridynamic horizon

    returns
    -------
        neighbor_list: np.array
            list of peridynamic neighbors for the 
            given peridynamic horizon. This list contains 
            for each node, its neighbors in the peridynamic
            horizon 
    """
    print("computing the neighbor list of the mesh (with the trial function) for horizon size of %f" %horizon)
    neighbor_lst = []

    elems = np.array(mesh.elements)
    points = np.array(mesh.points)
    elem_centroid = get_elem_centroid(mesh)

    for i in range(len(elems)):
        #temp.remove(elem_centroid[i])
        curr_dist = 0.0
        curr_neighbor_lst = []

        for j in range(i):
            curr_dist = compute_distance(elem_centroid[i], elem_centroid[j])
            if curr_dist <= horizon : 
                curr_neighbor_lst.append(j) # appending the element ID to neighbor_list

        for j in range(i+1, len(elems)):
            curr_dist = compute_distance(elem_centroid[i], elem_centroid[j])
            if curr_dist <= horizon : 
                curr_neighbor_lst.append(j) # appending the element ID to neighbor_list

        neighbor_lst.append(curr_neighbor_lst)

    return neighbor_lst

