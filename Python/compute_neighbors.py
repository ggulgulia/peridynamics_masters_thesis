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

    if (horizon > 0.3*min(lx_max, ly_max)):
        refine_factor = math.ceil(horizon/(0.3*min(ly_max, lx_max)))
        print("cells are too coarse for peridynamic simulation\n",
                    "refining the cells with refine factor of %i\n"
                    %refine_factor)

        mesh = uniform_refine_triangles(mesh, refine_factor)
        return  peridym_set_horizon(mesh)
    else:
        return mesh, horizon



#def peridym_compute_neighbors(mesh, horizon):
#    """TODO: Docstring for peridym_compute_neighbors.
#    :returns: neighbor_list 
#    """
#    neighbor_list = []
#    elems_centroid = get_elem_centroid(mesh)
#    for cellX, cellY, cellID in elems_centroid:
#
#        temp = elems_centroid
#        temp.remove(elems_centroid[cellID])
#
#        for nextCellX, nextCellY, nextCellID in temp:
#            dist = math.sqrt((cellX - nextCellX)**2 + (cellY - nextCellY)**2)
#            if dist < horizon:
#                neighbor_list.append([nextCellX, nextCellY, nextCellID])
#
#    return neighbor_list 
