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


def peridym_initialize(mesh, horizon):
    """
    this function computes the bond vector coordinates
    for each element in the neighborhood list of the 
    mesh
    
    input:
    ------
        neighbor_list : list/np.arry of int
            peridynamic neighborhood list
        elem_centroid : list/np.array of doubles
            coordinates of centroid of each element
    returns
    -------
        bond_vector_list : np.array/list of doubles
            bond vector for each element in neighborhood list 

    """
    nbr_lst = peridym_compute_neighbors(mesh, horizon)
    elem_centroid = get_elem_centroid(mesh)
    elem_area = get_elem_areas(mesh)

    bnd_vector_lst = []
    bnd_len_lst = []
    infl_fld_lst = []

    for i in range(len(elem_centroid)):
        curr_node_coord = elem_centroid[i]
        
        #declare empty lists for current node neighbor
        #attributes like neighbor bond vector, bond len,
        #and influence field 
        curr_node_bnd_lst = []
        curr_node_bnd_len_lst = []
        curr_node_infl_fld_lst = []
        m = np.zeros(len(elem_centroid)) #m is wighted volume
        #refer ch5 algo1  of handbook of peridynamic modelling
        #by silling etal 

        curr_node_nbr_lst = nbr_lst[i] 
        for j in range(len(curr_node_nbr_lst)):
            #curr_nbr_id = curr_node_nbr_lst[j]
            curr_nbr_coord = elem_centroid[curr_node_nbr_lst[j]]
            
            curr_bnd_len = compute_distance(curr_nbr_coord, curr_node_coord)
            
            curr_infl  = ifm.gaussian_influence_function(curr_bnd_len, horizon)
            
            curr_bnd_vctr = vect_diff(curr_nbr_coord, curr_node_coord)
            #curr_bnd_vctr = [curr_nbr_coord[0] -curr_node_coord[0], curr_nbr_coord[1]-curr_node_coord[1], curr_bond_len]
            
            m[i] += curr_infl*curr_bnd_len**2*elem_area[j]

            curr_node_bnd_lst.append(curr_bnd_vctr)
            curr_node_bnd_len_lst.append(curr_bnd_len)
            curr_node_infl_fld_lst.append(curr_infl)

        bnd_vector_lst.append(curr_node_bnd_lst)
        bnd_len_lst.append(curr_node_bnd_len_lst)
        infl_fld_lst.append(curr_node_infl_fld_lst)

    return nbr_lst, bnd_vector_lst, bnd_len_lst, infl_fld_lst, m
