from mesh_tools import *
from helper import *
import peridynamic_influence_function_manager as ifm
from meshpy.geometry import bounding_box


def peridym_set_horizon(mesh, horizon=None):

    """
    computes the optimal global horizon based on the given mesh,
    if the mesh is too coarse, this method refines the mesh
    and recomputes the horizon.
    
    optimal means that the horizon is 5 times the maximum lengnth
    of the edge of a cell in descritization.
    
    global horizon means that entire domain has only single 
    horizon value
    
    input:
    ------
        mesh    : FEniCS mesh 2D-tri/3D-tet
        horizon : float, peridynamic horizon
        
    returns:
    --------
        mesh :    FEniCS mesh (useful if the mesh is refined)
        horizon : float , global for the entire domain

    """
    el = mesh.hmax() #edge length
    if horizon is None:
        horizon = 2*max_len
        
    corner_min, corner_max = get_domain_bounding_box(mesh)

    lx_max = abs(corner_min[0] - corner_max[0])
    ly_max = abs(corner_min[1] - corner_max[1])

    #if horizon size is close to the bounding box size
    if (horizon > 0.3*min(lx_max, ly_max)): 
        print("cells are too coarse for peridynamic simulation\n",
                    "refining the cells\n"
                    %refine_factor)

        print("Old mesh stats:\nNumber of points: %i\n \
                Number of Cells: %i\n \
                max edge length: %4.5f\n \
                min edge length: %4.5f\n" \
                %(mesh.num_vertices(), mesh.num_cells(), \
                  mesh.hmax(), mesh.hmin()))

        mesh = refine(mesh)
        
        print("New mesh stats:\nNumber of points: %i\n \
                Number of Cells: %i\n \
                max edge length: %4.5f\n \
                min edge length: %4.5f\n" \
                %(mesh.num_vertices(), mesh.num_cells(), \
                  mesh.hmax(), mesh.hmin()))

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

    num_cells = mesh.num_cells()
    cell_centroid = get_cell_centroids_centroid(mesh)

    for i in range(len(elems)):
        #temp.remove(cell_cent[i])
        curr_dist = 0.0
        curr_neighbor_lst = []

        for j in range(i):
            curr_dist = la.norm(cell_cent[i] - cell_cent[j],2)
            if curr_dist <= horizon : 
                curr_neighbor_lst.append(j) # appending the element ID to neighbor_list

        for j in range(i+1, len(elems)):
            curr_dist =  la.norm(cell_cent[j] - cell_cent[i],2)
            if curr_dist <= horizon : 
                curr_neighbor_lst.append(j) # appending the element ID to neighbor_list

        neighbor_lst.append(np.array(curr_neighbor_lst))

    return np.array(neighbor_lst)


   
def peridym_get_neighbor_data(mesh, horizon):
    """
    this function computes the bond vector coordinates
    for each element in the neighborhood list of the 
    mesh
    
    input:
    ------
        mesh : meshpy.MeshInfo mesh data
        horizon : float, peridynamic horizon
    returns:
    -------
        nbr_lst         :
        nbr_bnd_vct_lst : np.array/list of doubles
            bond vector for each element in neighborhood list 
        nbr_bnd_len_lst :
        nbr_infl_fld_lst:
        mw              :
        

    """
    nbr_lst = peridym_compute_neighbors(mesh, horizon)
    cell_cent = get_cell_centroids(mesh)
    cell_vol = get_cell_volumes(mesh)

    nbr_bnd_vector_lst = []
    nbr_bnd_len_lst = []
    nbr_infl_fld_lst = []
    mw = np.zeros(len(cell_vol), dtype=float) #m is wighted volume

    for i in range(len(cell_cent)):
        curr_node_coord = cell_cent[i]
        
        #declare empty lists for current node neighbor
        #attributes like neighbor bond vector, bond len,
        #and influence field 
        curr_node_bnd_lst = []
        curr_node_bnd_len_lst = []
        curr_node_infl_fld_lst = []
        #refer ch5 algo1  of handbook of peridynamic modelling
        #by silling etal 

        curr_node_nbr_lst = nbr_lst[i] 
        for j, idx in enumerate(curr_node_nbr_lst):
        
            curr_nbr_coord = cell_cent[idx]
            curr_bnd_len = la.norm((cell_cent[idx] - cell_cent[i]), 2)  
            #curr_bnd_vctr = curr_nbr_coord - curr_node_coord
            #curr_bnd_len = np.linalg.norm(curr_bnd_vctr,2)
            curr_infl  = ifm.gaussian_influence_function(curr_bnd_len, horizon)            
            curr_bnd_vctr = vect_diff(curr_nbr_coord, curr_node_coord)            
            mw[i] += curr_infl*curr_bnd_len**2*cell_vol[idx]

            curr_node_bnd_lst.append(curr_bnd_vctr)
            curr_node_bnd_len_lst.append(curr_bnd_len)
            curr_node_infl_fld_lst.append(curr_infl)

        nbr_bnd_vector_lst.append(curr_node_bnd_lst)
        nbr_bnd_len_lst.append(curr_node_bnd_len_lst)
        nbr_infl_fld_lst.append(curr_node_infl_fld_lst)

    return nbr_lst, nbr_bnd_vector_lst, nbr_bnd_len_lst, nbr_infl_fld_lst, mw
