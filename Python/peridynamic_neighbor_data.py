from fenics_mesh_tools import *
from helper import *
import timeit as tm
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
        
        
        mesh = refine(mesh)
        print_mesh_stats(mesh)        

        return  peridym_set_horizon(mesh)
    else:
        return mesh, horizon


def peridym_compute_neighbors(mesh, horizon, structured_mesh=False):
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
    structured_mesh: boolean, True if we use structured cell config
                     from FEniCS mesh
    returns
    -------
        neighbor_list: np.array
            list of peridynamic neighbors for the 
            given peridynamic horizon. This list contains 
            for each node, its neighbors in the peridynamic
            horizon 
    """
    start = tm.default_timer()
    print("computing the neighbor list of the mesh for horizon size of %f" %horizon)
    neighbor_lst = []

    if(structured_mesh):
        cell_cent = structured_cell_centroids(m)
    else:
        cell_cent = get_cell_centroids(mesh)
    
    num_cells = len(cell_cent)
    for i in range(num_cells):
        curr_dist = 0.0
        curr_neighbor_lst = []

        for j in range(i):
            curr_dist = la.norm(cell_cent[i] - cell_cent[j],2)
            if curr_dist <= horizon : 
                curr_neighbor_lst.append(j) # appending the element ID to neighbor_list

        for j in range(i+1, num_cells):
            curr_dist =  la.norm(cell_cent[j] - cell_cent[i],2)
            if curr_dist <= horizon : 
                curr_neighbor_lst.append(j) # appending the element ID to neighbor_list

        neighbor_lst.append(np.array(curr_neighbor_lst))
    print("time taken for computation of naive neighbor list for the given mesh is %4.3f sec"%(tm.default_timer()-start))
    return np.array(neighbor_lst)


def peridym_compute_weighted_volume(mesh, nbr_lst, horizon, omega_fun, structured_mesh=False):
    """
    computes the weighted volume of the peridynammic 
    mesh based on influence function and horizon value
    """
    if(structured_mesh):
        cell_cent = structured_cell_centroids(mesh)
        cell_vol = structured_cell_volumes(mesh)
    else:
        cell_cent = get_cell_centroids(mesh)
        cell_vol = get_cell_volumes(mesh)

    mw = np.zeros(len(cell_vol), dtype=float) #m is wighted volume

    for i in range(len(cell_cent)):
        curr_node_coord = cell_cent[i]
        
        #declare empty lists for current node neighbor
        #attributes like neighbor bond vector, bond len,
        #and influence field 
        #refer ch5 algo1  of handbook of peridynamic modelling
        #by silling etal 

        curr_nbr_lst = nbr_lst[i] 
        curr_nbr_bnd_vct = cell_cent[curr_nbr_lst] - curr_node_coord
        curr_nbr_bnd_len = la.norm(curr_nbr_bnd_vct, 2, axis=1)
        mw[i] = sum(omega_fun(curr_nbr_bnd_vct, horizon)*curr_nbr_bnd_len**2*cell_vol[curr_nbr_lst])

    return mw

   
def peridym_get_neighbor_data(mesh, horizon, omega_fun, structured_mesh=False):
    """
    this function computes the bond vector coordinates
    for each element in the neighborhood list of the 
    mesh
    
    input:
    ------
        mesh : meshpy.MeshInfo mesh data
        horizon : float, peridynamic horizon
        omega_fun: pointer to the influence function
        structured_mesh: boolean, if peridiynamic mesh is structured
    returns:
    -------
        nbr_lst         :
        nbr_bnd_vct_lst : np.array/list of doubles
            bond vector for each element in neighborhood list 
        nbr_bnd_len_lst :
        nbr_infl_fld_lst:
        mw              : weighted mass 
        

    """
    nbr_lst = peridym_compute_neighbors(mesh, horizon, structured_mesh)
    start = tm.default_timer()
    print("computing the remaining peridynamic neighbor data for the mesh with horizon: %4.2f" %horizon)

    mw = peridym_compute_weighted_volume(mesh, nbr_lst, horizon, omega_fun, structured_mesh)

    if(structured_mesh):
        cell_cent = structured_cell_centroids(mesh)
        cell_vol  = structured_cell_volumes(mesh)
    else:
        cell_cent = get_cell_centroids(mesh)
        cell_vol = get_cell_volumes(mesh)

    nbr_bnd_vector_lst = []
    nbr_bnd_len_lst = []
    nbr_infl_fld_lst = []

    for i in range(len(cell_cent)):
        curr_node_coord = cell_cent[i]
        
        #declare empty lists for current node neighbor
        #attributes like neighbor bond vector, bond len,
        #and influence field 
        #refer ch5 algo1  of handbook of peridynamic modelling
        #by silling etal 

        curr_nbr_lst = nbr_lst[i] 
        curr_nbr_bnd_vct = cell_cent[curr_nbr_lst] - curr_node_coord
        curr_nbr_bnd_len = la.norm(curr_nbr_bnd_vct, 2, axis=1)

        nbr_bnd_vector_lst.append(curr_nbr_bnd_vct)
        nbr_bnd_len_lst.append(curr_nbr_bnd_len)

    print("time taken for computation of remaining neighbor data for the given mesh is %4.3f seconds"%(tm.default_timer()-start))
    
    return nbr_lst, nbr_bnd_vector_lst, nbr_bnd_len_lst, mw
