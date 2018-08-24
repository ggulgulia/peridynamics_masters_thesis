from mesh_tools import *
from helper import *
import influence_function_manager as ifm
import compute_neighbors as nbrs


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
    nbr_lst = nbrs.peridym_compute_neighbors(mesh, horizon)
    elem_centroid = get_elem_centroid(mesh)
    elem_area = get_elem_areas(mesh)

    bnd_vector_lst = []
    bnd_len_lst = []
    infl_fld_lst = []
    m = np.zeros(len(elem_centroid)) #m is wighted volume

    for i in range(len(elem_centroid)):
        curr_node_coord = elem_centroid[i]
        
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
            #curr_nbr_id = curr_node_nbr_lst[j]
            curr_nbr_coord = elem_centroid[curr_node_nbr_lst[j]]
            
            curr_bnd_len = compute_distance(curr_nbr_coord, curr_node_coord)
            
            curr_infl  = ifm.gaussian_influence_function(curr_bnd_len, horizon)
            
            curr_bnd_vctr = vect_diff(curr_nbr_coord, curr_node_coord)
            #curr_bnd_vctr = [curr_nbr_coord[0] -curr_node_coord[0], curr_nbr_coord[1]-curr_node_coord[1], curr_bond_len]
            
            m[i] += curr_infl*curr_bnd_len**2*elem_area[idx]

            curr_node_bnd_lst.append(curr_bnd_vctr)
            curr_node_bnd_len_lst.append(curr_bnd_len)
            curr_node_infl_fld_lst.append(curr_infl)

        bnd_vector_lst.append(curr_node_bnd_lst)
        bnd_len_lst.append(curr_node_bnd_len_lst)
        infl_fld_lst.append(curr_node_infl_fld_lst)

    return nbr_lst, bnd_vector_lst, bnd_len_lst, infl_fld_lst, m


def peridym_compute_extension(nbr_lst, nbr_bnd_vct_lst,
                                nbr_bnd_ln_lst, infl_fld_lst, 
                                mw, disp_vect):
    """
    routine for calculation of internal force density for a linear
    peridynamic solid material with gaussian influence function
    Refer ch5, algo2 of Handbook of peridynamic modelling from
    Silling etal
    
    input:
    ------
        :nbr_lst: peridynamic neighborhood list
        :nbr_bnd_vct_lst: peridynamic coefficients of vector bond coordinates
                          in the neighborhood list
        :nbr_bnd_ln_lst: length of all bonds in the nbr_lst 
        :inf_fld_lst: list of values of influence field for all nodes in discretization
        :disp_vect: list of the positions of displaced nodes 
    
    returns: 
    --------
    rel_displ_lst : list/np.array 
        array of relative displacement
        in peridynamic neighborhood list
    e             : list/np.array
        extension scalar state in the neigborhood list
        of all nodes 
    -------
    """

    #no of nodes
    length = len(nbr_lst)
    force_vect = np.zeros(length, dtype=float) #internal force
    theta_vect = np.zeros(length, dtype=float) #dilatation
    e = [] #extension scalar state
    rel_displ_lst = [] # eta in the pseudo code
    for i in range(length):
        curr_node_nbr_lst = nbr_lst[i]

        curr_node_rel_displ_lst  = []
        e_local = []
        for j, idx in enumerate(curr_node_nbr_lst):
            rel_displ = displ_vect[idx] -  displ_vect[i]

            curr_bnd_vct_coord = nbr_bnd_vct_lst[i][j]
            curr_node_rel_displ.append(rel_displ)

            bond_vect_plus_rel_displ = mod(vect_sum(curr_bnd_vct_coor, rel_displ)) #define mod function in helper
            ee = (np.array(bond_vect_plus_rel_displ) - np.abs(curr_bnd_vct_coord)).tolist() 
            e_local.append(ee)

            theta_vect[i] += 3*infl_fld_lst[i][j]*mod(curr_bnd_vct_coord)*ee*elem_area[idx] #define mod function in helper


        rel_displ_lst.append(curr_node_rel_displ_lst)
        e.append(e_local)

    return rel_displ_lst, e

