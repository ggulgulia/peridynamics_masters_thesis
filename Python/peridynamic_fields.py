from mesh_tools import *
from helper import *
import peridynamic_influence_function_manager as ifm
import peridynamic_compute_neighbors as nbrs


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

    nbr_bnd_vector_lst = []
    nbr_bnd_len_lst = []
    nbr_infl_fld_lst = []
    mw = np.zeros(len(elem_centroid)) #m is wighted volume

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
            
            mw[i] += curr_infl*curr_bnd_len**2*elem_area[idx]

            curr_node_bnd_lst.append(curr_bnd_vctr)
            curr_node_bnd_len_lst.append(curr_bnd_len)
            curr_node_infl_fld_lst.append(curr_infl)

        nbr_bnd_vector_lst.append(curr_node_bnd_lst)
        nbr_bnd_len_lst.append(curr_node_bnd_len_lst)
        nbr_infl_fld_lst.append(curr_node_infl_fld_lst)

    return nbr_lst, nbr_bnd_vector_lst, nbr_bnd_len_lst, nbr_infl_fld_lst, mw


def peridym_compute_global_force_density(nbr_lst, nbr_bnd_vct_lst, nbr_bnd_len_lst, 
                                        nbr_infl_fld_lst, mw, nbr_rel_disp_lst,
                                        nbr_ext_scalar_st_lst, theta_vct, disp_vct,
                                        elem_area ):
    k = 25.5 #shear modulus of aluminum (google)
    mu = 6.8e10 #bulk modulus of aluminum (google)
    """
    TODO: Docstring for peridym_compute_globl_force_density.

    input:
    -----
        nbr_lst                         :(list of int)peridynamic neighbor list 
        nbr_bnd_vct_lst                 :(list of floats)vector coefficeients of bonds in nbr_lst 
        nbr_bnd_len_lst                 :(list of floats)length of bond in nbr_lst 
        nbr_infl_fld_lst                :(list of floats)influence field in nbr_lst 
        mw                              :(np.array of float) weighted volume
        nbr_rel_disp_lst                :(list of floats) relative displacement of bonds 
                                         in nbr_lst computed using the disp_vect
        nbr_ext_scalar_st_lst           :(list of floats) extension scalar state of bonds
                                         in the nbr_lst (see reference above)
        theta_vct                       :(np.array of floats) dilatation vector 
        elem_area                       :(list of floats) element area in mesh  
        disp_vct                        :(np.array of floats) displacement vector
                                         of each node in the mesh
    
    returns
    --------
        force_density_vct               :np.array of floats
        nbr_force_scalar_st_lst         :peridym list of doubles 
        nbr_deformed_bnd_unit_vct_lst   :peridym list of doubles
        ed                              :deviatoric extension scalar state, peridym list 
        
    """
    force_density_vct = np.zeros((len(disp_vct),2),dtype=float)
    nbr_force_scalar_st_lst = []
    nbr_deformed_bnd_unit_vct_lst = []
    ed = [] #deviatoric extension scalar state

    for i in range(len(disp_vct)):
        curr_node_nbr_lst = nbr_lst[i]
        curr_force_scalar_st_lst = []
        curr_deformed_bnd_unit_vct_lst = []
        curr_ed = []
        
        curr_ext = nbr_ext_scalar_st_lst[i]
        curr_bnd_len_lst = nbr_bnd_len_lst[i]
        curr_node_infl_fld_lst = nbr_infl_fld_lst[i]
        curr_bnd_vct_lst = nbr_bnd_vct_lst[i]
        curr_node_rel_disp_lst = nbr_rel_disp_lst[i]

        for j, idx in enumerate(curr_node_nbr_lst):
            eed = curr_ext[j] - theta_vct[i]*curr_bnd_len_lst[j]/3
            t = (3*k*theta_vct[i]*curr_bnd_len_lst[j] + 15*mu*eed)*curr_node_infl_fld_lst[j]/mw[i]
            deformed_bnd_vct = np.array(vect_sum(curr_bnd_vct_lst[j], curr_node_rel_disp_lst[j]))
            deformed_bnd_unit_vct = deformed_bnd_vct/mod(deformed_bnd_vct)

            force_density_vct[i] += t*deformed_bnd_unit_vct*elem_area[idx]
            force_density_vct[j] -= t*deformed_bnd_unit_vct*elem_area[i]

            curr_deformed_bnd_unit_vct_lst.append(deformed_bnd_unit_vct)
            curr_ed.append(eed)
            curr_force_scalar_st_lst.append(t)

        nbr_force_scalar_st_lst.append(curr_force_scalar_st_lst)
        nbr_deformed_bnd_unit_vct_lst.append(curr_deformed_bnd_unit_vct_lst)
        ed.append(curr_ed)

    return nbr_force_scalar_st_lst, nbr_deformed_bnd_unit_vct_lst, ed, force_density_vct
