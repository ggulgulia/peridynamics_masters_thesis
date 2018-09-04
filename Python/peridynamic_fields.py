from mesh_tools import *
from helper import *
import peridynamic_influence_function_manager as ifm


def peridym_compute_extension(nbr_lst, nbr_bnd_vct_lst,
                                nbr_bnd_ln_lst, nbr_infl_fld_lst, 
                                mw, disp_vct, elem_area):
    """
    routine for calculation of internal force density for a linear
    peridynamic solid material with gaussian influence function
    Refer ch5, algo2 of Handbook of peridynamic modelling from
    Silling etal
    
    input:
    ------
        nbr_lst             :peridynamic neighborhood list
        nbr_bnd_vct_lst     :peridynamic coefficients of vector bond coordinates
                             in the neighborhood list
        nbr_bnd_ln_lst      :length of all bonds in the nbr_lst 
        inf_fld_lst         :list of values of influence field for all nodes in discretization
        disp_vct            :list of the positions of displaced nodes in the mesh
    
    returns: 
    --------
    rel_displ_lst           :list/np.array 
                            array of relative displacement
                            in peridynamic neighborhood list
    e                       :list/np.array
                             extension scalar state in the neigborhood list
                             of all nodes 
    """

    #no of nodes
    length = len(disp_vct)
    force_vect = np.zeros(length, dtype=float) #internal force
    theta_vct = np.zeros(length, dtype=float) #dilatation
    e = [] #extension scalar state
    rel_disp_lst = [] # eta in the pseudo code
    for i in range(length):
        curr_node_nbr_lst = nbr_lst[i]

        curr_node_rel_disp_lst  = []
        e_local = []
        for j, idx in enumerate(curr_node_nbr_lst):
            rel_disp_vct = vect_diff(disp_vct[idx], disp_vct[i])
            curr_node_rel_disp_lst.append(rel_disp_vct.tolist())

            curr_bnd_vct_coord = nbr_bnd_vct_lst[i][j]

            bond_vect_plus_rel_disp = mod(vect_sum(curr_bnd_vct_coord, rel_disp_vct)) #define mod function in helper
            ee = bond_vect_plus_rel_disp - mod(curr_bnd_vct_coord) 
            e_local.append(ee)

            theta_vct[i] += 3*nbr_infl_fld_lst[i][j]*mod(curr_bnd_vct_coord)*ee*elem_area[idx] #define mod function in helper


        rel_disp_lst.append(curr_node_rel_disp_lst)
        e.append(e_local)

    return rel_disp_lst, e, theta_vct



def peridym_compute_global_force_density(nbr_lst, nbr_bnd_vct_lst, nbr_bnd_len_lst, 
                                        nbr_infl_fld_lst, mw, nbr_rel_disp_lst,
                                        nbr_ext_scalar_st_lst, theta_vct, disp_vct,
                                        elem_area ):

    """
    this method computes the global force density for the peridynamic state
    for a given neighborhood data and a given displacement of nodes in the 
    descritization. 
    
    It is based on pseudocode algorithm 2, Ch5 of Handbook of Peridynamic Modeling
    by S.A Silling et.al

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
        nbr_force_scalar_st_lst         :peridym list of doubles 
        nbr_deformed_bnd_unit_vct_lst   :peridym list of doubles
        ed                              :deviatoric extension scalar state, peridym list 
        force_density_vct               :np.array of floats
    """
    k = 25.5*1e10 #shear modulus of aluminum (google)
    mu = 6.8*1e10 #bulk modulus of aluminum (google)
    
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
    


def peridym_compute_damage(mesh, horizon, disp_vct):
    """TODO: Docstring for peridym_compute_damage.

    :mesh: TODO
    :horizon: TODO
    :disp_vct: TODO
    :returns: TODO

    """
