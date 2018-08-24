from helper import *

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
    -------
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

