from mesh_tools import *
from helper import *



def compute_purturb_exten(node_num, purt_node, trv_lst, trv_bnd_vct_lst, 
                          trv_bnd_len_lst, trv_infl_fld_lst, mwi, elem_cent, elem_area):

    """
    returns the extension data (e, ed, unit vector state Mij, i refers to reference
    node, j refers to its corresponding neighbor in horizon with which node i makes
    a bond)

    input:
    ------
        node_num           : TODO
        purt_node          : np.array, (1,dim) centroid of purturbed node
        trv_lst            : traversal list for and including purturbed node
        trv_bnd_vct_lst    : traversal list of vectors of bonds
        trv_bnd_len_lst    : traversal list for bond lengths
        trv_infl_fld_lst   : traversal list for influence field
        mwi                : weighted volume of purturbed node
        elem_cent          : list of element centroid
        elem_area          : list of element area

    returns
    -------
        e                  : extension scalar state for trv_lst
        eta                : deformend bond vector list for trv_lst
        theta_i            : dilatation for purturbed node
        Mij                : deformed unit vector state
    """

    e = []
    eta = []
    Mij = []
    theta_i = 0. #dilatation for current node
    mwi_inv = 1/mwi
    for i, idx in enumerate(trv_lst):
        #eta_loc = vect_diff(purt_node, elem_cent[idx]) #deformation relative to only one bond 
        eta_loc = elem_cent[idx] - purt_node
        #relative to only one node which happens to be the current purturbed node
        #eta_plus_psi = vect_sum(trv_bnd_vct_lst[i], eta_loc)
        eta_plus_psi = trv_bnd_vct_lst[i] + eta_loc
        e_loc = mod(eta_plus_psi) - trv_bnd_len_lst[i]
        theta_i += 3*mwi_inv*trv_infl_fld_lst[i]*trv_bnd_len_lst[i]*e_loc*elem_area[idx]

        Mij.append(eta_plus_psi/mod(eta_plus_psi))
        eta.append(eta_loc)
        e.append(e_loc)

    return e, eta, theta_i, Mij

def compute_ed(e, thetai, trv_bnd_len_lst):
    """
    computes the deviatoric extension scalar state

    input:
    ------
        e               : trv list like extension scalar state for single node 
        thetai          : dilatation of corresponding node
        trv_bnd_len_lst : traversal list of bond length for a purturbed node  
    returns:
    --------
        ed              : trv list like deviatoric extension scalar state
    """

    ed = []
    for i in range(len(e)):
        #ed_loc = e[i] - thetai*trv_bnd_len_lst[i]
        ed_loc = e[i] - (thetai*trv_bnd_len_lst[i]/3)
        ed.append(ed_loc)

    return ed 


def compute_T(k, mu, mwi, theta_i, trv_bnd_len_lst, trv_infl_fld_lst,ed, M):
    """
    computes the force density vector state at a given node i

    input:
    ------
        k: TODO
        mu: TODO
        mwi: TODO
        theta_i: TODO
        trv_bnd_len_lst: TODO
        trv_infl_fld_lst: TODO
    returns:
    ---------
        T : np.array(1, dim) force density vector state of node i

    """
    T = []
    mwi_inv = 1/mwi
    for i in range(len(trv_bnd_len_lst)):
        t_loc = (3*k*theta_i*trv_bnd_len_lst[i] + 15*mu*ed[i])*trv_infl_fld_lst[i]*mwi_inv
        T_loc = t_loc*np.array(M[i])
        T.append(T_loc)

    return T

def peridym_tangent_stiffness_matrix(nbr_lst, nbr_bnd_vct_lst, nbr_bnd_len_lst, 
                                     nbr_infl_fld_lst, mw, mesh):
    """
    this function returns the peridynamic tangent stiffness matirx corresponding to the mesh 
    for a given horizon. The code is based on algorithm described in Ch5, Algorithm 4 in the 
    Handbook of Peridynamic Modeling by S.A Silling et.al 

    this method is currently based on central finite deifference method, 
    TODO : include more finite difference schemes(?)

    purturbation factor, bulk moudulus and shear modulus are currently hard coded
    TODO : compute tangent matrix for a general material

    input:
    ------
        nbr_lst: TODO
        nbr_bnd_vct_lst: TODO
        nbr_bnd_len_lst: TODO
        mesh: meshpy.MeshInfo
        purtub_fact: TODO
    returns:
    ---------
        K            : np.ndarray , the tangent stiffness matrix 

    """
    import timeit as tm
    start = tm.default_timer()

    print("computing the tangent stiffness matrix for the genereted mesh")
    k = 16.8*1e9; mu = 2.7*1e9; #bulk and shear modulus for aluminum from google
    #some precomputations
    dim = len(nbr_bnd_vct_lst[0][0])
    num_els = len(nbr_lst)
    size = (dim*num_els)

    #initializing empty data 
    K = np.zeros((size, size), dtype=float)

    elem_area = get_elem_areas(mesh)
    elem_cent = get_elem_centroid(mesh)

    trv_lst = copy.deepcopy(nbr_lst)
    trv_infl_fld_lst = copy.deepcopy(nbr_infl_fld_lst)
    trv_bnd_len_lst = copy.deepcopy(nbr_bnd_len_lst)
    trv_bnd_vct_lst = copy.deepcopy(nbr_bnd_vct_lst)
    purt_fact = 1e-6
    inv_purt_fact = 1e6
    #include self node at the beginning
    # (position 0) in neighborhood list of each node
    for i in range(num_els):
        #create and expand data structure from
        #neighbor list to that needed for traversal list
        trv_lst[i].insert(0,i)
        trv_infl_fld_lst[i].insert(0,1.) #infl fld for self node 1
        node_i_cent = elem_cent[i]
        trv_bnd_vct_lst[i].insert(0, [0.]*dim)
        trv_bnd_len_lst[i].insert(0,0.) #bond len for self node is zero

        for j, idx in enumerate(trv_lst[i]):
            node_j_cent = elem_cent[idx]
            elem_area_j = elem_area[idx]
            for d in range(dim):

                purt_node_i_pos = copy.deepcopy(node_i_cent) 
                purt_node_i_neg = copy.deepcopy(node_i_cent)

                purt_node_i_pos[d] += purt_fact
                purt_node_i_neg[d] -= purt_fact

                e_i_pos, eta_i_pos, theta_i_pos, Mi_pos = compute_purturb_exten(i, purt_node_i_pos, trv_lst[i], trv_bnd_vct_lst[i], 
                                                                        trv_bnd_len_lst[i], trv_infl_fld_lst[i], mw[i], elem_cent, elem_area) 

                e_i_neg, eta_i_neg, theta_i_neg, Mi_neg = compute_purturb_exten(i, purt_node_i_neg, trv_lst[i], trv_bnd_vct_lst[i], 
                                                                        trv_bnd_len_lst[i], trv_infl_fld_lst[i], mw[i], elem_cent, elem_area) 

                ed_pos = compute_ed(e_i_pos, theta_i_pos, trv_bnd_len_lst[i])
                ed_neg = compute_ed(e_i_neg, theta_i_neg, trv_bnd_len_lst[i])

                T_i_pos = compute_T(k, mu, mw[i], theta_i_pos, trv_bnd_len_lst[i], trv_infl_fld_lst[i], ed_pos, Mi_pos)
                T_i_neg = compute_T(k, mu, mw[i], theta_i_neg, trv_bnd_len_lst[i], trv_infl_fld_lst[i], ed_neg, Mi_neg)

                for k, kidx in enumerate(trv_lst[i]):
                    """
                    here instead of traversing through the neighbor
                    list as described in the algorithm, the traversal 
                    list is traversed, otherwise the diagonal entries
                    of K will be zero (TODO Check if correct)?
                    """
                    #area_fact = elem_area[i]*elem_area[kidx]
                    #f_eps_pos = T_i_pos[k]*area_fact
                    #f_eps_neg = T_i_neg[k]*area_fact
                    #f_diff = (f_eps_pos - f_eps_neg)*0.5*inv_purt_fact
                    f_diff = (T_i_pos[k] - T_i_neg[k])*0.5*inv_purt_fact*elem_area[i]*elem_area[kidx]

                    for dd in range(dim):
                        K[dim*i+dd][dim*kidx + d] += f_diff[dd]
    stop = tm.default_timer()

    print("total time taken to assemble tangent stiffness matrix: %4.3f seconds\n" %(stop-start))
    return K
