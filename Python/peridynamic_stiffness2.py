from mesh_tools import *
from helper import *



def compute_purturb_exten(purt_node, trv_lst, trv_bnd_vct_lst, 
                          trv_bnd_len_lst, trv_infl_fld_lst, elem_cent):

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
        Mij                : deformed unit vector state
    """

    dim = np.shape(trv_bnd_vct_lst)[1]
    Mij = np.zeros((len(trv_lst), dim), dtype=float)
    eta = elem_cent - purt_node
    eta_plus_psi = trv_bnd_vct_lst + eta
    mod_eta_psi = np.linalg.norm(eta_plus_psi, 2, axis=1)
    e = mod_eta_psi - trv_bnd_len_lst

    for i, idx in enumerate(trv_lst):
        Mij[i] = eta_plus_psi[i]/mod_eta_psi[i]

    return e, eta, Mij

def compute_theta(trv_lst, trv_bnd_len_lst, trv_infl_fld_lst, e, mwi, elem_area):
    """
    computes the dilatiation of purturbed node (only once for each node)

    :trv_lst: TODO
    :trv_bnd_len_lst: TODO
    :trv_infl_fld_lst: TODO
    :e: TODO
    :mwi: TODO
    :elem_area: TODO
    :returns: TODO

    """
    theta = np.sum(3*trv_infl_fld_lst*trv_bnd_len_lst*e*elem_area[trv_lst]/mwi)
    return theta

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

    ed = e - thetai*trv_bnd_len_lst/3
    return ed 

def compute_T_plane_stress(bulk, mu, mwi, theta_i, trv_lst, trv_bnd_len_lst, trv_infl_fld_lst, ed, M, elem_area):
    """
    computes the force density vector state at a given node i

    input:
    ------
        bulk: bulk modulus of elasticity
        mu: shear modulus
        mwi: weighted mass 
        theta_i: dilatation  
        trv_bnd_len_lst: 
        trv_infl_fld_lst: TODO
    returns:
    ---------
        T : np.array(1, dim) force density vector state of node i

    """
    ll = len(trv_lst)
    dim = np.shape(M[0])[0]
    T = np.zeros((ll, dim), dtype=float)
    gamma = 4*mu/(3*bulk + 4*mu)
    mwi_inv = 1/mwi
    t_loc = (gamma*bulk*theta_i*trv_bnd_len_lst + 8*mu*ed)*trv_infl_fld_lst*mwi_inv
    for i, _ in enumerate(trv_lst):
        area_fact = elem_area[i]*elem_area[0]
        T[0] += t_loc[i]*M[i]*area_fact #substituted by lines 108,109 (vectorized), 115
        T[i] -= t_loc[i]*M[i]*area_fact


    return T

def compute_T(bulk, mu, mwi, theta_i, trv_lst, trv_bnd_len_lst, trv_infl_fld_lst, ed, M, elem_area):
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
    ll = len(trv_lst)
    dim = np.shape(M[0])[0]
    T = np.zeros((ll, dim), dtype=float)
    mwi_inv = 3/mwi
    t_loc = (bulk*theta_i*trv_bnd_len_lst + 5*mu*ed)*trv_infl_fld_lst*mwi_inv
    for i, _ in enumerate(trv_lst):
        area_fact = elem_area[i]*elem_area[0]
        T[0] += t_loc[i]*M[i]*area_fact #substituted by lines 108,109 (vectorized), 115
        T[i] -= t_loc[i]*M[i]*area_fact


    return T

def peridym_tangent_stiffness_matrix2(nbr_lst, nbr_bnd_vct_lst, nbr_bnd_len_lst, 
                                     nbr_infl_fld_lst, mw, mesh, problem="plane_stress"):
    """
    this function returns the peridynamic tangent stiffness matirx corresponding to the mesh 
    for a given horizon. The code is based on algorithm described in Ch5, Algorithm 4 in the 
    Handbook of Peridynamic Modeling by S.A Silling et.al 

    this method is currently based on central finite deifference method, 
    TODO : include more finite difference schemes(?)

    purturbation factor, bulk moudulus and shear modulus are currently hard coded
    TODO : compute tangent matrix for a general material
    TODO : implement strategy of function pointer to resolve between 2D and 3D problems

    input:
    ------
        nbr_lst: TODO
        nbr_bnd_vct_lst: TODO
        nbr_bnd_len_lst: TODO
        mesh: meshpy.MeshInfo
        purtub_fact: TODO
        problem: string defining the problem type
                can be 'plane_stress', 'plane_strain', 'elastic_3d'
    returns:
    ---------
        K            : np.ndarray , the tangent stiffness matrix 

    """
    import timeit as tm
    start = tm.default_timer()

    #point to appropriate function 
    #for computing force density depending on problem type
    if problem is "plane_stress":
        compute_force_density = compute_T_plane_stress

    print("computing the tangent stiffness matrix for the genereted mesh using vectorized method")
    bulk = 68*1e9; mu = 27*1e9; #bulk and shear modulus for aluminum from google
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
        trv_lst[i] = np.insert(trv_lst[i], 0, i)
        trv_infl_fld_lst[i] = np.insert(trv_infl_fld_lst[i], 0, -1.) #infl fld for self node 1
        node_i_cent = elem_cent[i]
        trv_bnd_vct_lst[i] = np.insert(trv_bnd_vct_lst[i],0, np.zeros((1, dim), dtype=float), axis=0)
        trv_bnd_len_lst[i] = np.insert(trv_bnd_len_lst[i], 0, 0.) #bond len for self node is zero

        for j, idx in enumerate(trv_lst[i]):
            node_j_cent = elem_cent[idx]
            elem_area_j = elem_area[idx]
            for d in range(dim):

                purt_node_i_pos = copy.deepcopy(node_i_cent) 
                purt_node_i_neg = copy.deepcopy(node_i_cent)

                purt_node_i_pos[d] += purt_fact
                purt_node_i_neg[d] -= purt_fact

                e_i_pos, eta_i_pos, Mi_pos = compute_purturb_exten(purt_node_i_pos, trv_lst[i], trv_bnd_vct_lst[i], 
                                                                        trv_bnd_len_lst[i], trv_infl_fld_lst[i], elem_cent[trv_lst[i]]) 

                e_i_neg, eta_i_neg, Mi_neg = compute_purturb_exten(purt_node_i_neg, trv_lst[i], trv_bnd_vct_lst[i], 
                                                                        trv_bnd_len_lst[i], trv_infl_fld_lst[i], elem_cent[trv_lst[i]])

                #test : compute thet_i once only for each node i
                if j == 0:
                    theta_i_pos = compute_theta(trv_lst[i], trv_bnd_len_lst[i], trv_infl_fld_lst[i], 
                                                    e_i_pos, mw[i], elem_area)

                    theta_i_neg = compute_theta(trv_lst[i], trv_bnd_len_lst[i], trv_infl_fld_lst[i], 
                                                    e_i_neg, mw[i], elem_area)


                ed_pos = compute_ed(e_i_pos, theta_i_pos, trv_bnd_len_lst[i])
                ed_neg = compute_ed(e_i_neg, theta_i_neg, trv_bnd_len_lst[i])


                T_i_pos = compute_force_density(bulk, mu, mw[i], theta_i_pos, trv_lst[i], trv_bnd_len_lst[i], 
                                        trv_infl_fld_lst[i], ed_pos, Mi_pos, elem_area[trv_lst[i]])
                
                T_i_neg = compute_force_density(bulk, mu, mw[i], theta_i_neg, trv_lst[i], trv_bnd_len_lst[i], 
                                        trv_infl_fld_lst[i], ed_neg, Mi_neg, elem_area[trv_lst[i]])

                for k, kidx in enumerate(trv_lst[i]):
                    """
                    here instead of traversing through the neighbor
                    list as described in the algorithm, the traversal 
                    list is traversed, otherwise the diagonal entries
                    of K will be zero (TODO Check if correct)?
                    """
                    #f_diff = (T_i_pos[k] - T_i_neg[k])*0.5*inv_purt_fact*elem_area[i]*elem_area[kidx]
                    f_diff = (T_i_pos[k] - T_i_neg[k])*0.5*inv_purt_fact

                    for dd in range(dim):
                        K[dim*i+dd][dim*kidx + d] += f_diff[dd]
    stop = tm.default_timer()

    print("total time taken to assemble tangent stiffness matrix: %4.3f seconds\n" %(stop-start))
    return K
