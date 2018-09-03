from peridynamic_fields import *
from peridynamic_bond_manager import *
from peridynamic_fields import *
import peridynamic_compute_neighbors as nbrs
from helper import *



def compute_purturb_exten(node_num, purt_node, trv_lst, trv_bnd_vct_lst, 
                          trv_bnd_len_lst, trv_infl_fld_lst, mwi, elem_cent, elem_area):

    """
    returns the force state of the purturbed node

    :node_j_cent: TODO
    :purt_node_j_cent: TODO
    :returns: TODO

    """
    e = []
    eta = []
    Mij = []
    theta_i = 0. #dilatation for current node
    mwi_inv = 1/mwi
    for i, idx in enumerate(trv_lst):
        eta_loc = vect_diff(purt_node, elem_cent[idx]) #deformation relative to only one bond 
        #relative to only one node which happens to be the current purturbed node
        eta_plus_psi = vect_sum(trv_bnd_vct_lst[i], eta_loc)
        e_loc = mod(eta_plus_psi) - trv_bnd_len_lst[i]
        theta_i += 3*mwi_inv*trv_infl_fld_lst[i]*trv_bnd_len_lst[i]*e_loc*elem_area[trv_lst[i]]

        Mij.append(eta_plus_psi/mod(eta_plus_psi))
        eta.append(eta_loc)
        e.append(e_loc)

    return e, eta, theta_i, Mij

def compute_ed(e, thetai, trv_bnd_len_lst):
    """
    computes the deviatoric extension 

    :e: TODO
    :thetai: TODO
    :returns: TODO

    """
    ed = []
    for i in range(len(e)):
        ed_loc = e[i] - thetai*trv_bnd_len_lst[i]
        ed.append(ed_loc)

    return ed 


#T_i_pos = compute_T(k, mu, mw[i], theta_i_pos, trv_bnd_len_lst, trv_infl_fld_lst)
def compute_T(k, mu, mwi, theta_i, trv_bnd_len_lst, trv_infl_fld_lst,ed, M):
    """
    computes the force vector state at a given node i

    :k: TODO
    :mu: TODO
    :mwi: TODO
    :theta_i: TODO
    :trv_bnd_len_lst: TODO
    :trv_infl_fld_lst: TODO
    :returns: TODO

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
    returns the peridynamic tangent stiffness matirx corresponding to the mesh, neighborhood list and 
    dimension of the problem

    :nbr_lst: TODO
    :nbr_bnd_vct_lst: TODO
    :nbr_bnd_len_lst: TODO
    :mesh: meshpy.MeshInfo
    :purtub_fact: TODO
    :returns: TODO

    """
    k = 68*1e5; mu = 27*1e5; #bulk and shear modulus for aluminum from google
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

        Ti_pos = []
        Ti_neg = []
        for j, idx in enumerate(trv_lst[i]):
            node_j_cent = elem_cent[idx]
            elem_area_j = elem_area[idx]
            for d in range(dim):

                purt_node_i_pos = node_i_cent.copy() 
                purt_node_i_neg = node_i_cent.copy()

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
                    area_fact = elem_area[i]*elem_area[kidx]
                    f_eps_pos = T_i_pos[k]*area_fact
                    f_eps_neg = T_i_neg[k]*area_fact
                    f_diff = (f_eps_pos - f_eps_neg)*0.5/purt_fact

                    for dd in range(dim):
                        K[dim*i+dd][dim*kidx + d] += f_diff[dd]
                    
    return K
