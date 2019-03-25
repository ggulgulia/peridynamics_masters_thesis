import numpy.linalg as la
import math 
import numpy as np

def compute_critical_stretch(G0, bulkMod, horizon):
    """
    computes the critical stretch of a material
    according to the relation 5.3 in handbook of
    peridynamics
    input:
    ------
        G0 : fracture energy per unit area
        bulkMod: bulk mouduls of the material
        horizon: peridynamic horizon
    output:
    -------
        s0 : criticl stretch
    """

    s0 = math.sqrt(5*G0/(9*bulkMod*horizon))
    return s0

def compute_bond_damage(s0, cell_cent, nbr_lst, u_disp, dmg_flag=False):
    """
    this method computes the bond damage according to the 
    algorithm described in chapter 5 of handbook of peridynamics
    
    input:
    ------
        s0       : critical stretch factor
        cell_cent: cell centroids 
        nbr_lst  : neighbor list 
        u_disp   : displacement field 
        dmg_flag : boolean whether to compute damage or not
                   returns an empty bnd_dmg_lst incase False
    output:
    -------
        bnd_dmg: list analogus to nbr_lst

    """

    num_els = len(cell_cent)
    bnd_dmg_lst = []
    if dmg_flag:
        for i in range(num_els):
            curr_nbrs = nbr_lst[i] 
            curr_dmg_lst = np.zeros(len(curr_nbrs), dtype=float)
            xi = cell_cent[curr_nbrs] - cell_cent[i]
            eta = u_disp[curr_nbrs] - u_disp[i]
            xi_plus_eta = xi + eta
            xi_len = la.norm(xi, 2, axis=1)
            xi_plus_eta_len = la.norm(xi_plus_eta, 2, axis=1)
            exten = (xi_plus_eta_len - xi_len)
            s = np.divide(exten, xi_len)
            curr_dmg_lst[np.where(s>=s0)[0]] = 1.0
            bnd_dmg_lst.append(curr_dmg_lst)
    else:
        bnd_dmg_lst = cpy.deepcopy(nbr_lst)
        for i in range(num_els):
            bnd_dmg_lst[i][:] = 0.0

    return bnd_dmg_lst

