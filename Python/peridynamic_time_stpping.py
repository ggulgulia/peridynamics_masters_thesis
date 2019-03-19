import numpy as np
import numpy.linalg as la
import math


def compute_t_critical(rho, bulkMod, horizon, nbr_lst, nbr_beta_lst, cell_cents, cell_vol):
    """
    del_t_critical for lps,
    the main formula is borrowed from that of 
    microelastic brittle material but the micromodulus
    is modified for lps

    section 5.8.2 from handbook of peridynamics

    input:
    ------
        rho     : material density
        bulkMod : self explanatory
        horizon : peridynamic horizon
        cell_vol: array of particle cell volume
    output:
    -------
        delT_cri : time step according to the aforementioned literature
    """

    num_els = len(cell_cents)
    two_times_rho = 2*rho
    delT_cr_array = np.zeros(num_els, dtype=float)
    for i in range(num_els):
        curr_cell = i
        curr_nbrs = nbr_lst[curr_cell]
        curr_nbr_beta = nbr_beta_lst[curr_cell]
        xi = cell_cents[curr_nbrs] - cell_cents[curr_cell]
        bond_len = la.norm(xi, 2, axis=0)
        temp = math.pi*horizon**4*sum(bond_len)
        Cp_eff = 18*bulkMod/temp
        denominator = sum(cell_vol[curr_nbrs]*curr_nbr_beta)*Cp_eff
        delT_cr_array[i] = math.sqrt(two_times_rho/denominator)

    detT_cr = np.min(delT_cr_array)

    return delT_cr


def intitialize_simulation():
    """
    intitializes the simulation 

    same for both implicit and explicit time integration 
    schemes

    sets
    initial time ,t = 0
    displacement field, u = 0
    acceleration, a= 0
    """
    time = 0.0
    u_disp = 0.0
    acc = 0.0
    return time, u_disp, acc

def update_time_step(t_n0, t_step):
    """
    updates the time steps

    :t_n0: TODO
    :t_step: TODO
    :returns: TODO

    """
    t_n1 = t_n0 + t_step
    t_n05 = 0.5(t_n0 + t_n1)
    
    return t_n05, t_n1


