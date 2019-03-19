import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from peridynamic_neighbor_data import*
from fenics_mesh_tools import *
from scipy.integrate import quad

def integrand(r):
    """ 
    integrand for the 'exact' integral needed for 
    the stability factor in the correspondence material 
    model in peridynamics
    """
    return math.exp(-r**2)*r**4
    
def integrand2(r):
    """ 
    integrand for the 'exact' integral needed for 
    the stability factor in the correspondence material 
    model in peridynamics
    """
    return math.exp(-r**2)
    
def computeGreenStrinTensor(def_grad_tensor):
    """
    computes the green strain tensor using equation

    E = 0.5(F^TF -I), F= def_grad_tensor, E = green strainTensr
    I = identity matrix

    :def_grad_tensor: TODO
    :returns: E, green_strain_tensor

    """
    F = def_grad_tensor 
    dim = np.shape(F)[0]
    I = np.identity(dim)

    F_t = np.transpose(F)
    E_green = 0.5*(np.matmul(F_t, F) - I)

    return  E_green #green_strain_tensor


def computeSecondPiolaStressTensor(E_green, lamda, mu):
    """

    computes the second piola kirchoff stress tensor using
    S2 = lamda*tr(E)*I  + mu*E

    input:
    -----
    E_green : 2nd Order Green Strain Tensor
    lamda   : lame's first parameter
    mu      : lame's second parameter

    output:
    -------
    S2_piola : 2nd order tensor, 2nd piola kirchoff setress tensor 
    """

    dim = np.shape(E_green)[0]
    trace = sum(E_green[d][d] for d in range(dim))
    S2_piola = lamda*trace*np.identity(dim) + 2*mu*E_green
    return S2_piola 


def computeInternalForce_correct_zero_energy_mode(curr_cell ,u, horizon, nbr_lst, nbr_beta_lst, cell_vol, cell_cent, mw, bulkMod, shearMod, gamma, E, lamda, gamma_corr, omega_fun):
    """
    computes the force state based on the correspondence model 
    for Peridynamics

    this routine computes the stabilized corresponedence material model 
    based on the paper : Stability of peridynamic peridynamic correspondence
    material models and their particle discretizations by S.A Silling
    Elsvier Journal
    Specifically internalForceComputation employes equation 56 in this paper

    Note: 'literature' referred to in the code points to this journal

    input:
    ------
        curr_cell: TODO
        u: TODO
        horizon: TODO
        nbr_beta_lst: TODO
        nbr_beta_lst: TODO
        cell_vol: TODO
        cell_cent: TODO
        bulkMod: TODO
        shearMod: TODO
        omega_fun: TODO

    output:
    -------
        T : internal force state (different from peridynamic force state density)
    
    """
    
    dim = len(cell_cent[0])

    num_els=len(cell_vol)
    mu = shearMod
    bondDamage = 0.0

    #compute dilatation for each node
    trv_lst = np.insert(nbr_lst[curr_cell],0,curr_cell)
    f_global=np.zeros((len(cell_cent),dim), dtype=float) #placeholder for force state

    # Compute pairwise contributions to the global force density vector
    for idx in trv_lst:
        shape_tensor = np.zeros((dim, dim), dtype=float)
        def_grad_tensor = np.zeros((dim, dim), dtype=float)
        curr_nbrs = nbr_lst[idx]
        curr_beta_lst = nbr_beta_lst[idx]
        curr_cell_vol = cell_vol[curr_nbrs]*curr_beta_lst
        xi = cell_cent[curr_nbrs] - cell_cent[idx]
        y_xi_curr = cell_cent[idx] + u[idx]
        y_xi_nbrs = cell_cent[curr_nbrs] + u[curr_nbrs]
        y_xi = y_xi_nbrs - y_xi_curr
        omega = omega_fun(xi, horizon)
        omega_damaged = (1.0 - bondDamage)*omega
        
        shape_tensor = np.sum(np.einsum('ij,ik->ijk', xi, xi)*omega_damaged[:,None,None],axis=0)
        def_grad_tensor = np.sum(np.einsum('ij,ik->ijk', y_xi, xi)*omega_damaged[:,None,None], axis=0)

        K_shp = shape_tensor*cell_vol[idx]
        K_inv = la.inv(K_shp)
        F = np.matmul(def_grad_tensor*cell_vol[idx], K_inv)
        E_green = computeGreenStrinTensor(F)
        S2_piola = computeSecondPiolaStressTensor(E_green, lamda, mu)
        S1_piola = np.matmul(F,S2_piola)
        temp2 = np.matmul(S1_piola, K_inv)

        curr_cell_vol *= cell_vol[idx]

        #vectorized computation of z<xi> as in equation 38 of literature
        temp2_vect = y_xi - np.einsum('ij,kj->ki',F,xi) 


        #vectorized computation of T_hat<xi> as in equation 56. 
        # Note the dV multiplication is done here to fit in the vectorized computation appropriately
        temp3_vect = (np.einsum('ij,kj->ki', temp2, xi) + gamma_corr*temp2_vect)*omega_damaged[:,None]*curr_cell_vol[:,None]
        f_global[idx] += sum(temp3_vect)
        f_global[curr_nbrs] -= temp3_vect

    return f_global


## Internal force routinee based on correspondance elastic material
def computeInternalForce_naive(curr_cell ,u, horizon, nbr_lst, nbr_beta_lst, cell_vol, cell_cent, mw, bulkMod, shearMod, gamma, E, lamda, gamma_corr, omega_fun):
    """
    computes the force state based on the correspondence model 
    for Peridynamics

    this version of internal force doesn't account for the correction
    of zero energy modes and computes the Material response of the 
    correspondence model as in equation 17 of the literature
    input:
    ------
        curr_cell: TODO
        u: TODO
        horizon: TODO
        nbr_beta_lst: TODO
        nbr_beta_lst: TODO
        cell_vol: TODO
        cell_cent: TODO
        bulkMod: TODO
        shearMod: TODO
        omega_fun: TODO

    output:
    -------
        T : internal force state (different from peridynamic force state density)
    
    """
    
    dim = len(cell_cent[0])

    num_els=len(cell_vol)
    mu = shearMod
    bondDamage = 0.0

    #compute dilatation for each node
    trv_lst = np.insert(nbr_lst[curr_cell],0,curr_cell)
    f_global=np.zeros((len(cell_cent),dim), dtype=float) #placeholder for force state
    
    curr_cell_vol_factor = 1.0 #curr_cell centroid is the centre of the horizon
    # Compute pairwise contributions to the global force density vector
    for idx in trv_lst:
        shape_tensor = np.zeros((dim, dim), dtype=float)
        def_grad_tensor = np.zeros((dim, dim), dtype=float)
        curr_nbrs = nbr_lst[idx]
        curr_beta_lst = nbr_beta_lst[idx]
        curr_cell_vol = cell_vol[curr_nbrs]*curr_beta_lst
        xi = cell_cent[curr_nbrs] - cell_cent[idx]
        y_xi_curr = cell_cent[idx] + u[idx]
        y_xi_nbrs = cell_cent[curr_nbrs] + u[curr_nbrs]
        y_xi = y_xi_nbrs - y_xi_curr
        omega = omega_fun(xi, horizon)
        omega_damaged = (1.0 - bondDamage)*omega
        

        shape_tensor = np.sum(np.einsum('ij,ik->ijk', xi, xi)*omega_damaged[:,None,None],axis=0)
        def_grad_tensor = np.sum(np.einsum('ij,ik->ijk', y_xi, xi)*omega_damaged[:,None,None], axis=0)

        K_shp = shape_tensor*cell_vol[idx]
        K_inv = la.inv(K_shp)
        F = np.matmul(def_grad_tensor*cell_vol[idx], K_inv)
        E_green = computeGreenStrinTensor(F)
        S2_piola = computeSecondPiolaStressTensor(E_green, lamda, mu)
        S1_piola = np.matmul(F,S2_piola)
        temp2 = np.matmul(S1_piola, K_inv)
        
        curr_cell_vol *=cell_vol[idx]
        temp3_vect = np.einsum('ij,kj->ki', temp2, xi)*omega_damaged[:,None]*curr_cell_vol[:,None]
        f_global[idx] += sum(temp3_vect)
        f_global[curr_nbrs] -= temp3_vect

    return f_global

def computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, mw, cell_cent, E, nu, shearMod, bulkMod, gamma, omega_fun, u_disp):
    
    """
    computes the tangent stiffness matrix based on central difference method
    formulas for computation of constants needed in plane stress 
    (compare Handbook Section 6.3.1.1):

    nu=0.3  #for steel
    E=200e9 #for steel
    bulkMod = E/(3*(1 - (2*nu)))
    shearMod=E/(2*(1 + nu))
    gamma= 4*shearMod / (3*bulkMod + 4*shearMod)

    input:
    ------
        horizon : peridynamic horizon
        cell_vol: numpy array of cell volume
        nbr_lst : numpy array of peridynamic neighbor list
        nbr_beta_lst: list of volume fraction corresponding to the
                    nbr_lst
        mw      : weighted volume
        cell_cent: centroid of each element in peridynamic discretization
        E, nu, shearMod, bulkMod, gamma : material properites
        omega_fun : pointer to peridynamic influence function
        u_disp  : displacement field 

    output:
    ------
        K : numpy nd array , the tangent stiffness matrix
    """
    print("Beginning to compute the tangent stiffness matrix using Correspondence material model")

    import timeit as tm
    start = tm.default_timer()
    lamda = 3*bulkMod*(3*bulkMod- E)/(9*bulkMod - E)

    """
    alpha = 15*mu/mw if dim ==3 else 8*mu/mw
    alpha depends on individual mw of the cell in consideration
    hence we do 'inplace computation' of the final gamma_corr
    """
     
    correct_zero_energy_modes = True

    if correct_zero_energy_modes:
        computeInternalForce = computeInternalForce_correct_zero_energy_mode
        G = 9.0 #positive constant of order 1, from paper mailed by Dr Silling
        gamma_corr = 18*G*bulkMod/(math.pi*quad(integrand2, 0, horizon)[0]) #alpha is still missing
    else:
        computeInternalForce = computeInternalForce_naive
        gamma_corr=None

    # Compose stiffness matrix
    num_els = len(cell_vol)
    dim = len(cell_cent[0])
    dof = num_els*dim
    K_naive = np.zeros((dof, dof), dtype=float)
    small_val=1e-6 #purtub factor
    inv_small_val = 1.0/small_val
    
    for i in range(num_els):
        for d in range(dim):
            u_e_p = cpy.deepcopy(u_disp)
            u_e_m = cpy.deepcopy(u_disp)
            u_e_p[i][d]= 1.0*small_val
            u_e_m[i][d]= -1.0*small_val
            
            f_p=computeInternalForce(i,u_e_p,horizon,nbr_lst,nbr_beta_lst,cell_vol,cell_cent,mw,bulkMod,shearMod,gamma, E, lamda, gamma_corr, omega_fun)
            f_m=computeInternalForce(i,u_e_m,horizon,nbr_lst,nbr_beta_lst,cell_vol,cell_cent,mw,bulkMod,shearMod,gamma, E, lamda, gamma_corr, omega_fun)
            
            for dd in range(dim):
                K_naive[dd::dim][:,dim*i+d] += (f_p[:,dd] - f_m[:,dd])*0.5*inv_small_val
    
    end = tm.default_timer()
    print("Time taken for the composition of tangent stiffness matrix seconds: %4.3f seconds\n" %(end-start))

    return K_naive

