import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from peridynamic_neighbor_data import*
from peridynamic_boundary_conditions import *
from fenics_mesh_tools import *

def omega_fun1(xi, horizon):

    if(len(xi) >2): #if we have an array of bond length
        bnd_len = la.norm(xi, 2, axis=1)
        return -np.exp(-(bnd_len**2/(0.5*horizon)**2))*(bnd_len<horizon).astype(float)
    else: # if there's a single bond
        bnd_len = la.norm(xi, 2, axis=0)
        return -np.exp(-(bnd_len**2/(0.5*horizon)**2))*float(la.norm(xi,2, axis=0)<horizon)



def computeTheta(u, horizon, nbr_lst, trv_lst,cell_vol, cell_cent, mw, gamma):
    """computes the dilatation vector, theta

    :u: TODO
    :horizon: TODO
    :nbr_lst: TODO
    :trv_lst: TODO
    :cell_vol: TODO
    :mwi: TODO
    :returns: TODO

    """
    num_els = len(cell_vol)
    dim = np.shape(cell_cent[0])[0]
    theta = np.zeros(num_els, dtype=float)

    for i in trv_lst:
        curr_nbr = nbr_lst[i]
        xi = cell_cent[curr_nbr] - cell_cent[i]
        bnd_len = la.norm(xi, 2, axis=1)
        eta = u[curr_nbr] - u[i]
        exten = la.norm((xi+eta), 2, axis=1) - bnd_len
        omega = omega_fun1(xi, horizon)

        cur_nbr_cell_vol = cell_vol[curr_nbr] #cn stands for curr_nbr
        theta[i] = sum(3*omega*bnd_len*exten*cur_nbr_cell_vol/mw[i])

    return theta


#vectorized version of Felix's code
def computeInternalForce(d,u,horizon, nbr_lst, cell_vol, cell_cent, mw, bulk, mu, gamma):
    """
    computes the internal force using pairwise force function

    the pairwise force function here is valid only for plane stress
    TODO: generalize to 3d stess, elasticity, viscoplasticity, 
          thermal models, etc
    """
    num_els=len(cell_vol)
    dim = np.shape(cell_cent[0])[0]
   
    #compute dilatation for each node
    theta=np.zeros(num_els, dtype=float)
    trv_lst = np.insert(nbr_lst[d],0,d)

    theta = computeTheta(u, horizon, nbr_lst, trv_lst, cell_vol, cell_cent, mw, gamma)
    
    # Compute pairwise contributions to the global force density vector
    f=np.zeros((num_els,dim), dtype=float)
    for i in trv_lst:
        curr_nbr = nbr_lst[i]
        xi = cell_cent[curr_nbr] - cell_cent[i]
        bnd_len = la.norm(xi, 2, axis=1)
        eta = u[curr_nbr] - u[i]
        xi_plus_eta = xi + eta
        mod_xi_plus_eta = la.norm(xi_plus_eta, 2, axis=1)
        exten = mod_xi_plus_eta - bnd_len
        omega = omega_fun1(xi, horizon)
        exten_d = exten - theta[i]*bnd_len*gamma/3
        t = (gamma*bulk*theta[i]*bnd_len + 8*mu*exten_d)*omega/mw[i]
        M = xi_plus_eta/mod_xi_plus_eta[:,None]
        
        f[i] += sum(M*cell_vol[curr_nbr][:,None]*t[:,None])*cell_vol[i]
        f[curr_nbr] -= M*t[:,None]*cell_vol[curr_nbr][:,None]*cell_vol[i]

    return f

    
def computeK(horizon, cell_vol, nbr_lst, mw, cell_cent, E, nu, mu, bulk, gamma):
    
    """
    computes the tangent stiffness matrix based on central difference method
    formulas for computation of constants needed in plane stress 
    (compare Handbook Section 6.3.1.1):

    nu=0.3
    E=200e9
    bulk = E/(3*(1 - (2*nu)))
    mu=E/(2*(1 + nu))
    gamma= 4*mu / (3*bulk + 4*mu)

    input:
    ------
        horizon : peridynamic horizon
        cell_vol: numpy array of cell volume
        nbr_lst : numpy array of peridynamic neighbor list
        mw      : weighted volume
        cell_cent: centroid of each element in peridynamic discretization
        E, nu, mu, bulk, gamma : material properites

    output:
    ------
        K : numpy nd array , the tangent stiffness matrix
    """
    print("Beginning to compute the tangent stiffness matrix")

    import timeit as tm
    start = tm.default_timer()
    #specify material constants
    #hard coded :(

    # Compose stiffness matrix
    num_els= len(cell_vol)
    dim = np.shape(cell_cent[0])[0]
    dof = num_els*dim
    K_naive = np.zeros((dof, dof), dtype=float)
    small_val=1e-10 #purtub factor
    inv_small_val = 1.0/small_val
    
    for i in range(num_els):
        for d in range(dim):
            u_e_p=np.zeros((num_els,dim), dtype=float)
            u_e_m=np.zeros((num_els,dim), dtype=float)
            u_e_p[i][d]= 1.0*small_val
            u_e_m[i][d]= -1.0*small_val
    
            f_p=computeInternalForce(i,u_e_p,horizon,nbr_lst,cell_vol,cell_cent,mw,bulk,mu,gamma)
            f_m=computeInternalForce(i,u_e_m,horizon,nbr_lst,cell_vol,cell_cent,mw,bulk,mu,gamma)
            
            for dd in range(dim):
                K_naive[dd::dim][:,dim*i+d] = (f_p[:,dd] - f_m[:,dd])*0.5*inv_small_val
            #K_naive[0::dim][:,dim*i+d] = (f_p[:,0] - f_m[:,0])/(2*small_val)
            #K_naive[1::dim][:,dim*i+d] = (f_p[:,1] - f_m[:,1])/(2*small_val)
    
    end = tm.default_timer()
    print("Time taken for the composition of tangent stiffness matrix seconds: %4.3f seconds\n" %(end-start))

    return K_naive
