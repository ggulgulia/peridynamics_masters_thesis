import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from peridynamic_neighbor_data import*
from peridynamic_boundary_conditions import *
from mesh_tools import *

def omega_fun1(xi, horizon):

    if(len(xi) >2): #if we have an array of bond length
        bnd_len = la.norm(xi, 2, axis=1)
        return -np.exp(-(bnd_len**2/(0.5*horizon)**2))*(bnd_len<horizon).astype(float)
    else: # if there's a single bond
        bnd_len = la.norm(xi, 2, axis=0)
        return -np.exp(-(bnd_len**2/(0.5*horizon)**2))*float(la.norm(xi,2, axis=0)<horizon)



def computeTheta(u, horizon, nbr_lst, trv_lst,elem_area, elem_cent, mw, gamma):
    """computes the dilatation vector, theta

    :u: TODO
    :horizon: TODO
    :nbr_lst: TODO
    :trv_lst: TODO
    :elem_area: TODO
    :mwi: TODO
    :returns: TODO

    """
    num_els = len(elem_area)
    dim = np.shape(elem_cent[0])[0]
    theta = np.zeros(num_els, dtype=float)

    for i in trv_lst:
        curr_nbr = nbr_lst[i]
        xi = elem_cent[curr_nbr] - elem_cent[i]
        bnd_len = la.norm(xi, 2, axis=1)
        eta = u[curr_nbr] - u[i]
        exten = la.norm((xi+eta), 2, axis=1) - bnd_len
        omega = omega_fun1(xi, horizon)

        cnelem_area = elem_area[curr_nbr] #cn stands for curr_nbr
        theta[i] = sum(3*omega*bnd_len*exten*cnelem_area/mw[i])

    return theta


#vectorized version of Felix's code
def computeInternalForceV(d,u,horizon, nbr_lst, elem_area, elem_cent, mw, bulk, mu, gamma):

    num_els=len(elem_area)
    dim = np.shape(elem_cent[0])[0]
   
    #compute dilatation for each node
    theta=np.zeros(num_els, dtype=float)
    trv_lst = np.insert(nbr_lst[d],0,d)

    theta = computeTheta(u, horizon, nbr_lst, trv_lst, elem_area, elem_cent, mw, gamma)
    
    # Compute pairwise contributions to the global force density vector
    f=np.zeros((num_els,dim), dtype=float)
    for i in trv_lst:
        curr_nbr = nbr_lst[i]
        xi = elem_cent[curr_nbr] - elem_cent[i]
        bnd_len = la.norm(xi, 2, axis=1)
        eta = u[curr_nbr] - u[i]
        xi_plus_eta = xi + eta
        mod_xi_plus_eta = la.norm(xi_plus_eta, 2, axis=1)
        exten = mod_xi_plus_eta - bnd_len
        omega = omega_fun1(xi, horizon)
        exten_d = exten - theta[i]*bnd_len*gamma/3
        t = (gamma*bulk*theta[i]*bnd_len + 8*mu*exten_d)*omega/mw[i]
        M = xi_plus_eta/mod_xi_plus_eta[:,None]
        
        f[i] += sum(M*elem_area[curr_nbr][:,None]*t[:,None])
        f[curr_nbr] -= M*elem_area[i]*t[:,None]

    return f

def computeInternalForce(d,u,horizon, nbr_lst, elem_area, elem_cent, mw, bulk, mu, gamma):

    num_els=len(elem_area)
    dim = np.shape(elem_cent[0])[0]
   
    #compute dilatation for each node
    theta=np.zeros(num_els, dtype=float)
    trv_lst = np.insert(nbr_lst[d],0,d)
    
    for i in trv_lst:
        for j in nbr_lst[i]:
            xi=elem_cent[j] - elem_cent[i]
            eta = u[j] - u[i]
            exten = la.norm((xi + eta),2) - la.norm(xi,2)
            omega = omega_fun1(xi, horizon)
            theta[i] += 3*omega*la.norm(xi,2)*exten*elem_area[j]/mw[i]
    
    # Compute pairwise contributions to the global force density vector
    f=np.zeros((num_els,dim), dtype=float)
    for i in trv_lst:
        for j in nbr_lst[i]:
            xi = elem_cent[j] - elem_cent[i]
            eta = u[j] - u[i]

            omega = omega_fun1(xi,horizon)
            exten = la.norm((xi + eta),2) - la.norm(xi,2)
            exten_d = exten - theta[i]*la.norm(xi,2)*gamma/3
            t = (gamma*bulk*theta[i]*la.norm(xi,2) + 8*mu*exten_d)*omega / mw[i]
            
            M = (xi + eta)/la.norm((xi + eta),2)
            f[i] += t*M*elem_area[j]
            f[j] -= t*M*elem_area[i]

    return f
    
    
def computeK(horizon, elem_area, nbr_lst, mw, elem_cent):

    print("Beginning to compute the tangent stiffness matrix")

    import timeit as tm
    start = tm.default_timer()
    #specify material constants
    #hard coded :(
    nu=0.3
    E=200e9
    bulk = E/(3*(1 - (2*nu)))
    mu=E/(2*(1 + nu))
    # constant needed in plane stress (compare Handbook Section 6.3.1.1)
    gamma= 4*mu / (3*bulk + 4*mu)

    # Compose stiffness matrix
    num_els= len(elem_area)
    dim = np.shape(elem_cent[0])[0]
    size = num_els*dim
    K_naive = np.zeros((size, size), dtype=float)
    small_val=1e-10 #purtub factor
    
    for i in range(num_els):
        for d in range(dim):
            u_e_p=np.zeros((num_els,dim), dtype=float)
            u_e_m=np.zeros((num_els,dim), dtype=float)
            u_e_p[i][d]= 1.0*small_val
            u_e_m[i][d]= -1.0*small_val
    
            f_p=computeInternalForce(i,u_e_p,horizon,nbr_lst,elem_area,elem_cent,mw,bulk,mu,gamma)
            f_m=computeInternalForce(i,u_e_m,horizon,nbr_lst,elem_area,elem_cent,mw,bulk,mu,gamma)
            
            K_naive[0::dim][:,dim*i+d] = (f_p[:,0] - f_m[:,0])/(2*small_val)
            K_naive[1::dim][:,dim*i+d] = (f_p[:,1] - f_m[:,1])/(2*small_val)
    
    end = tm.default_timer()
    print("Time taken for the composition of tangent stiffness matrix seconds: %4.3f\n" %(end-start))

    return K_naive
    
def computeKV(horizon, elem_area, nbr_lst, mw, elem_cent):

    print("Beginning to compute the tangent stiffness matrix")

    import timeit as tm
    start = tm.default_timer()
    #specify material constants
    #hard coded :(
    nu=0.3
    E=200e9
    bulk = E/(3*(1 - (2*nu)))
    mu=E/(2*(1 + nu))
    # constant needed in plane stress (compare Handbook Section 6.3.1.1)
    gamma= 4*mu / (3*bulk + 4*mu)

    # Compose stiffness matrix
    num_els= len(elem_area)
    dim = np.shape(elem_cent[0])[0]
    size = num_els*dim
    K_naive = np.zeros((size, size), dtype=float)
    small_val=1e-10 #purtub factor
    
    for i in range(num_els):
        for d in range(dim):
            u_e_p=np.zeros((num_els,dim), dtype=float)
            u_e_m=np.zeros((num_els,dim), dtype=float)
            u_e_p[i][d]= 1.0*small_val
            u_e_m[i][d]= -1.0*small_val
    
            f_p=computeInternalForceV(i,u_e_p,horizon,nbr_lst,elem_area,elem_cent,mw,bulk,mu,gamma)
            f_m=computeInternalForceV(i,u_e_m,horizon,nbr_lst,elem_area,elem_cent,mw,bulk,mu,gamma)
            
            K_naive[0::dim][:,dim*i+d] = (f_p[:,0] - f_m[:,0])/(2*small_val)
            K_naive[1::dim][:,dim*i+d] = (f_p[:,1] - f_m[:,1])/(2*small_val)
    
    end = tm.default_timer()
    print("Time taken for the composition of tangent stiffness matrix seconds: %4.3f\n" %(end-start))

    return K_naive
