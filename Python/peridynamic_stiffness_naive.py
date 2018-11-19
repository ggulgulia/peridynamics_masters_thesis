import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from peridynamic_neighbor_data import*
from peridynamic_boundary_conditions import *
from mesh_tools import *

def omega_fun1(xi, horizon):

    return -np.exp(-(la.norm(xi,2)**2/(0.5*horizon)**2))*float(la.norm(xi,2)<horizon)


def computeInternalForce(d,u,horizon, nbr_lst, elem_area, mw, bulk, mu, elem_cent):

    num_els=len(elem_area)
    dim = np.shape(elem_cent[0])[0]
   
    #compute dilatation for each node
    theta=np.zeros(num_els, dtype=float)
    gamma= 4*mu/(3*bulk + 4*mu)
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
    
            f_p=computeInternalForce(i,u_e_p,horizon,nbr_lst,elem_area,mw, bulk,mu,elem_cent)
            f_m=computeInternalForce(i,u_e_m,horizon,nbr_lst,elem_area,mw, bulk,mu,elem_cent)
            
            K_naive[0::dim][:,dim*i+d] = (f_p[:,0] - f_m[:,0])/(2*small_val)
            K_naive[1::dim][:,dim*i+d] = (f_p[:,1] - f_m[:,1])/(2*small_val)
    
    end = tm.default_timer()
    print("Time taken for the composition of tangent stiffness matrix seconds: %4.3f\n" %(end-start))

    return K_naive
    

# specify elasticity constants nu and E

#mesh and geometry info
#m = rectangle_mesh(subdiv=(4,2))
#for i in range(2):
#    m = uniform_refine_triangles(m,2);
#
#horizon = 1.5*max(get_edge_lengths(m));
#
#elem_area = get_elem_areas(m)
#elem_cent = get_elem_centroid(m)
#dim = np.shape(elem_cent[0])[0]
#num_els = np.shape(elem_cent)[0]
#nbr_lst, _, _, _, mw = peridym_get_neighbor_data(m, horizon)


# Compute weighted volume for each node
#mw=np.zeros(num_els, dtype=float)
#for i in range(num_els):
#    for j in nbr_lst[i]:
#        xi=elem_cent[j] - elem_cent[i]
#        omega = omega_fun1(xi, horizon)
#        mw[i] += omega*la.norm(xi,2)**2*elem_area[j]
#    

# Compute tangent stiffness matrix
#K = computeK(horizon, elem_area, nbr_lst, mw, bulk , mu, elem_cent)

#print("applying boundary conditions")
#
##boundary conditions given out as key value pair
#bc = {'left':'dirichlet', 'right':'force'}
#K_bc, fb = peridym_apply_bc(m, K, bc, -2.5e10);
#
#print("solving the linear system")
#sol = la.solve(K_bc,fb)
#
#
##to plot the displacement do some work
#u_disp = copy.deepcopy(sol)
#u_disp = np.reshape(u_disp, (int(len(sol)/dim), dim))
#
#a, b = get_peridym_mesh_bounds(m)
#node_ids = a['left']
#ll = b['left']
#
#for i, nk in enumerate(node_ids):
#    u_disp = np.insert(u_disp, nk, np.zeros(dim, dtype=float), axis=0)
#
#x, y = elem_cent.T
#disp_cent = elem_cent + u_disp
##plot_(m, new_fig=True, annotate=False)
#plt.scatter(x,y, color='r', marker='o')
#x,y = disp_cent.T
#plt.scatter(x,y, color='b', marker='o')
#plt.show(block=False)
