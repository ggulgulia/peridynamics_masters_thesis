from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_boundary_conditions import *
from peridynamic_materials import *
import mshr
import timeit as tm
from peridynamic_infl_fun import *

def solve_peridynamic_bar(horizon, m=mesh, nbr_lst=None, nbr_beta_lst=None, npts=15, material='steel', omega_fun=None, plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
    """
    solves the peridynamic bar with a specified load

    """

    print('horizon value: %4.3f\n'%horizon)

    
    bc_type = {0:'dirichlet', 1:'force'}
    bc_vals = {'dirichlet': 0, 'force': force}
    bc_loc = [0,1]
    num_lyrs = 2
    cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd) 
    
    el = get_peridym_edge_length(cell_cent, struct_grd) 
    extents = compute_modified_extents(cell_cent, el, struct_grd)
    purtub_fact = 1e-6
    dim = m.topology().dim() 
    
    if material is 'steel':
        E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)

    if (nbr_lst==None or nbr_beta_lst==None):
        tree = QuadTree()
        tree.put(extents, horizon)
        nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
        
    mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst, nbr_beta_lst, horizon, omega_fun) 
    K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun)
    
    K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, num_lyrs, struct_grd)
    
    print("solving the stystem")
    start = tm.default_timer()
    sol = np.linalg.solve(K_bound, fb)
    end = tm.default_timer()
    print("Time taken for solving the system of equation: %4.3f secs" %(end-start))
    u_disp = copy.deepcopy(sol)
    u_disp = np.reshape(u_disp, (int(len(sol)/dim), dim))
   
                       #recover_original_peridynamic_mesh
    cell_cent, u_disp = recover_original_peridynamic_mesh(cell_cent, u_disp, el, bc_type, num_lyrs, struct_grd)
    disp_cent = u_disp + cell_cent

    if plot_ is True: 
        x, y = cell_cent.T
        plt.figure()
        plt.scatter(x,y, s=300, color='r', marker='o', alpha=0.1, label='original configuration')
        x,y = (cell_cent + 20*u_disp).T
        plt.scatter(x,y, s=300, color='b', marker='o', alpha=0.6, label='horizon='+str(horizon))
        #plt.ylim(-0.5, 1.5)
        plt.legend()
        plt.title("influence function:"+str(omega_fun))
        plt.show(block=False)
    
    return K, K_bound, disp_cent, u_disp
