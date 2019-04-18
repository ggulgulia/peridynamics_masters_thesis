from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import computeK
from peridynamic_correspondence import computeKCorrespondance
from peridynamic_boundary_conditions import *
from peridynamic_solvers import direct_solver
from peridynamic_materials import *
import timeit as tm

from peridynamic_damage import *


def solve_peridynamic_patch_test(horizon, m=mesh, nbr_lst=None, nbr_beta_lst=None, material='steel', omega_fun=None, plot_=True, force=25e9, vol_corr=True, struct_grd=True, response='LPS'):
    """
    solves the peridynamic bar with a specified load

    """
    print('********** SOLVING PERIDYNAMIC PATCH TEST LOAD PROBLEM ***********')
    if response == 'LPS':
        computeStiffness = computeK
    if response == 'correspondance':
        computeStiffness = computeKCorrespondance
    

    dmg_flag=False
    print('horizon value: %4.3f\n'%horizon)

    
    bc_type = {'dirichletX':0, 'forceX':1}
    bc_vals = {'dirichletX': 0, 'forceX': force}
    bc_loc = [0,1]
    num_lyrs = 3 #three times discretization width
    cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd) 
    el = get_peridym_edge_length(cell_cent, struct_grd)
    extents = compute_modified_extents(cell_cent, el, struct_grd)
    dim = m.topology().dim() 
     
    
    #boudary conditions management
    node_ids_dir = get_boundary_layers(cell_cent, el, num_lyrs, bc_loc, struct_grd)
    node_ids_frc = get_boundary_layers(cell_cent, el, 2*num_lyrs, bc_loc, struct_grd)
    ghost_lyr_node_ids = node_ids_dir

    if material is 'steel':
        E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)

    if (nbr_lst==None or nbr_beta_lst==None):
        tree = QuadTree()
        tree.put(extents, horizon)
        nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
    
    ## Initialize simulation
    G0 = 1000000/2
    s0 = compute_critical_stretch(G0, bulk, horizon)
    u_disp = np.zeros((len(cell_cent), dim), dtype=float)
    bnd_dmg_lst = compute_bond_damage(s0, cell_cent, nbr_lst, u_disp, dmg_flag)

    mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst, nbr_beta_lst, horizon, omega_fun) 
    K = computeStiffness(horizon, cell_vol, nbr_lst, nbr_beta_lst, bnd_dmg_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun, u_disp)   
    #apply bc
    K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, node_ids_dir, node_ids_frc, struct_grd)

    #fix the lower corners of dirichlet node ids manually :(
    fixed_nodes = node_ids_dir[bc_type['dirichletX']][0][0:num_lyrs]
    for nk in fixed_nodes:
        K_bound[nk*dim+0] = 0.0; K_bound[nk*dim+1] = 0.0; 
        K_bound[nk*dim+0][nk*dim+0] = 1.0; K_bound[nk*dim+1][nk*dim+1] = 1.0
        fb[nk*dim+0] = 0.0; fb[nk*dim+1] = 0.0
    #solve
    u_disp = direct_solver(K_bound, fb, dim, reshape=True)
   
    cell_cent_orig, u_disp_orig, _ = recover_original_peridynamic_mesh(cell_cent, u_disp, bc_type, ghost_lyr_node_ids, struct_grd)
    #disp_cent = u_disp + cell_cent
    
    disp_cent = get_displaced_soln(cell_cent_orig, u_disp_orig, horizon, dim, plot_=plot_, zoom=20)
    
    return K, K_bound, disp_cent, u_disp_orig
