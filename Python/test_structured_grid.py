from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_solvers import direct_solver
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
import mshr


def test_structured_grid(vol_corr=True):
    m = RectangleMesh(Point(0,0), Point(2,1), 20, 10)
    struct_grd = True
    vol_corr = True

    """hard coded bcs"""
    bc_loc = [0,1]
    bc_type = {'dirichlet':0, 'forceY':1}
    bc_vals = {'dirichlet':0, 'forceY':-5e8}
    num_lyrs = 2 #num layers of cells for BC

    cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd)
    el = get_peridym_edge_length(cell_cent, struct_grd)
    extents = compute_modified_extents(cell_cent, el, struct_grd)
    dim = np.shape(cell_cent[0])[0]
    
    #boundary conditions managment
    node_ids_dir = get_boundary_layers(cell_cent, el, num_lyrs, bc_loc, struct_grd)
    node_ids_frc = get_boundary_layers(cell_cent, el, 2*num_lyrs, bc_loc, struct_grd)
    ghost_lyr_node_ids = node_ids_dir

    omega_fun = gaussian_infl_fun1
    E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)
    
    horizon = 2.001*np.abs(np.diff(cell_cent[0:2][:,0])[0])
    tree = QuadTree()
    tree.put(extents, horizon)
    
    nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
    mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst,nbr_beta_lst, horizon, omega_fun)
    
    K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun)
    
    K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, node_ids_dir, node_ids_frc, struct_grd)
    u_disp = direct_solver(K_bound, fb, dim, reshape=True) 
    cell_cent_orig, u_disp_orig, u_disp_ghost = recover_original_peridynamic_mesh(cell_cent, u_disp, bc_type, ghost_lyr_node_ids, struct_grd)
    disp_cent = get_displaced_soln(cell_cent_orig, u_disp_orig, horizon, dim, plot_=True, zoom=10)
    
    return K, disp_cent
