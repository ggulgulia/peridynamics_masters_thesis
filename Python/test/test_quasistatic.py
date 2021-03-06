from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_solvers import direct_solver
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
import mshr

def run_quasistatic_test():
    #m = box_mesh(Point(0,0,0), Point(2,1,1), 10,5,5)
    #m = box_mesh_with_hole(numpts=20)
    #domain = mshr.Rectangle(Point(0,0), Point(3,1))
    #m = mshr.generate_mesh(domain, 30)
    m = RectangleMesh(Point(0,0), Point(2,1), 20, 10)
    #m = rectangle_mesh(numptsX=20, numptsY=10)
    #m = rectangle_mesh_with_hole(npts=25)
    
    struct_grd = True
    vol_corr = True
    
    f0 = -5e8
    delta_t = 0.001
    ft = -1e10
    
    bc_type = {'dirichlet':0,'forceY':1}
    bc_vals = {'dirichlet':0,'forceY':0}
    bc_loc = [0,1]
    num_lyrs = 2 #num of additional layers on boundary
    cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd) 
    dim = np.shape(cell_cent[0])[0]
    el = get_peridym_edge_length(cell_cent, struct_grd)
   
    # dirichlet bc on ghost layers
    node_ids_dir = get_boundary_layers(cell_cent, el, num_lyrs, bc_loc, struct_grd)
    node_ids_frc = get_boundary_layers(cell_cent, el, 2*num_lyrs, bc_loc, struct_grd)
    ghost_lyr_node_ids = node_ids_dir

    omega_fun = gaussian_infl_fun2
    E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)

    horizon = 2.001*np.abs(np.diff(cell_cent[0:2][:,0])[0])

    num_steps = 5
    u_disp = np.zeros((len(cell_cent), dim), dtype=float)

    for i in range(num_steps):
        force = f0 + ft*i*delta_t
        bc_vals['forceY'] = force
        extents = compute_modified_extents(cell_cent, el, struct_grd)
        
        tree = QuadTree()
        tree.put(extents, horizon)
        
        nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
        mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst, nbr_beta_lst, horizon, omega_fun)
        
        K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun, u_disp)
        
        K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, node_ids_dir, node_ids_frc, struct_grd)
        
        u_disp = direct_solver(K_bound, fb, dim, reshape=True)
        cell_cent_orig, u_disp_orig, u_disp_ghost = recover_original_peridynamic_mesh(cell_cent, u_disp, bc_type, ghost_lyr_node_ids, struct_grd)
        disp_cent = get_displaced_soln(cell_cent_orig, u_disp_orig, horizon, dim, plot_=True, zoom=10)

        u_disp = u_disp_ghost
        cell_cent += u_disp

