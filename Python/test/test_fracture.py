from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_solvers import direct_solver
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
from peridynamic_damage import *
from datetime import datetime as dttm
import mshr                                 
from os import path, getcwd, mkdir

def run_fracture_test():
    #m = box_mesh(Point(0,0,0), Point(2,1,1), 10,5,5)
    #m = box_mesh_with_hole(numpts=20)
    #domain = mshr.Rectangle(Point(0,0), Point(3,1))
    #m = mshr.generate_mesh(domain, 30)
    m = RectangleMesh(Point(0,0), Point(2,1), 30, 15)
    #m = rectangle_mesh(numptsX=20, numptsY=10)
    #m = rectangle_mesh_with_hole(npts=25)
    
    struct_grd = True
    vol_corr   = True
    dmg_flag   = True
    save_fig   = True
    pplot      = False
    
    f0 = 10e8
    delta_t = 0.00025
    ft = 50e9
    t  = 0.0
    
    bc_type = {'dirichlet':0,'forceX':1}
    bc_vals = {'dirichlet':0,'forceX':0}
    bc_loc = [0,1]
    num_lyrs = 2 #num of additional layers on boundary
    cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd) 
    dim = np.shape(cell_cent[0])[0]
    el = get_peridym_edge_length(cell_cent, struct_grd)
   
    # dirichlet bc on ghost layers
    node_ids_dir = get_boundary_layers(cell_cent, el, 3, bc_loc, struct_grd)
    node_ids_frc = get_boundary_layers(cell_cent, el, 2*num_lyrs, bc_loc, struct_grd)
    ghost_lyr_node_ids = node_ids_dir

    omega_fun = gaussian_infl_fun2
    E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)

    horizon = 3.001*np.abs(np.diff(cell_cent[0:2][:,0])[0])

    num_steps = 500
    u_disp = np.zeros((len(cell_cent), dim), dtype=float)
    
    G0 = 1000000/2
    s0 = s0 = compute_critical_stretch(G0, bulk, horizon)

    if save_fig:
        pwd = getcwd()
        today = dttm.now().strftime("%Y%m%d%%H%M%S")
        data_dir_top = path.join(pwd, 'fractue_test_tensile')
        mkdir(data_dir_top)
    else:
        data_dir_top = None

    for i in range(num_steps):
        
        print("********************************************")
        print("********************************************")
        print("START : time step %i"%i)
        
        t = (i+1)*delta_t
        force = f0 + ft*i*delta_t + 0.10*ft*t**2
        print("appyling external body force: %.4g" %force)

        bc_vals['forceX'] = force
        extents = compute_modified_extents(cell_cent, el, struct_grd)
        
        tree = QuadTree()
        tree.put(extents, horizon)
        
        nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
        bnd_dmg_lst = compute_bond_damage(s0, cell_cent, nbr_lst, u_disp, dmg_flag)
        mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst, nbr_beta_lst, horizon, omega_fun)
        
        K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, bnd_dmg_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun, u_disp)
        
        K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, node_ids_dir, node_ids_frc, struct_grd)
        
        img_name = 'fracture_test_'+str(i).zfill(3)+'.png'
        if save_fig:
            data_dir = path.join(data_dir_top, img_name)
        else:
            data_dir = None
        u_disp = direct_solver(K_bound, fb, dim, reshape=True)
        cell_cent_orig, u_disp_orig, u_disp_ghost = recover_original_peridynamic_mesh(cell_cent, u_disp, bc_type, ghost_lyr_node_ids, struct_grd)
        disp_cent = get_displaced_soln(cell_cent_orig, u_disp_orig, horizon, dim, data_dir, plot_=pplot, save_fig=save_fig, zoom=10)

        u_disp = u_disp_ghost
        cell_cent += u_disp

        print("END : time step %i"%i)
        print("********************************************")
        print("********************************************")

