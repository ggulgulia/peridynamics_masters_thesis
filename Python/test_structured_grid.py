from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
import mshr


def test_structured_grid():
    m = RectangleMesh(Point(0,0), Point(2,1),20,10)
    struct_grd = True

    bc_loc = [0,1]
    bc_type = {0:'dirichlet', 1:'force'}
    bc_vals = {'dirichlet': 0, 'force': -5e8}
    num_lyrs = 2 #num layers of cells for BC

    cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd)
    el = get_peridym_edge_length(cell_cent, struct_grd)
    extents = compute_modified_extents(cell_cent, struct_grd)
    
    purtub_fact = 1e-6
    dim = np.shape(cell_cent[0])[0]
    
    omega_fun = gaussian_infl_fun1
    E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)
    
    horizon = 3*np.diff(cell_cent[0:2][:,0])
    tree = QuadTree()
    tree.put(extents, horizon)
    
    nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, struct_grd)
    mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst,nbr_beta_lst, horizon, omega_fun)
    
    K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun)
    
    K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, 2, struct_grd)
    
    print("solving the stystem")
    start = tm.default_timer()
    sol = np.linalg.solve(K_bound, fb)
    end = tm.default_timer()
    print("Time taken for solving the system of equation: %4.3f secs" %(end-start))
    u_disp = copy.deepcopy(sol)#
    u_disp = np.reshape(u_disp, (int(len(sol)/dim), dim))
    a, _ = get_modified_boundary_layers(cell_cent, el, num_lyrs, struct_grd)
    
    node_ids = a[0][0] #normal to x directon
    
    for i, nk in enumerate(node_ids):
        u_disp = np.insert(u_disp, nk, np.zeros(dim, dtype=float), axis=0)
    
    disp_cent = cell_cent + u_disp

    del_ids = np.zeros(0, dtype=int)
    for loc in bc_loc:
        del_ids = np.concatenate((del_ids, a[loc][0]))
    
    cell_cent = np.delete(cell_cent, del_ids,axis=0)
    disp_cent = np.delete(disp_cent, del_ids, axis=0)
    u_disp    = np.delete(u_disp, del_ids, axis=0)

    
    if dim == 2:
        plt.figure()
        x, y = cell_cent.T
        plt.scatter(x,y, s=300, color='r', marker='o', alpha=0.3, label='original config')
        x,y = (cell_cent + 40*u_disp).T
        plt.scatter(x,y, s=300, color='b', marker='o', alpha=0.6, label='horizon='+str(horizon))
        #plt.ylim(-0.5, 1.5)
        plt.legend()
        plt.show(block=False)
    
    if dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        x, y, z = cell_cent.T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z, s=300, color='r', marker='o', alpha=0.1, label='original config')
        x,y,z = disp_cent.T
        ax.scatter(x,y,z,s=300, color='g', marker='o', alpha=1.0, label='deformed config')
        ax.axis('off')
        plt.legend()
        plt.show(block=False)
    
