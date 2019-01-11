from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
import mshr


def test_structured_grid():
    m = RectangleMesh(Point(0,0), Point(3,1),60,30)
    structured_mesh = True
    cell_cent = structured_cell_centroids(m)
    cell_vol = structured_cell_volumes(m)
    purtub_fact = 1e-6
    dim = np.shape(cell_cent[0])[0]
    
    omega_fun = gaussian_infl_fun1
    E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)
    
    extents = get_domain_bounding_box(m)
    horizon = 3*np.diff(cell_cent[0:2][:,0])
    tree = QuadTree()
    tree.put(extents, horizon)
    
    nbr_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon)
    mw = peridym_compute_weighted_volume(m, nbr_lst, horizon, omega_fun,structured_mesh=True)
    
    K = computeK(horizon, cell_vol, nbr_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun)
    bc_type = {0:'dirichlet', 1:'force'}
    bc_vals = {'dirichlet': 0, 'force': -5e8}
    
    K_bound, fb = peridym_apply_bc(m, K, bc_type, bc_vals, cell_vol, structured_mesh)
    
    print("solving the stystem")
    start = tm.default_timer()
    sol = np.linalg.solve(K_bound, fb)
    end = tm.default_timer()
    print("Time taken for solving the system of equation: %4.3f secs" %(end-start))
    u_disp = copy.deepcopy(sol)#
    u_disp = np.reshape(u_disp, (int(len(sol)/dim), dim))
    a, _ = get_peridym_mesh_bounds(m, structured_mesh)
    
    node_ids = a[0][0] #normal to x directon
    
    for i, nk in enumerate(node_ids):
        u_disp = np.insert(u_disp, nk, np.zeros(dim, dtype=float), axis=0)
    
    
    disp_cent = cell_cent + u_disp
    
    if dim == 2:
        x, y = cell_cent.T
        plt.figure()
        plt.scatter(x,y, s=300, color='r', marker='o', alpha=0.1, label='original config')
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
    
