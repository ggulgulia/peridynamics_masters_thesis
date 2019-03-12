from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_solvers import direct_solver
#from peridynamic_correspondence import *
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
import mshr

#m = box_mesh(Point(0,0,0), Point(2,1,1), 10,5,5)
#m = box_mesh_with_hole(numpts=20)
m = rectangle_mesh(numptsX=10, numptsY=5)
#m = rectangle_mesh_with_hole(npts=25)

struct_grd = True
vol_corr = False
bc_type = {'dirichlet':0,'forceX':3}
bc_vals = {'dirichlet':0,'forceX':50e8}
bc_loc = [0,3]
num_lyrs = 2 #num of additional layers on boundary
cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd) 

el = get_peridym_edge_length(cell_cent, struct_grd)
extents = compute_modified_extents(cell_cent, el, struct_grd)

purtub_fact = 1e-6
dim = np.shape(cell_cent[0])[0]

omega_fun = gaussian_infl_fun2
E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)

horizon = 3.001*np.abs(np.diff(cell_cent[0:2][:,0])[0])
#horizon = 0.3001
tree = QuadTree()
tree.put(extents, horizon)

nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst, nbr_beta_lst, horizon, omega_fun)

K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun)
K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, num_lyrs, struct_grd)

u_disp = direct_solver(K_bound, fb, dim, reshape=True)

cell_cent_orig, u_disp_orig = recover_original_peridynamic_mesh(cell_cent, u_disp, el, bc_type, num_lyrs, struct_grd)
K_orig = recover_stiffness_for_original_mesh(K, cell_cent, el, bc_type, num_lyrs, struct_grd)

u_disp_flat = np.zeros(len(cell_cent_orig)*dim, dtype=float)
internal_force = np.zeros( (len(cell_cent_orig), dim), dtype=float)

#flatten the array and compute internal forces in the particles
for d in range(dim):
    u_disp_flat[d::dim] = u_disp_orig[:,d]

internal_force_flat = np.matmul(K_orig, u_disp_flat)
for d in range(dim):
    internal_force[:,d] = internal_force_flat[d::dim]

disp_cent = plot_displaced_soln(cell_cent_orig, u_disp_orig, horizon, dim, zoom=40)
