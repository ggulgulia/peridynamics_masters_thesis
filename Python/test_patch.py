from peridynamic_neighbor_data import *
from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_stiffness import*
from peridynamic_solvers import direct_solver
#from peridynamic_correspondence import *
from peridynamic_boundary_conditions import *
from peridynamic_infl_fun import *
from peridynamic_materials import *
from peridynamic_damage import *
import mshr

m = RectangleMesh(Point(0,0), Point(2,1), 40, 20)

struct_grd = True
vol_corr = False
bc_type = {'dirichletX':0,'forceX':1}
bc_vals = {'dirichletX':0,'forceX':500e8}
bc_loc = [0,1]
num_lyrs = 2 #num of additional layers on boundary
cell_cent, cell_vol = add_ghost_cells(m, bc_loc, num_lyrs, struct_grd) 
el = get_peridym_edge_length(cell_cent, struct_grd)
extents = compute_modified_extents(cell_cent, el, struct_grd)
dim = np.shape(cell_cent[0])[0]

node_ids_dir = get_boundary_layers(cell_cent, el, num_lyrs, bc_loc, struct_grd)
node_ids_frc = get_boundary_layers(cell_cent, el, 2*num_lyrs, bc_loc, struct_grd)
ghost_lyr_node_ids = node_ids_dir

node_cents_dir = get_bound_cell_cents(node_ids_dir, cell_cent)
node_cents_frc = get_bound_cell_cents(node_ids_frc, cell_cent)
omega_fun = gaussian_infl_fun2
E, nu, rho, mu, bulk, gamma = get_steel_properties(dim)

horizon = 3.001*np.abs(np.diff(cell_cent[0:2][:,0])[0])
#horizon = 0.3001
tree = QuadTree()
tree.put(extents, horizon)

G0 = 1000000/2 
s0 = compute_critical_stretch(G0, bulk, horizon)

nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, vol_corr, struct_grd)
mw = peridym_compute_weighted_volume(cell_cent, cell_vol, nbr_lst, nbr_beta_lst, horizon, omega_fun)

#initialize 
u_disp = np.zeros((len(cell_cent), dim), dtype=float)
bnd_dmg_lst = compute_bond_damage(s0, cell_cent, nbr_lst, u_disp)

K = computeK(horizon, cell_vol, nbr_lst, nbr_beta_lst, bnd_dmg_lst, mw, cell_cent, E, nu, mu, bulk, gamma, omega_fun, u_disp)
K_bound, fb = peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, node_ids_dir, node_ids_frc, struct_grd)

u_disp = direct_solver(K_bound, fb, dim, reshape=True)

cell_cent_orig, u_disp_orig, u_disp_ghost = recover_original_peridynamic_mesh(cell_cent, u_disp, bc_type, ghost_lyr_node_ids, struct_grd)
K_orig = recover_stiffness_for_original_mesh(K, cell_cent, bc_type, ghost_lyr_node_ids, struct_grd)

u_disp_flat = np.zeros(len(cell_cent_orig)*dim, dtype=float)
disp_cent = get_displaced_soln(cell_cent_orig, u_disp_orig, horizon, dim, plot_=True, zoom=20)
