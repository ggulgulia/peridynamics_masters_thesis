from fenics_mesh_tools import *

def recover_original_peridynamic_mesh(cell_cent, u_disp, el, bc_type, num_lyrs, struct_grd=False):
    """
    given the cell_centroids (with ghost layer on original peridynamic mesh)
    and the locations where the bc's have been applied, the method
    recovers the original mesh (centroids) and it's corresponding solution

    input:
    ------
    cell_cent : TODO
    u_disp    : TODO
    el        : numpy arrays of edge length
    bc_type   :
    num_lyrs  : 
    struct_grd: TODO
    
    output:
    -------
    orig_cell_cent :
    orig_u_disp    :

    """
    dim = len(cell_cent[0])
    keys = bc_type.keys()
    a, b = get_modified_boundary_layers(cell_cent, el, num_lyrs, struct_grd)

    del_ids = np.zeros(0, dtype = int) #placeholder for ghost lyer node ids
    for kk in keys:
        bc_name = bc_type[kk]
        if(bc_name == 'dirichlet'):
            node_ids = a[kk][0]

            for i, nk in enumerate(node_ids):
                u_disp = np.insert(u_disp, nk, np.zeros(dim, dtype=float), axis=0)

        del_ids = np.concatenate((del_ids, a[kk][0]), axis=0)

    orig_cell_cent = np.delete(cell_cent, del_ids, axis=0)
    orig_u_disp    = np.delete(u_disp, del_ids, axis=0)

    return orig_cell_cent, orig_u_disp
