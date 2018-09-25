from helper import *
from mesh_tools import *


def peridym_apply_bc(mesh, K, bc_list, force=-1e9):
    """
    the function applies a boundary conditions on the mesh provided
    BC type : Dirichlet(fixed, displacement), Neumann
        
        ¦y
        ¦
        ¦
       -¦------>X

    ASSUMPTION : the mesh is a rectangular mesh 

    :mesh: TODO
    :K   : tangent stiffness matrix 
    :bc_type: TODO
    :bc_location: TODO
    :returns:
    --------
        K_bound   : np.ndarry stiffness matrix with bc applied
        u         :

    """

    print("boundary conditions on the mesh:")
    bound_name = bc_list.keys()
    for k in bound_name:
        print("%s node set : %s bc" %(k, bc_list[k]))

    print("\n")

    #numel = len(np.array(mesh.elements))
    dim = len(get_elem_centroid(mesh)[0])
    dof = np.shape(K)[0]

    K_bound = copy.deepcopy(K)
    rhs = np.zeros(dof,dtype=float) #create a rhs filled with zeros
    #bound_elem_id, bound_elem_coords = get_peridym_mesh_bounds(mesh)
    a, b = get_peridym_mesh_bounds(mesh)
    # 'a' is node numbers for bounaries 'left', 'right', 'top', 'bottom'
    # 'b' are the corresponding node centroids

    for k in bound_name:
        node_set = b[k]
        node_ids = a[k]
        bc_type = bc_list[k]

        if bc_type is "dirichlet":
            print("applying dirichlet bc on %s nodes"%k)

            for i, nk in enumerate(node_ids):
                for d in range(dim):
                    K_bound = np.delete(K_bound, nk*dim + d, axis=0) #deletes the row
                    K_bound = np.insert(K_bound, nk*dim + d, np.zeros(dof, dtype=float), axis=0)
                    K_bound[nk*dim + d][nk*dim + d] = 1.0

        if bc_type is "force":
            print("applying foce dirichlet bc on %s nodes"%k)
            for i, nk in enumerate(node_ids):
                for d in range(dim):
                    rhs[nk*dim + 1] = force #hard coded negative y-axis force
                    #rhs has not yet bc applied to it

    fb = copy.deepcopy(rhs)  #force vector with bc applied
    ll = b['left']
    lkey = a['left']

    for i, k in enumerate(lkey):
        for d in range(dim):
            fb[k*dim + d] = 0.0

    return K_bound, fb 
