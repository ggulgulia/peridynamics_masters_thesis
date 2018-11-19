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

    #rhs is a 1-D array
    rhs = np.zeros(dof,dtype=float) #create a rhs filled with zeros

    # 'a': dictonary is node numbers for bounaries 'left', 'right', 'top', 'bottom' 
    # 'b': dictonary are the corresponding node centroids
    a, b = get_peridym_mesh_bounds(mesh)

    for bb in bound_name:
        bc_type = bc_list[bb]
        
        #apply force on the rhs
        if bc_type is "force":
            node_set = b[bb]
            node_ids = a[bb]
            print("applying foce dirichlet bc on %s nodes"%k)
            for i, nk in enumerate(node_ids):
                rhs[nk*dim + 1] = force #hard coded negative y-axis force (denoted by 1)
                    #rhs has not yet bc applied to it

    for bb in bound_name:
        bc_type = bc_list[bb]

        if bc_type is "dirichlet":
            node_set = b[bb]
            node_ids = a[bb]
            print("applying dirichlet bc on %s nodes"%k)

            for i, nk in enumerate(node_ids):
                for d in range(dim):
                    K_bound = np.delete(K_bound, (nk-i)*dim + d, axis=0) #deletes the row
                    K_bound = np.delete(K_bound, (nk-i)*dim + d, axis=1) #deletes the col
                    rhs     = np.delete(rhs, (nk-i)*dim+d)                   #deletes the row on rhs


    fb = copy.deepcopy(rhs)  #force vector with bc applied
    ll = b['left']
    lkey = a['left']


    return K_bound, -fb 

def peridym_apply_force_bc(mesh, bound_name, force_dir='neg', force_val=-1e10):
    """TODO: Docstring for peridym_apply_force_bc.

    :mesh: meshpy mesh object
    :bound_name: name of boundary where force is to be applied: 'left', 'right', 'top', 'bottom'
    :force_dir: direction along the given boundary: 'pos' or 'neg'
    :force_val: float , value of force to be appplied
    :returns: 
        f_bc : 1-d array of force bc applied

    """
    return f_bc
