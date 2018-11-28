from helper import *
from fenics_mesh_tools import *


def peridym_apply_bc(mesh, K, bc_type, bc_vals, force=-1e9):
    """
    the function applies a boundary conditions on the mesh provided
    BC type : Dirichlet(fixed, displacement), Neumann
            ____________________________________________
           /¦                                          /¦
          / ¦                       y_max             / ¦
         /  ¦                                        /  ¦
        /___¦_______________________________________/   ¦
        ¦   ¦                                      ¦    ¦
        ¦   ¦                                      ¦   x_max 
  x_min ¦   ¦                                      ¦    ¦
        ¦   ¦ z_max                                ¦    ¦
        ¦   ¦______________________________________¦____¦ 
        ¦   /                                      ¦   /
        ¦  /                    y_min              ¦  / 
        ¦ /                                        ¦ / 
        ¦/__z_min__________________________________¦/
    
        ¦z /y
        ¦ /
        ¦/ 
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
    #dictonary that maps integral numbers to appropirate surface normal
    dir_to_surface_map = {0:"x_min", 1:"x_max", 2:"y_min", 3:"y_max", 4:"z_min", 5:"z_max"}

    print("boundary conditions on the mesh:")
    bound_name = bc_type.keys()
    for k in bound_name:
        print("%s node set : %s bc" %(k, bc_type[k]))
    print("\n")

    dim = len(get_cell_centroids(mesh)[0])
    dof = np.shape(K)[0]

    K_bound = copy.deepcopy(K)

    #rhs is a 1-D array
    rhs = np.zeros(dof,dtype=float) #create a rhs filled with zeros

    # 'a': dictonary is node numbers for boundaries 
    # 'b': dictonary are the corresponding node centroids
    # a.keys() = {0,1,2,3,4,5,} with each key mapping to the normals along axes as outlined below:
            # 0 : x_min
            # 1 : x_max
            # 2 : y_min
            # 3 : y_max
            # 4 : z_min
            # 5 : z_max
            #see the diagram in doc string comments above
    a, b = get_peridym_mesh_bounds(mesh)

    #apply force on the rhs
    for bb in bound_name:
        
        if bc_type[bb] is "force":
            node_ids   = a[bb][0]
            node_cents = b[bb]
            print("applying foce dirichlet bc on %s nodes"%k)
            for i, nk in enumerate(node_ids):
                rhs[nk*dim + 1] = bc_vals[bc_type[bb]] #hard coded negative y-axis force (denoted by 1)
                    #rhs has not yet bc applied to it
    #apply dirichlet bc 
    for bb in bound_name:

        if bc_type[bb] is "dirichlet" and bc_vals[bc_type[bb]] is 0:
            node_ids   = a[bb][0]
            print("applying dirichlet bc on %s nodes" )

            for i, nk in enumerate(node_ids):
                for d in range(dim):
                    K_bound = np.delete(K_bound, (nk-i)*dim, axis=0) #deletes the row
                    K_bound = np.delete(K_bound, (nk-i)*dim, axis=1) #deletes the col
                    rhs     = np.delete(rhs, (nk-i)*dim)                   #deletes the row on rhs

    return K_bound, -rhs 

