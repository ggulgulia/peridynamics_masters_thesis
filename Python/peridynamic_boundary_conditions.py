from helper import *
from fenics_mesh_tools import *


def peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, num_lyrs=2, struct_grd=False):
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
    
        ¦y /z
        ¦ /
        ¦/ 
       -¦------>X

    ASSUMPTION : the mesh is a rectangular mesh 

    dir_to_surface_map = {0:"x_min", 1:"x_max", 2:"y_min", 3:"y_max", 4:"z_min", 5:"z_max"}

    input:
    ------
    cell_cent: peridynamic cell centroids
    K   : tangent stiffness matrix 
    bc_type: TODO
    bc_location: TODO
    
    output:
    -------
        K_bound   : np.ndarry stiffness matrix with bc applied
        u         :

    """
    import timeit as tm
    print("beginning the application of boundary conditions to the given mesh")
    start = tm.default_timer()

    print("boundary conditions on the mesh:")
    bound_name = bc_type.keys()
    for k in bound_name:
        print("%s node set : %s bc" %(k, bc_type[k]))
    print("\n")

    dim = len(cell_cent[0])
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

    """
    to map the equivalent of volume bc to the edge, we apply the volume bc to as many number of layers within the volume as the number of ghost layers 
    """
    el = get_peridym_edge_length(cell_cent, struct_grd)
    a, b = get_modified_boundary_layers(cell_cent, el, 2*num_lyrs, struct_grd)

    #apply force on the rhs
    for bb in bound_name:
        if bc_type[bb] is "force":
            node_ids   = a[bb][0]
            node_cents = b[bb]
            vol_sum = sum(cell_vol[node_ids])
            f_density = bc_vals[bc_type[bb]]/vol_sum #external force applied as force density
            print("applying foce dirichlet bc on %s nodes"%k)
            if((cell_vol[0] == cell_vol).all()):
                f_density *=cell_vol[0] #precompute for struct mesh
                for i, nk in enumerate(node_ids):
                    rhs[nk*dim+1] = f_density
            else: #need this for unsrtuctured grids
                for i, nk in enumerate(node_ids):
                    rhs[nk*dim + 1] = f_density*cell_vol[nk]  #hard coded negative y-axis force 
                    #rhs has not yet bc applied to it
    #apply dirichlet bc 
    a, b = get_modified_boundary_layers(cell_cent, el, num_lyrs, struct_grd)
    for bb in bound_name:

        if bc_type[bb] is "dirichlet" and bc_vals[bc_type[bb]] is 0:
            node_ids   = a[bb][0]
            print("applying dirichlet bc on %s nodes" )

            for i, nk in enumerate(node_ids):
                for d in range(dim):
                    K_bound = np.delete(K_bound, (nk-i)*dim, axis=0) #deletes the row
                    K_bound = np.delete(K_bound, (nk-i)*dim, axis=1) #deletes the col
                    rhs     = np.delete(rhs, (nk-i)*dim)                   #deletes the row on rhs

    print("time taken for the application of boundary condition is %4.3f seconds"%(tm.default_timer()-start))

    return K_bound, -rhs 

