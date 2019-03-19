from helper import *
from fenics_mesh_tools import *

def recover_bc_dictonary_with_unique_values(bc_type):
    """
    we need to remove duplicates from bc_type dictonary
    to avoid deleting possible same set of nodes having 
    different boundary conditions eg {'dririclet':0, forceX':1, forceY:1}
    here 1 refers to node set on right end of domain.

    This routine is needed to recover original grid by removing the
    ghost layer particles that are applied to the locations where
    we apply our bounary conditions. 
    
    input data is a dictonary with keys being the name of boundary condition
    and values being the location and thsese locations might be duplicated
    input:
    ------
        bc_type : dictonary with boundary conditon data 

    output:
    -------
        bc_type_new : dictonary with boundary conditon data 
                      with duplicates removed
    """
    bc_typ = bc_type.keys()
    bc_type_new = {}
    for bct in bc_typ:
        bc_loc = bc_type[bct]
        if bc_loc not in bc_type_new.values():
            bc_type_new[bct] = bc_loc 

    return bc_type_new

    
def recover_original_peridynamic_mesh(cell_cent, u_disp, bc_type, ghost_lyr_node_ids,  struct_grd=False, expnd_u_dsp=True):
    """
    given the cell_centroids (with ghost layer on original peridynamic mesh)
    and the locations where the bc's have been applied, the method
    recovers the original mesh (centroids) and it's corresponding solution

    input:
    ------
    cell_cent   : cell centroids with ghost layers where bc has been applied
    u_disp      : displacement field (nodal points fixed dirichlet are deletd 
                  in the soln array)
    bc_type     : dictonary of structured grid 
    struct_grd  : boolean, whether or not struct grid 
    expnd_u_dsp : boolean, whether the retrun value should include 
                     u_disp with ghost layer of dirichlet nodes added 
                     to it
    output:
    -------
    orig_cell_cent :
    orig_u_dsp     :
    u_disp_ghst    : u_disp with ghost layer of fixed dirichlet nodes added to it 

    """
    bc_type_new = recover_bc_dictonary_with_unique_values(bc_type)
    bc_typ = bc_type_new.keys()

    dim = len(cell_cent[0])
    a = ghost_lyr_node_ids
    b = get_bound_cell_cents(ghost_lyr_node_ids, cell_cent)
    u_dsp_ghst = cpy.deepcopy(u_disp)

    del_ids = np.zeros(0, dtype = int) #placeholder for ghost lyer node ids
    for bct in bc_typ:
        bc_loc = bc_type[bct]
        if(bct == 'dirichlet'):
            dir_node_ids = a[bc_loc][0]
            for i, nk in enumerate(dir_node_ids):
                u_dsp_ghst = np.insert(u_dsp_ghst, nk, np.zeros(dim, dtype=float), axis=0)
        del_ids = np.concatenate((del_ids, a[bc_loc][0]), axis=0)

    del_ids = np.unique(del_ids)
    orig_cell_cent = np.delete(cell_cent, del_ids, axis=0)
    orig_u_dsp    = np.delete(u_dsp_ghst, del_ids, axis=0)

    if expnd_u_dsp:
        return orig_cell_cent, orig_u_dsp, u_dsp_ghst
    else:
        return  orig_cell_cent, orig_u_dsp 
    
def recover_stiffness_for_original_mesh(K, cell_cent, bc_type, ghost_lyr_node_ids, struct_grd=False):
    """
    routine to generate the tangent stiffness matrix for the 
    grid without addiditonal ghost boundary layers

_________:
    ------
        K:  tangent stiffness matrix for grid with ghost layers 
        cell_cent: cell centroid for grid with ghost layers 
        el: edge length
        bc_type: dictonary for boundary conditions
        struct_grd: boolean
    output:
    ------
        K_orig : 

    """
    dim = len(cell_cent[0])
    a = ghost_lyr_node_ids
    b = get_bound_cell_cents(ghost_lyr_node_ids, cell_cent)
    del_ids = np.zeros(0, dtype=int)
    K_orig = cpy.deepcopy(K)

    del_keys = ghost_lyr_node_ids.keys()
    for kk in del_keys:
        del_ids = np.concatenate((del_ids, a[kk][0]))

    del_ids = np.unique(del_ids)
    for i, nk in enumerate(del_ids):
        for d in range(dim):
            K_orig = np.delete(K_orig, (nk-i)*dim, axis=0) #deletes the row
            K_orig = np.delete(K_orig, (nk-i)*dim, axis=1) #deletes the col
    
    return K_orig
    


def get_boundary_layers(cell_cent, el, num_lyrs, bc_loc, struct_grd):
    """
    after adding ghost layers, the boundary layers are 
    modified and we need the modified BL's to do 
    further pre- and post-processing

    input:
    ------
        cell_cent: np.array of modified cell centroids
        el       : np array of edge lengths
        num_lyrs : int, number of lyers desired
    ouput:
    ------
        bound_cents: np.array of cell centroids lying on BL along the 
                     given dimension
    """
    dim = len(el)
    bound_range = np.zeros(2*dim, dtype=float)
    bound_nodes = {} #dict to store the node numbers of centroids that lie within bound_range
    bound_cents  = {} #dict to store the node centroids corresponding to node numbers above
    
    if(struct_grd):
        fctr = 1
        corr = 0
        lyrs = float(num_lyrs-1)+ 0.0001
    else:
        fctr = 2
        corr = 1
        lyrs = float(num_lyrs)+ 0.0001

    lyrs = 1.0001*float(num_lyrs-1)
    
    for d in range(dim):
        bound_range[2*d] = np.min(cell_cent[:,d]) + corr*np.diff(np.unique(cell_cent[:,d])[0:2])[0] + lyrs*el[d]
        bound_range[2*d+1] = np.max(cell_cent[:,d]) - corr*np.diff(np.unique(cell_cent[:,d])[0:2])[0] - lyrs*el[d]

        bound_nodes[2*d] = np.where(cell_cent[:,d] <= bound_range[2*d])
        bound_nodes[(2*d+1)] = np.where(cell_cent[:,d] >= bound_range[2*d+1])


    #store only those key value pair that are in the bc_loc
    keys = bound_nodes.keys()
    keys_temp = [kk for kk in keys]
    for kk in keys_temp:
        if kk not in bc_loc:
            bound_nodes.pop(kk, None)
            
    return bound_nodes

def get_bound_cell_cents(bound_nodes, cell_cent):
    """
    TODO : doc string

    input:
    -------
        bound_node_ids:
        cell_cent     :
        dim           :

    output:
    -------
        bound_cell_cent: 
    """
    bound_cents = {}
    keys = bound_nodes.keys()
    for kk in keys:
        bound_cents[kk]   = cell_cent[bound_nodes[kk][0]]

    return bound_cents
    
def peridym_apply_bc(K, bc_type, bc_vals, cell_cent, cell_vol, node_ids_dir, node_ids_frc, struct_grd=False):
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
    bc_name = bc_type.keys()
    for k in bc_name:
        print("%s node set : %s bc" %(bc_type[k], k))
    print("\n")

    dim = len(cell_cent[0])
    dof = np.shape(K)[0]
    K_bound = copy.deepcopy(K)
    force_dir = {"forceX": 0, "forceY":1, "forceZ":2}
    #rhs is a 1-D array
    rhs = np.zeros(dof,dtype=float) #create a rhs filled with zeros

    """
    to map the equivalent of volume bc to the edge, we apply the volume bc to as many number of layers within the volume as the number of ghost layers 
    """

    #apply force on the rhs
    for bcn in bc_name:
        if bcn[0:5] == "force":
            b = get_bound_cell_cents(node_ids_frc, cell_cent)
            bb = bc_type[bcn] #bc location on grid according to comments above
            dd = force_dir[bcn]
            node_ids   = node_ids_frc[bb][0]
            node_cents = b[bb]
            vol_sum = sum(cell_vol[node_ids])
            f_density = bc_vals[bcn]/vol_sum #external force applied as force density
            print("applying foce dirichlet bc on %s nodes"%k)
            if((cell_vol[0] == cell_vol).all()):
                f_density *=cell_vol[0] #precompute for struct mesh
                for i, nk in enumerate(node_ids):
                    rhs[nk*dim+dd] = f_density
            else: #need this for unsrtuctured grids
                for i, nk in enumerate(node_ids):
                    rhs[nk*dim + dd] = f_density*cell_vol[nk]  #hard coded negative y-axis force 
                    #rhs has not yet bc applied to it
    #apply dirichlet bc 
    for bcn in bc_name:
        if bcn is "dirichlet" and bc_vals[bcn] is 0:
            b = get_bound_cell_cents(node_ids_dir, cell_cent)
            bb = bc_type[bcn]
            node_ids   = node_ids_dir[bb][0]
            print("applying dirichlet bc on %s nodes" )

            for i, nk in enumerate(node_ids):
                for d in range(dim):
                    K_bound = np.delete(K_bound, (nk-i)*dim, axis=0) #deletes the row
                    K_bound = np.delete(K_bound, (nk-i)*dim, axis=1) #deletes the col
                    rhs     = np.delete(rhs, (nk-i)*dim)                   #deletes the row on rhs

    print("time taken for the application of boundary condition is %4.3f seconds"%(tm.default_timer()-start))

    return K_bound, -rhs 

