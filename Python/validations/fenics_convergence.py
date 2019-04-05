import matplotlib.ticker as mtick
from fenics_plane_stress import *
from fenics_patch_test import solve_patch_test
import sys


def generate_mesh_list(corner1=Point(0,0), corner2=Point(3,1), numptsX=30, numptsY=10, delta=10, num_meshes=3):
    """
    generates mesh array for convergence tests
    inputs:
    -------
        corner1: Point minima of the Rectangle Mesh 
        corner2: Point maxima of the Rectangle Mesh 
        numptsX: int, base number of points in X dir
        numptsY: int base number of points in Y dir
        delta  : increments in number of points  
        num_meshes : number of mesh in the list with incremental refinements 

    output:
    -------
        mesh_lst : lst of fenics mesh of lenth num_meshes
    """

    mesh_lst = []

    for i in range(num_meshes):
        nxpts = numptsX + i*delta
        nypts = numptsY + int((i*delta)/3) 
        mm = RectangleMesh(corner1, corner2, nxpts, nypts)
        mesh_lst.append(mm)
    
    return mesh_lst

def generate_refined_mesh_(corner1=Point(0,0), corner2=Point(3,1), numptsX=30, numptsY=10, delta=10):
    """
    generates mesh array for convergence tests
    inputs:
    -------
        corner1: Point minima of the Rectangle Mesh 
        corner2: Point maxima of the Rectangle Mesh 
        numptsX: int, base number of points in X dir
        numptsY: int base number of points in Y dir
        delta  : increments in number of points  
        num_meshes : number of mesh in the list with incremental refinements 

    output:
    -------
        mesh_lst : lst of fenics mesh of lenth num_meshes
    """


    #nxpts = numptsX + i*delta
    #nypts = numptsY + int((i*delta)/3) 
    mm = RectangleMesh(corner1, corner2, nxpts, nypts)
    
    return mm

def generate_struct_mesh_list_for_pd_tests():
    """
    generates 2 separate lists of fenics mesh
    ensuring consistent particle sizes
    in non-uniform and uniform mesh distribution

    this generates a structured mesh for peridynamics
    with particle counts 648, 968, 1352, 1800, 
    input: None
    ------

    output:
    -------
        unstrct_msh_lst : list of fenics mesh for unstruct_grid test
        struct_msh_lst  : list of FEniCS mesh for struct_grd test
        
    """

    unstrct_msh_lst = []
    struct_msh_lst  = []

    delta = 10;
    minNumptsX = 36;
    minNumptsY = 18;
    
    ratio = minNumptsX/minNumptsY
    delta = 4
    num_meshes = 5
    for i in range(num_meshes):
        numptsX = int(minNumptsX + i*delta*ratio)
        numptsY = int(minNumptsY + i*delta)
        m = RectangleMesh(Point(0,0), Point(2,1), numptsX, numptsY)
        struct_msh_lst.append(m)

        numptsY = int(numptsY*0.5)
        m = RectangleMesh(Point(0,0), Point(2,1), numptsX, numptsY)
        unstrct_msh_lst.append(m)

    return unstrct_msh_lst, struct_msh_lst

def interpolate_fe_soln_at_boundary(u_fe, cell_cent, boundaryName='top'):
    """
    depending on the problem we are solving, we might want to obtain the
    fe solution interpolated at the (boundary)particle location of 
    corresponding peridynamic discretization. For this purpose the method
    does the interpolation correctly and returns the solution

    NOTE: the method works only for 2d  rectangular geometry 
    and is done to make my life easier while doing tests 
    my masters for thesis

    :u_fe: FE solution at nodal locations(from FENICS)
    :boundaryName: name of the boundary where solution is desired
    :cell_cent: peridynamic discretization of domain 
    :returns: u_fe_atBoundary

    """
    #cardinal search dimension for boundary locations
    ## for left/right we look along x-dim(index 0)
    ## for top/bottom we look along y-dim(index 1)
    searchDimDict = {'top':1,'bottom':1, 'left':0, 'right':0}
    sd = searchDimDict[boundaryName]

    if boundaryName == 'top' or boundaryName == 'right':
        boundEls= np.ravel((np.argwhere(cell_cent[:,sd] == np.max(cell_cent[:,sd]))))
    if boundaryName == 'bottom' or boundaryName == 'left':
        boundEls= np.ravel((np.argwhere(cell_cent[:,sd] == np.min(cell_cent[:,sd]))))

    cell_cent_bound = cell_cent[boundEls]
   
    #place holder for interpolated fe solution at boundary of 
    # corresponding peridynamic mesh
    u_fe_bnd_cent = np.zeros((len(cell_cent_bound), 2), dtype=float)
    for idx, cc in enumerate(cell_cent_bound):
        u_fe_bnd_cent[idx] = u_fe(cc)
    return u_fe_bnd_cent, cell_cent_bound

def fenics_mesh_convergence(struct_grd=False, numptsX=10, numptsY=5, problem='transverseTraction',tol=None, plot_=True):
    """
    checks the convergence of fenics for a 
    2D displacement 

    input:
    ------
        mesh_lst : list of mesh with different dicretizations
        plot_    : boolean, whether we want to plot FE solns
    """

    ##initialize solution paramters for the specific problem
    fe_solution_method = {'patchTest': solve_patch_test, 'transverseTraction': solve_fenic_bar}
    boundLocationDict = {'transverseTraction': 'top', 'patchTest':'right'}
    
    #set the reference to correct function
    solve_fe = fe_solution_method[problem]
    bName    = boundLocationDict[problem]

    if(tol == None):
        tol = 1e-5
    
    #assign function reference for cell centroids
    if struct_grd:
        cell_centroid_function = structured_cell_centroids
    else: 
        cell_centroid_function = get_cell_centroids

    u_disp_top_lst = []
    cell_cent_top_lst = []
    error_lst = []
    err = 1
    u_fe_conv = None 

    ##### Initial Mesh
    delta = 10; corner1 = Point(0,0); corner2 = Point(2,1);
    i = 0; 
    nxpts_i = numptsX + i*delta
    nypts_i = numptsY + int((i*delta)/2) 
    mm_i = RectangleMesh(corner1, corner2, nxpts_i, nypts_i) # coarse mesh

    dim = mm_i.topology().dim()
    ##solve the coarse solution
    print("solving bending problem for coarse grid with %i elements "%mm_i.num_cells())
    
    cell_cent_i = cell_centroid_function(mm_i)
    u_fe_i = solve_fe(mm_i, cell_cent_i, plot_=False)

    mesh_lst = []
    err_norm_lst = []
    ############ Generate a coarse mesh and a fine mesh on the next level ########
    while(True):
        j = i+1;
         
        mesh_lst.append(mm_i)
        print(" Level %i. solving for FE convergence"%i)

        ##### get cell cetntroids in coarse and fine mesh ####
        nxpts_j = numptsX + j*delta
        nypts_j = numptsY + int((j*delta)/3) 
        mm_j = RectangleMesh(corner1, corner2, nxpts_j, nypts_j) # fine mesh
        cell_cent_j = cell_centroid_function(mm_j)

        #solve the two solutions
        print("solving bending problem for fine grid with %i elements "%mm_j.num_cells())
        u_fe_j = solve_fenic_bar(mm_j, cell_cent_j, plot_=False)
        
        u_disp_top_j, _ = interpolate_fe_soln_at_boundary(u_fe_j, cell_cent_j, boundaryName=bName)
        u_disp_top_i, cell_cent_top_j= interpolate_fe_soln_at_boundary(u_fe_i, cell_cent_j, boundaryName=bName)

        err = la.norm(u_disp_top_j-u_disp_top_i, 2, axis=1)

        error_lst.append(err)
        err_norm_lst.append(np.max(err))
        u_disp_top_lst.append(u_disp_top_i)
        cell_cent_top_lst.append(cell_cent_top_j)
        if((err<=tol).all()):
            u_fe_conv = u_fe_j
            #new_mesh_lst = mesh_lst[0:idx+1]
            print("error converged within tolerance of %.3e at iteration %i"%(tol, i))
            break

        else:
            #swap assign fine to coarse solutions
            mm_i = mm_j;  i = j; u_fe_i = u_fe_j
            cell_cent_i = cell_cent_j
    err_norm_lst = np.array(err_norm_lst)
    #plt.figure()
    #colors = get_colors(len(u_disp_top_lst)+1)
    #ii = 0
    #for i,_ in enumerate(u_disp_top_lst):
    #    xx = cell_cent_top_lst[i][:,0]
    #    yy = u_disp_top_lst[i][:,1]
    #    plt.plot(xx, yy, color=colors[i], linewidth=2, label='num cells:'+str(mesh_lst[i].num_cells()))
    #    ii += 1

    ##plt.title('mesh convergence for FE soln, bar of size 3x1, load=-5e-8, tol=%.3e'%tol, fontsize=20)
    #plt.legend(loc='center left', fontsize=14)
    #plt.xlabel('top element centroids along x-axis [m]', fontsize=20)
    #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    #plt.yticks(fontsize=14)
    #plt.ylabel('y-displacement of centroids [m]', fontsize=20)
    #plt.xticks(np.arange(0, 3.5, 0.5), fontsize=14)
    #plt.xlim((0, 2.1)); 
    #
    #plt.show(block=False)
    #
    ##L2 norm of error

    #x_axis = np.arange(1, len(err_norm_lst)+1, 1)
    #ax = plt.figure()
    ##plt.title('L2 norm of error for 3x1 steel bar under traction load' ,fontsize=20)
    #plt.plot(x_axis, err_norm_lst, marker='^', markersize=14, linewidth=2, color='k')
    #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    #plt.yticks(np.arange(0, 8e-4, 1e-4),fontsize=14)
    #plt.xticks(np.arange(0, len(err_norm_lst)+2, 1), fontsize=14)
    #plt.xlabel('Refinement level', fontsize=18)
    #plt.ylabel('L2 norm of error', fontsize=18)
    ##for xval, yval in zip(x_axis, err_norm_lst):
    ##    plt.annotate(format(yval, '.3E'),xy=(xval, yval), xytext=(xval+0.05, yval+0.00003), fontsize=12)
    #plt.show(block=False)


    ##u_disp_top_lst.append(u_disp_top_fine)
    return u_disp_top_lst, cell_cent_top_lst, mesh_lst, u_fe_conv 
