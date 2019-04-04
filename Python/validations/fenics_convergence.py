import colorsys 
import matplotlib.ticker as mtick
from fenics_plane_stress import *
import sys

def get_colors(num_colors):
    """
    creates n distint colors for plts, where n = num_colors
    source: https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors/4382138#4382138
    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
         hue = i/360.
         lightness = (50 + np.random.rand() * 10)/100.
         saturation = (90 + np.random.rand() * 10)/100.
         colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def get_markers(num_markers):
    """
    returns list of markers to be used for plt 
    functions; max markers allowed = 18

    input:
    ------
        num_markers: int, number of markers needed
    output:
    -------
        markers : list of distinct plotting markers

    """
    markers = ['^','o','P','X','*', 'd','<', '>', ',','|', '1','2','3','4','s','p','*','h','+']
    if(num_markers>18):
        sys.exit("cannot create more than 18 markers, refactor your code; force exiting")

    return markers[0:num_markers]


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

def fenics_mesh_convergence(struct_grd=False, numptsX=10, numptsY=5, tol=None, plot_=True):
    """
    checks the convergence of fenics for a 
    2D displacement 

    input:
    ------
        mesh_lst : list of mesh with different dicretizations
        plot_    : boolean, whether we want to plot FE solns
    """
    if(tol == None):
        tol = 1e-5
    
    #assign function reference for cell centroids
    if struct_grd:
        cell_centroid_function = structured_cell_centroids
    else: 
        cell_centroid_function = get_cell_centroids

    mesh_lst = generate_mesh_list(num_meshes=20, numptsX=numptsX, numptsY=numptsY)
    num = len(mesh_lst)
    dim = len(mesh_lst[0].coordinates()[0])
    fine_mesh = mesh_lst[-1]
    mesh_lst.pop(len(mesh_lst)-1)
    dim = fine_mesh.topology().dim()
    cell_cent_fine = cell_centroid_function(fine_mesh)

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

    ##solve the coarse solution
    print("solving bending problem for coarse grid with %i elements "%mm_i.num_cells())
    
    cell_cent_i = cell_centroid_function(mm_i)
    u_fe_i = solve_fenic_bar(mm_i, cell_cent_i, plot_=False)

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

        #### get the topmost cell centroids of the beam in fine mesh #######
        top_els_i= np.ravel((np.argwhere(cell_cent_i[:,1] == np.max(cell_cent_i[:,1]))))
        cell_cent_top_i = cell_cent_i[top_els_i]

        #solve the two solutions
        print("solving bending problem for fine grid with %i elements "%mm_j.num_cells())
        u_fe_j = solve_fenic_bar(mm_j, cell_cent_j, plot_=False)
        
        #declare empty array to store  displacement soulution of cell centroids 
        u_disp_top_i = np.zeros((len(cell_cent_top_i), dim), dtype=float)
        u_disp_top_j = np.zeros((len(cell_cent_top_i), dim), dtype=float)

        for idx, cc in enumerate(cell_cent_top_i):
            u_disp_top_i[idx] = u_fe_i(cc)
            u_disp_top_j[idx] = u_fe_j(cc)
        
        err = la.norm(u_disp_top_j-u_disp_top_i, 2, axis=1)
        error_lst.append(err)
        err_norm_lst.append(np.max(err))
        u_disp_top_lst.append(u_disp_top_i)
        cell_cent_top_lst.append(cell_cent_top_i)
        if((err<=tol).all()):
            u_fe_conv = u_fe_i
            #new_mesh_lst = mesh_lst[0:idx+1]
            print("error converged within tolerance of %.3e at iteration %i"%(tol, idx))
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
