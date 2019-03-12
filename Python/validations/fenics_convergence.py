import colorsys 
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
    markers = ['1','2','3','4','s','p','*','h','+','x','o','^','x','d', 'v','<', '>', ',','|']
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

    mesh_lst = generate_mesh_list(num_meshes=20, numptsX=numptsX, numptsY=numptsY)
    num = len(mesh_lst)
    dim = len(mesh_lst[0].coordinates()[0])
    fine_mesh = mesh_lst[-1]
    mesh_lst.pop(len(mesh_lst)-1)
    dim = fine_mesh.topology().dim()
    if(struct_grd):
        cell_cent_fine = structured_cell_centroids(fine_mesh)
    else:
        cell_cent_fine = get_cell_centroids(fine_mesh)

#################################################################################
#################################################################################
        ## compute a benchmark solution with a very fine grid
    top_els = np.ravel((np.argwhere(cell_cent_fine[:,1] == np.max(cell_cent_fine[:,1]))))
    cell_cent_top_fine = cell_cent_fine[top_els]
    print("Solving for a fine grid with %i cells\n"%fine_mesh.num_cells())
    u_fe = solve_fenic_bar(fine_mesh, cell_cent_fine, plot_=False)
    u_disp_top_fine = np.zeros((len(cell_cent_top_fine), dim), dtype=float)
    for i, cc in enumerate(cell_cent_top_fine):
        u_disp_top_fine[i] = u_fe(cc)

#################################################################################
#################################################################################

    u_disp_top_lst = []
    error_lst = []
    err = 1
    idx = 0
    err = np.ones(len(u_disp_top_fine), dtype=float)
    new_mesh_lst = None 
    u_fe_conv = None 
    while(idx<len(mesh_lst)):
        print("%i. solving for FE convergence"%idx)
        u_disp_top_coar = np.zeros((len(cell_cent_top_fine), dim), dtype=float)
        mm = mesh_lst[idx]
        cell_cent = get_cell_centroids(mm)
        u_fe = solve_fenic_bar(mm, cell_cent, plot_=False)

        for i, cc in enumerate(cell_cent_top_fine):
            u_disp_top_coar[i] = u_fe(cc)
        
        u_disp_top_lst.append(u_disp_top_coar)
        err = la.norm(u_disp_top_fine-u_disp_top_coar, 2, axis=1)
        error_lst.append(err)
        if((err<=tol).all()):
            u_fe_conv = u_fe
            new_mesh_lst = mesh_lst[0:idx+1]
            print("error converged within tolerance of %.3e at iteration %i"%(tol, idx))
            break
        idx += 1

    plt.figure()
    colors = get_colors(len(u_disp_top_lst)+1)
    xx = cell_cent_top_fine[:,0]
    ii = 0
    for i,_ in enumerate(u_disp_top_lst):
        yy = u_disp_top_lst[i][:,1]
        plt.plot(xx, yy, color=colors[i], linewidth=2, label='num cells:'+str(mesh_lst[i].num_cells()))
        ii += 1

    plt.plot(xx, u_disp_top_fine[:,1], color=colors[ii], linewidth=2,  label='num cells:'+str(fine_mesh.num_cells()))
    plt.title('mesh convergence for FE soln, bar of size 3x1, load=-5e-8, tol=%.3e'%tol)
    plt.legend()
    plt.show(block=False)

    #u_disp_top_lst.append(u_disp_top_fine)
    return u_disp_top_lst, cell_cent_top_fine, new_mesh_lst, u_fe_conv 
