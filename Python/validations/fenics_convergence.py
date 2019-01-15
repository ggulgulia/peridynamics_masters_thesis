from comparisons import *


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


def fenics_mesh_convergence(mesh_lst, plot_=True):
    """
    checks the convergence of fenics for a 
    2D displacement 

    input:
    ------
        mesh_lst : list of mesh with different dicretizations
        plot_    : boolean, whether we want to plot FE solns
    """
    num = len(mesh_lst)
    dim = len(mesh_lst[0].coordinates()[0])

    disp_cent_lst = []
    u_disp_lst = []
    top_els_lst = []
    cell_cents_top_lst = []

    for i, mm in enumerate(mesh_lst):
        num_cells = mm.num_cells()
        cell_cents = get_cell_centroids(mm)
        top_els =  np.ravel(np.argwhere(cell_cents[:,1] == np.max(cell_cents[:,1]))) 
        disp_cent_i, u_disp_i = solve_fenic_bar(mm, cell_cents, plot_=plot_)
        disp_cent_lst.append(disp_cent_i)
        u_disp_lst.append(u_disp_i)
        top_els_lst.append(top_els)
        cell_cents_top_lst.append(cell_cents[top_els])

    plt.figure()
    for i in range(len(mesh_lst)):
        top_els = top_els_lst[i]
        u_top_disp = u_disp_lst[i][top_els]
        plt.plot(cell_cents_top_lst[i][:,0], u_top_disp[:,1], linewidth=2, label='num cells:'+str(mesh_lst[i].num_cells()))

    plt.title('mesh convergence for FE soln, bar of size 3x1, load=-5e-8')
    plt.legend()
    plt.show(block=False)
    
    return disp_cent_lst, u_disp_lst

