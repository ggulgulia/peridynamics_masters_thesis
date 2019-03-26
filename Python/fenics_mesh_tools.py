from fenics import *
import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import timeit as tm
import numpy.linalg as la
from math import factorial as fact
from evtk.hl import pointsToVTK
import copy as cpy

def plot_fenics_mesh(mesh, new_fig=True):
    """
    plots the fenics mesh as it is

    input:
    -----
        mesh: 2D-tri/3D-tet mesh
    output:
    ------
        plots the mesh
    """
    if(new_fig):
        plt.figure()

    plot(mesh)
    #plt.title("FEniCS mesh")
    plt.show(block=False)

    pass

def plot_peridym_mesh(mesh=None, cell_cent=None, disp_cent=None, annotate=False):
    """
    plots the mesh/centroids of mesh as is expected in peridynamics
    
    either mesh or cell_cent is to be provided by user
    neither provinding mesh nor providing cell_cent is 
    wrong
    input:
    ------
        mesh: 2D-tri/3D-tet mesh from FEniCS
        cell_cent: particle positions in euclidean space
        disp_cent: displaced centroids after some load condition
        annotate: wether we wish to annotate the particle positions
    output:
    -------
        plots the centroids of tri/tets in FEniCS mesh

    """

    if mesh == None and (cell_cent==None).any():
        raise AssertionError("provide either fenics mesh or cell centroid of PD particles")
    if mesh != None and (cell_cent==None).any():
        cell_cent = get_cell_centroids(mesh)
    if cell_cent.any() !=None and mesh == None:
        pass
    
    ## we wish to scale the axis accordign to geometry
    extents = get_domain_bounding_box(mesh=mesh, cell_cent=cell_cent)
    dim = len(cell_cent[0])
    x_min = extents[0][0]; x_max = extents[1][0]
    y_min = extents[0][1]; y_max = extents[1][1]

    if y_min/x_min <0.8:
        fact = 0.3
    if y_min/x_min <0.5:
        fact = 2
    if y_min/x_min <0.4:
        fact = 3
    x=None; y=None; z=None
    fig = plt.figure()
    if dim == 3:
        z_min = corners[0][2]; z_max = corners[1][2]
        x,y,z = cell_cent.T
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z, s=70, marker='o', color='b', alpha=1.0, edgecolors='face')
        ax.axis('off')
    
    if dim == 2 : 
        ax = fig.add_subplot(111)
        x,y = cell_cent.T
        plt.scatter(x,y, s=300, color='c', marker='o', alpha=0.8)
        plt.xlim(x_min -0.1*x_min, x_max + 0.1*x_max)
        plt.ylim(y_min -0.1*y_min, y_max+0.1*y_max)
        plt.xlim(x_min  + 0.3*x_min, x_max + 0.3*x_max)
        plt.ylim(y_min  + fact*y_min, y_max + fact*y_max)
        plt.axis=('off')

    if annotate==True:
        for idx, cc in enumerate(cell_cent):
            plt.text(cc[0], cc[1],  str(idx), color='k', verticalalignment='bottom', horizontalalignment='right', fontsize='medium')

    ax.set_aspect('equal')
    plt.title("peridynamics mesh")
    plt.show(block=False)

def get_displaced_soln(cell_cent, u_disp, horizon, dim, data_dir=None, plot_=False, save_fig=False, zoom=40):
    """
    plots the displaced cell centroids after a solution 
    step. Additionally retrns the final cell centroid
    after additon of displacement field in the orginal
    configuration

    input:
    ------
    cell_cent: np.array of cell centroids 
    u_disp   : np.array of displacement field 
    zoom     : magnification desired in plot 
    
    output:
    --------
    disp_cent : final configuration after displacement

    """
    disp_cent = cell_cent + u_disp
    
    dpi = 3
    legend_size = {'size': str(6*dpi)}
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(111)
        x, y = cell_cent.T
        #plt.scatter(x,y, s=300, color='r', marker='o', alpha=0.1, label='original config')
        x,y = (cell_cent + zoom*u_disp).T 
        plt.scatter(x,y, s=150, color='b', marker='o', alpha=0.6, label=r'$\delta$ = '+str(horizon))
        plt.legend(prop=legend_size)
        plt.ylim(-0.5, 1.5)
        plt.xlim(-0.5, 2.5)

    if dim == 3:
        from mpl_toolkits.mplot3d import Axes3D 
        x, y, z = cell_cent.T
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d') 
        ax.scatter(x,y,z, s=150, color='r', marker='o', alpha=0.1, label='original config')
        x,y,z = (cell_cent + zoom*u_disp)

        ax.scatter(x,y,z,s=150, color='g', marker='o', alpha=1.0, label='deformed config')
        ax.axis('off')
        plt.legend()

    ax.set_aspect('equal')

    if plot_:
        plt.show(block=False)

    if save_fig:
        plt.savefig(data_dir)
        plt.close(fig)

    return disp_cent


def print_mesh_stats(mesh):
    """
    this function prints the mesh stats like:
    num cells, num vertices, max edge length, min edge length

    :mesh: TODO
    """
    print("num cells: %i\nnum vertices: %i\nmax edge length: %4.5f\nmin edge length: %4.5f\n \
          "%(mesh.num_cells(), mesh.num_vertices(), mesh.hmax(), mesh.hmin()))
    pass


def rectangle_mesh(point1=Point(0,0), point2=Point(2,1), numptsX=10, numptsY=5):
    """
    generates a triangulated Rectangular domain with a circular hole

    input
    ------
        point1: Point coordinates in 3D corner_min of box
        point2: Point coordinates in 3D corner_max of box
        numptsX: number of discretization points in X-direction
        numptsY: number of discretization points in Y-direction
    output:
    ------
        mesh: 2D FEniCS mesh

    """

    mesh = RectangleMesh(point1, point2, numptsX, numptsY )
    print_mesh_stats(mesh)
    
    return mesh

def rectangle_mesh_with_hole(point1=Point(0,0), point2=Point(3,1), hole_cent=Point(1.5,0.5), 
                                hole_rad=0.25, npts=15):
    """
    generates a triangulated Rectangular domain with a circular hole

    input
    ------
        point1: Point coordinates in 3D corner_min of box
        point2: Point coordinates in 3D corner_max of box
        cyl_cent1: Point coordinates in 3D of center1 of Cylinder
        cyl_cent2: Point coordinates in 3D of center1 of Cylinder
        cyl_rad:   Radius of cylinder
        npts:      number of discretization points
    output:
    ------
        mesh: 2D FEniCS mesh with circular hole

    """

    Router = mshr.Rectangle(point1, point2)
    Rinner = mshr.Circle(hole_cent, hole_rad)
    domain = Router - Rinner

    mesh = mshr.generate_mesh(domain, npts)
    print_mesh_stats(mesh)
    
    return mesh

def tensile_test_bar(numpts = 60, plot_=True):

    outer = mshr.Rectangle(Point(-100, -10), Point(100, 10))
    barLo = mshr.Rectangle(Point(-30, 6.25), Point(30, 10))
    barHi = mshr.Rectangle(Point(-30, -10), Point(30, -6.25))
    c1    = mshr.Circle(Point(-30, -19.5), 13.4)
    c2    = mshr.Circle(Point(30, -19.5), 13.4)
    c3    = mshr.Circle(Point(-30, 19.5), 13.4)
    c4    = mshr.Circle(Point(30, 19.5), 13.4)
    domain = outer - barLo - barHi - c1 - c2 - c3 -c4
    
    mm = mshr.generate_mesh(domain, numpts)

    if plot_:
        plt.figure()
        plot(mm, color='k')
        plt.xlim(-120, 120)
        plt.ylim(-20, 20)
        plt.show(block=False)
    return  mm

def structured_cell_centroids(mesh):
    """
    creates a structured cell centroids and cell volumes of 
    square or cubic lattice using the fenics mesh by averaging
    appropriate number of  2D/3D triangles 

    input:
    ------
        mesh : 2D/3D fenics mesh, not crossed topology
    output:
    ------
        cell_cents_struct : np.array of cell centroids
                            for corresponding structured
                            mesh
    """
    dim = mesh.topology().dim()
    stride = fact(dim)
    cents = get_cell_centroids(mesh)
    num_cells = int(mesh.num_cells()/stride)
    cell_cents_struct = np.zeros((num_cells,dim),dtype=float)

    for i in range(num_cells):
        start = int(stride*i)
        end   = int(stride*i)+stride
        cell_cents_struct[i] = np.average(cents[start:end],axis=0)

    return cell_cents_struct

def structured_cell_volumes(mesh):
    """
    returns an array of cell volumes of structured grid created from
    fenics triangular grid 

    input:
    ------
        mesh : 2D FEniCS mesh, not corssed topology
    """
    dim = mesh.topology().dim()
    stride = fact(dim)
    vols = get_cell_volumes(mesh)
    num_cells = int(mesh.num_cells()/stride)
    
    return np.ones(num_cells, dtype=float)*stride*vols[0]


def box_mesh(point1=Point(0,0,0), point2=Point(2,1,1),
             numptsX=8, numptsY=4, numptsZ=4):
    """
    generates a 3D box mesh with tetrahedral elements with a cylindrical hole in it

    input
    ------
        point1: Point coordinates in 3D corner_min of box
        point2: Point coordinates in 3D corner_max of box
        cyl_cent1: Point coordinates in 3D of center1 of Cylinder
        cyl_cent2: Point coordinates in 3D of center1 of Cylinder
        cyl_rad:   Radius of cylinder
    output:
    ------
        mesh: 3D FEniCS mesh

    """
    mesh = BoxMesh(point1, point2, numptsX, numptsY, numptsZ)
    print_mesh_stats(mesh)

    return mesh 


def box_mesh_with_hole(point1=Point(0,0,0), point2=Point(2,1,1), cyl_cent1 = Point(1, -10, 0.5), 
                      cyl_cent2= Point(1, 10, 0.5), cyl_rad=0.25, numpts=15):
    """
    generates a 3D box mesh with tetrahedral elements with a cylindrical hole in it

    input
    ------
        point1: Point coordinates in 3D corner_min of box
        point2: Point coordinates in 3D corner_max of box
        cyl_cent1: Point coordinates in 3D of center1 of Cylinder
        cyl_cent2: Point coordinates in 3D of center1 of Cylinder
        cyl_rad:   Radius of cylinder
        npts: Number of discretization points
    output:
    ------
        mesh: 3D FEniCS mesh with a cylindrical hole

    """
    Router = mshr.Box(point1, point2)
    Rinner = mshr.Cylinder(cyl_cent1, cyl_cent2, cyl_rad, cyl_rad)
    domain = Router - Rinner

    mesh = mshr.generate_mesh(domain, numpts)
    print_mesh_stats(mesh)
    
    return mesh


def get_cell_centroid2(cents, extents):
    """
    returns the cell centroids lying within given
    geometric extents

    input:
    ------
        cell_cents: np.array of cell centroids
        extents   : np.array((2,dim)) of bounding 
                    box for sub domains
    output:
    -------
        cents_in_extents 
    """
    cells_in_ee = np.empty(0,int)
    for i in range(len(cents)):
        c = cents[i]
        if( (c > extents[0]).all() and (c <= extents[1]).all() ):
            cells_in_ee = np.append(cells_in_ee, [i], axis=0)

    return cells_in_ee


def get_cell_centroids(mesh):
    """
    given a fenics mesh/mshr.mesh as argument, returns
    the centroid of the cells in the mesh

    input
    -----
        mesh: 2D tri/3D-tet mesh generated by fenics
    output
    ------
        cell_cent: np.array having cell centroid for each cell in mesh 
    """
    num_els = mesh.num_cells()
    coords = mesh.coordinates()
    cells = mesh.cells()
    dim = len(coords[0])

    cell_cent = np.zeros((num_els, dim), dtype=float, order='c')

    for i in range(num_els):
        pts = [coords[idx] for idx in cells[i]]
        cell_cent[i] = (1/(dim+1))*sum(pts) #this works only for 2D/3D triangles

    return cell_cent

def get_cell_volumes(mesh):
    """
    given a fenics/mshr mesh as argument, this function 
    returns the area of each cell in the mesh

    input
    -----
        mesh: 2D-tri/3D-tet mesh generated by fenics
        works also for a n-dimensional tetrahedron B-)

    output
    ------
        cell_volume: np.array of doubles, each entry having volume of each cell in mesh
                     NOTE: in 2D volume = area, and in 4D and more, a hpyervolume
                

    """
    num_els = mesh.num_cells()
    coords = mesh.coordinates()
    cells = mesh.cells()
    dim = len(coords[0])

    cell_volume = np.zeros(num_els, dtype=float)
    div_fact = 1.0/float(fact(dim)) #division factor for n-dim tetrahderon
    
    for i in range(num_els):
        cell_volume[i] = abs(la.det(np.insert(coords[cells[i]], dim, 1, axis=1)))
    
    return div_fact*cell_volume

def get_domain_bounding_box(mesh=None, cell_cent=None):
    """
    given a fenics mesh, this function returns the bounding_box that fits around the domain

    input:
    -----
        mesh : 2D-tri/3D-tet mesh generated in fenics/mshr
    output:
    ------
        corner_min: np.ndim array of corner having minima in space
        corner_max: np.ndim array of corenr having maxima in sapce

    """
    if  mesh==None and (cell_cent == None).any():
        raise AssertionError("provide either fenics mesh or cell centroid of PD particles")
    if  (cell_cent == None).any():
        coords = mesh.coordinates()
    if cell_cent.all() and not mesh:
        coords = cell_cent
    
    dim = len(coords[0])

    corner_min = np.zeros(dim ,float)
    corner_max = np.zeros(dim, float)

    for d in range(dim):
        corner_min[d] = min(coords[:,d])
        corner_max[d] = max(coords[:,d])
    
    return np.vstack((corner_min, corner_max))

def get_deformed_mesh_domain_bbox(cell_cent, dim):

    corner_min = np.zeros(dim ,float)
    corner_max = np.zeros(dim, float)
    for d in range(dim):
        corner_min[d] = min(cell_cent[:,d])
        corner_max[d] = max(cell_cent[:,d])

    return np.vstack((corner_min, corner_max))

def get_peridym_mesh_bounds(mesh, struct_grd=False):
    """
    returns lists of elements and centroids of corresponding elements
    that lie in the peridynamic boundary
    
    Note: For nD mesh, we return (n-1)D info, D = dim = {2,3}

    
    
    input:
    -----
        mesh : 2D-tri/3D-tet mesh generated by fenics
        struct_grd: boolean
    output:
    ------
        ##see comments at the return statement
    returns: TODO

    """
    if(struct_grd):
        cell_cent = structured_cell_centroids(mesh)
        max_edge_len = np.diff(cell_cent[0:2][:,0])
        range_fact   = 2.001*max_edge_len 
    else:
        cell_cent = get_cell_centroids(mesh)
        max_edge_len = mesh.hmax()
        range_fact = 1.5001*max_edge_len

    dim = len(cell_cent[0])
    corner_min, corner_max = get_domain_bounding_box(mesh)
    num_els = len(cell_cent)

    bound_range = np.zeros(2*dim, dtype=float)
    bound_nodes = {} #dict to store the node numbers of centroids that lie within bound_range
    bound_cents  = {} #dict to store the node centroids corresponding to node numbers above

    for d in range(dim):
        """
        index to direction along which the normal to boundary occurs:#
        0 - x_min
        1 - x_max
        2 - y_min
        3 : y_max
        4 : z_min
        5 : z_max
        Note: z-normal not applicable to 2d problems
        """
        bound_range[2*d]    = corner_min[d] + range_fact #min bound for d
        bound_range[2*d +1] = corner_max[d] - range_fact #max bound for d
        bound_nodes[(2*d)]   = np.where(cell_cent[:,d] <= bound_range[2*d]) #node nums for min bound
        bound_nodes[(2*d+1)] = np.where(cell_cent[:,d] >= bound_range[2*d+1]) # node nums for max bound

        bound_cents[(2*d)]   = cell_cent[bound_nodes[2*d][0]] #node centroids for min bound
        bound_cents[(2*d+1)] = cell_cent[bound_nodes[2*d+1][0]] #node centroids for min bound

    return bound_nodes, bound_cents #convert list to np array 

def get_peridym_edge_length(cell_cent, struct_grd=False):
    """
    given a set of cell centroid beloning to regular (Square/Tri)
    discretization in 2D/3D, the method returns the edge length

    NOTE: el NOT EQUAL TO centroid distance

    input:
    ------
    cell_cent : nd-array of peridynamic cell centroids 
    struct_grd: boolean, whether the grid is struct(square lattices) unstruct(triangular lattices)

    output:
    -------
    el       : nd array of edge length

    """
    dim = len(cell_cent[0])
    el = np.zeros(dim, dtype = float)

    if(struct_grd):
        el_fact = 1.0
    else:
        el_fact = 3.0

    for d in range(dim):
        xx = np.unique(cell_cent[:,d])
        el[d] = el_fact*np.max(np.abs(np.diff(xx[0:2])))

    return el

def get_modified_boundary_layers(cell_cent, el, num_lyrs, struct_grd):
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
        factor = 1
        correction = 0
    else:
        factor = 2
        correction = 1

    lyrs = float(num_lyrs-1)+ 0.001
    
    for d in range(dim):
        bound_range[2*d] = factor*np.min(cell_cent[:,d]) + lyrs*el[d]
        bound_range[2*d+1] = np.max(cell_cent[:,d]) -lyrs*el[d] - el[d]/3*correction

        bound_nodes[2*d] = np.where(cell_cent[:,d] <= bound_range[2*d])
        bound_nodes[(2*d+1)] = np.where(cell_cent[:,d] >= bound_range[2*d+1])

        bound_cents[2*d]   = cell_cent[bound_nodes[2*d][0]]
        bound_cents[2*d+1] = cell_cent[bound_nodes[2*d+1][0]]

    return bound_nodes, bound_cents

def compute_modified_extents(cell_cent, el, struct_grd=False):
    """
    computes the extents of the new mesh after the addition of 
    ghost layers of centroids 

    NOTE: if the cell centroids are not modified
          then this returns the extents same as 
          method get_domain_bounding_box
    :cell_cent: TODO
    :bc_loc: TODO
    :el: TODO
    :returns: TODO

    """
    dim = len(cell_cent[0])

    extents = np.zeros((2, dim), float)
    min_corners = np.zeros(dim, float)
    max_corners = np.zeros(dim, float)

    if(struct_grd):
        shift_fact = 0.5
    else:
        shift_fact = 1.0/3.0

    for d in range(dim):
        min_corners[d] = np.min(cell_cent[:,d])
        max_corners[d] = np.max(cell_cent[:,d])

        """
        below is done to avoid round-off error due to
        substraction of two numbers near to each other
        
        This occurs when corners in one of the dimension
        remains unchanged but we still try to compute 
        the new extents
        """
        extents[0][d] = round(min_corners[d] - shift_fact*el[d], 16)
        extents[1][d] = round(max_corners[d] + shift_fact*el[d], 16)

    return extents


def add_ghost_cells(mesh, bc_loc, num_lyrs,struct_grd=False):
    """
    this method adds ghost layer to the mesh 
    along the edges where bc is intended to be applied 
    so that equivalent of peridynamic volume boundary
    condition is on the edge/bounary of the domain 

    bc_loc = [0, 1, 2, 3, 4, 5] (see methd: get_peridym_mesh_bounds
    for more details)
    input:
    ------
        mesh: FEniCS mesh 
        bc_loc: array/list (MUST BE SORTED) of integers specifiying the boundary 
                      locations (see method: get_peridym_mesh_bounds)
    output:
    -------
        cell_ids_ghost : np.array new cell ids  
        cell_cent_ghost: np.array new cell centroids 
        cell_vol_ghost : np.array new cell volume


extended domain (in '.' around '- & ¦') 
    .................................
    .  .                         .  .
    ...----------3:y_max----------... 
    .  ¦                         ¦  .
    .  ¦                         ¦  . 
    .  ¦                         ¦  .
    .0:x_min                 1:x_max. 
    .  ¦                         ¦  .
    .  ¦                         ¦  .
    ...¦---------2:y_max---------¦...
    .  .                         .  .
    .................................

            all even boundary indices refer to min along the
            corresponding cardinal direction {0:x_min, 2:y_min, 4:z_min}
            and similarly all odd indices refer to max along the
            cardinal direction

            The edges belonging to min has to be extended in -ve
            cardinal direction and those beloning to max has to be 
            extended along +ve cardinal direction

    """

    dim = mesh.topology().dim()
    dim_lst = [dd for dd in range(dim)]

    if(struct_grd):
        cell_cent = structured_cell_centroids(mesh)
        cell_vol  = structured_cell_volumes(mesh)
        dist_fact = 1.0 
        mul = 1 #for struct grid
    else:
        cell_cent = get_cell_centroids(mesh)
        cell_vol  = get_cell_volumes(mesh)
        mul = 2 #need this for FEniCS regular triangulations

    el = get_peridym_edge_length(cell_cent, struct_grd)
    new_cell_cents = cpy.deepcopy(cell_cent)

    for loc in bc_loc:
        for d in range(dim):
            #store max of distance along successive layer
            # as the hpyothetical edge length
            idxs = [dd for dd in range(dim)]
            idxs.pop(d)

            if(2*d == loc):
                _, bound_cents_d = get_modified_boundary_layers(new_cell_cents, el, num_lyrs,struct_grd)
                curr_cents = bound_cents_d[loc]
                coord = np.sort(np.unique(curr_cents[:,d]))[::-1]
                #find out the points where min along d dim occurs
                for ll in range(mul*num_lyrs):
                    temp_min_loc = curr_cents[np.ravel(np.argwhere(curr_cents[:,d] == coord[ll]))]
                    new_min_cents = cpy.deepcopy(temp_min_loc)
                    new_min_cents[:,d] -= num_lyrs*el[d]

                    #insert the new centroids to maintain the order
                    #of array of centroids
                    if(loc==0): #x-axis arrangement
                        for i, yy in enumerate(new_min_cents[:,idxs]):
                            idx = np.min(np.where(np.all(new_cell_cents[:,idxs] == yy, axis=1)))
                            new_cell_cents = np.insert(new_cell_cents, idx, new_min_cents[i], 0)
                    if(loc==2): #y-axis arrangement
                        new_cell_cents = np.vstack((new_min_cents, new_cell_cents))

            if(2*d+1 == loc):
                _, bound_cents_d = get_modified_boundary_layers(new_cell_cents, el, num_lyrs, struct_grd)
                curr_cents = bound_cents_d[loc]
                coord = np.sort(np.unique(curr_cents[:,d]))
                #find out the points where max along d dim occurs
                for ll in range(mul*num_lyrs):
                    temp_max_loc = curr_cents[np.ravel(np.where(curr_cents[:,d] == coord[ll]))]

                    new_max_cents = cpy.deepcopy(temp_max_loc)
                    new_max_cents[:,d] += num_lyrs*el[d]

                    #insert the new centroids to maintain the order
                    #of array of centroids
                    if(loc==1): #x-axis arrangement
                        for i, yy in enumerate(new_max_cents[:,idxs]):
                            idx = np.max(np.where(np.all(new_cell_cents[:,idxs] == yy, axis=1)))
                            new_cell_cents = np.insert(new_cell_cents, idx+1, new_max_cents[i],0)
                    if(loc==3): #y-axis arrangement
                        new_cell_cents = np.vstack((new_cell_cents, new_max_cents))
                
    new_cell_vols = np.ones(len(new_cell_cents), dtype=float)*cell_vol[0]

    return new_cell_cents, new_cell_vols


def write_to_vtk(mesh,  displacement=None, file_name="gridfile"):
    """
    writes the peridynamic mesh coordinates to a vetk supported file format
    """
    cents = get_cell_centroids(mesh)
    dim = len(cents[0])
    
    if displacement is not None:
        cents+= displacement
    
    file_name = "./"+file_name
    
    write_function=None
    if dim==3:
        write_function = write_to_vtk3D
    if dim==2:
        write_function = write_to_vtk2D
   
    write_function(cents, displacement, file_name)


    pass

def write_to_vtk3D(cents, displacement, file_name):
    """
    writes 3D data to vtk file 
    """

    x,y,z = cents.T
    x = np.array(x, order='c')
    y = np.array(y, order='c')
    z = np.array(z, order='c')

    if displacement is None:
        pointsToVTK(file_name, x, y, z, data={"x":x, "y":y, "z":z})

    else:
        dispX, dispY, dispZ = displacement.T
        dispX = np.array(dispX, order='c')
        dispY = np.array(dispY, order='c')
        dispZ = np.array(dispZ, order='c')
        
        pointsToVTK(file_name, x, y, z, data={"x":x, "y":y, "z":z, 
                    "dispX":dispX, "dispY":dispY, "dispZ":dispZ})

    pass

def write_to_vtk2D(cents, displacement, file_name):
    """
    writes 2D data to VTK 
    """

    x,y = cents.T
    x = np.copy(x, order='c')
    y = np.copy(y, order='c')
    z = np.zeros(len(x), order='c')

    if displacement is None:
        pointsToVTK(file_name, x, y, z, data={"x":x, "y":y})

    else:
        dispX, dispY  = displacement.T
        dispX = np.array(dispX, order='c')
        dispY = np.array(dispY, order='c')
        
        pointsToVTK(file_name, x, y, z, data={"x":x, "y":y, 
                    "dispX":dispX, "dispY":dispY})

    pass
