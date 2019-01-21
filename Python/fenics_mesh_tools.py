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

def plot_peridym_mesh(mesh, disp_cent=None, annotate=False):
    """
    plots the mesh/centroids of mesh as is expected in peridynamics

    input:
    ------
        mesh: 2D-tri/3D-tet mesh from FEniCS
    output:
    -------
        plots the centroids of tri/tets in FEniCS mesh

    """

    cell_cent = get_cell_centroids(mesh)
    dim = len(cell_cent[0])
    x=None; y=None; z=None
    fig = plt.figure()
    if dim == 3:
        x,y,z = cell_cent.T
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z, s=70, marker='o', color='b', alpha=1.0, edgecolors='face')
        ax.axis('off')
    
    if dim == 2 : 
        x,y = cell_cent.T
        plt.scatter(x,y, s=300, color='c', marker='o', alpha=0.8)
        plt.axis=('off')

    if annotate==True:
        for idx, cc in enumerate(cell_cent):
            plt.text(cc[0], cc[1],  str(idx), color='k', verticalalignment='bottom', horizontalalignment='right', fontsize='medium')


    plt.title("peridynamics mesh")
    plt.show(block=False)


def print_mesh_stats(mesh):
    """
    this function prints the mesh stats like:
    num cells, num vertices, max edge length, min edge length

    :mesh: TODO
    :returns: TODO

    """
    print("num cells: %i\nnum vertices: %i\nmax edge length: %4.5f\nmin edge length: %4.5f\n \
          "%(mesh.num_cells(), mesh.num_vertices(), mesh.hmax(), mesh.hmin()))
    pass

def uniform_square_mesh(point1=(0.0, 0.0), point2=(2.0, 1.0), nptsX=20, nptsY=10):
    """
    returns a domain descricitized with square mesh 
    """

    dx = (point2[0] - point1[0])/(nptsX+1)
    dy = (point2[1] - point1[1])/(nptsY+1)
    cell_volume = dx*dy
    #mesh = {}

    #mesh['volum'] = volume
    #mesh['centroid'] = centroids 

    #return mesh

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

def get_domain_bounding_box(mesh):
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
    coords = mesh.coordinates()
    dim = len(coords[0])

    corner_min = np.zeros(dim ,float)
    corner_max = np.zeros(dim, float)

    for d in range(dim):
        corner_min[d] = min(coords[:,d])
        corner_max[d] = max(coords[:,d])
    
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
        bound_cents[(2*d+1)]   = cell_cent[bound_nodes[2*d+1][0]] #node centroids for min bound

    return bound_nodes, bound_cents #convert list to np array 

def add_ghost_cells(mesh, bc_loc, struct_grd=False):
    """
    this method adds ghost layer to the mesh 
    along the edges where bc is intended to be applied 
    so that equivalent of peridynamic volume boundary
    condition is on the edge/bounary of the domain 

    input:
    ------
        mesh: FEniCS mesh 
        bc_locations: array/list of integers specifiying the boundary 
                      locations (see method: get_peridym_mesh_bounds)
    output:
    -------
        cell_ids_ghost : np.array new cell ids  
        cell_cent_ghost: np.array new cell centroids 
        cell_vol_ghost : np.array new cell volume

extend 
    ...----------3:y_max----------... 
    .  ¦                         ¦  .
    .  ¦                         ¦  . 
    .  ¦                         ¦  .
    .0:x_min                 1:x_max. 
    .  ¦                         ¦  .
    .  ¦                         ¦  .
    ...¦---------2:y_max---------¦...

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
    el  = np.zeros(dim, dtype=float)

    if(struct_grd):
        cell_cent = structured_cell_centroids(mesh)
        cell_vol  = structured_cell_volumes(mesh)
    else:
        cell_cent = get_cell_centroids(mesh)
        cell_vol  = get_cell_volumes(mesh)
    
    new_cell_cents = cpy.deepcopy(cell_cent)
    bound_nodes, bound_cents = get_peridym_mesh_bounds(mesh, struct_grd)
    bound_keys = bound_nodes.keys()

    for loc in bc_loc:
        curr_nodes = bound_nodes[loc]
        curr_cents = bound_cents[loc]
        
        for d in range(dim):
            #store max of distance along successive layer
            # as the hpyothetical edge length
            idxs = [dd for dd in range(dim)]
            idxs.pop(d)
            el[d] = np.max(np.diff(np.unique(curr_cents[:,d])))

            if(2*d == loc):
                #find out the points where min along d dim occurs
                temp_min_loc = curr_cents[np.ravel(np.argwhere(curr_cents[:,d] == np.min(curr_cents)))]
                new_min_cents = cpy.deepcopy(temp_min_loc)
                new_min_cents[:,d] -= el[d]

                #insert the new centroids to maintain the order
                #of array of centroids
                for i, yy in enumerate(new_min_cents[:,idxs]):
                    idx = np.min(np.where(np.all(cell_cent[:,idxs] == yy, axis=1)))
                    new_cell_cents = np.insert(new_cell_cents, idx, new_min_cents[i], 0)

            if(2*d+1 == loc):
                #find out the points where max along d dim occurs
                temp_max_loc = curr_cents[np.ravel(np.where(curr_cents[:,d] == np.max(curr_cents)))]

                new_max_cents = cpy.deepcopy(temp_max_loc)
                new_max_cents[:,d] += el[d]

                #insert the new centroids to maintain the order
                #of array of centroids
                for i, yy in enumerate(new_max_cents[:,idxs]):
                    idx = np.max(np.where(np.all(cell_cent[:,idxs] == yy, axis=1)))
                    new_cell_cents = np.insert(new_cell_cents, idx+1, new_max_cents[i],0)
                
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
