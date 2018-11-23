from fenics import *
import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as la
from math import factorial as fact
from evtk.hl import pointsToVTK

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

def plot_peridym_mesh(mesh):
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
        ax.scatter(x,y,z, s=70, marker='o', color='c', alpha=1.0, edgecolors='face')
        ax.axis('off')
    
    if dim == 2 : 
        x,y = cell_cent.T
        ax = fig.add_subplot(111)
        ax.scatter(x,y, s=50, marker='o', color='c', alpha=1.0, edgecolors='face')
        ax.axis=('off')

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

    mesh = RectangleMesh(point1, point2, numptsX, numptsY)
    print_mesh_stats(mesh)
    
    return mesh

def rectangle_mesh_with_hole(point1=Point(0,0), point2=Point(2,1), hole_cent=Point(1,0.5), 
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
    
    return corner_min, corner_max

def get_peridym_mesh_bounds(mesh):
    """
    returns lists of elements and centroids of corresponding elements
    that lie in the peridynamic boundary
    
    Note: For nD mesh, we return (n-1)D info, D = dim = {2,3}

    
    
    input:
    -----
        mesh : 2D-tri/3D-tet mesh generated by fenics
    output:
    ------
        ##see comments at the return statement
    returns: TODO

    """
    cell_cent = get_cell_centroids(mesh)
    dim = len(cell_cent[0])

    corner_min, corner_max = get_domain_bounding_box(mesh)
    num_els = mesh.num_cells()

    max_edge_len = mesh.hmax()
    range_fact = 1.25*max_edge_len

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
