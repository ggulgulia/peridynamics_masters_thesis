from __future__ import print_function
from fenics import *
from peridynamic_neighbor_data import *
from peridynamic_materials import *
import matplotlib.pyplot as plt

def get_displaced_cell_centroids(m, u_fe, cell_cent):
    """
    returns the displaced cell centroid of the mesh
    after FE solution
    input:
    ------
        m: original FEniCS Mesh
        u_fe: FE solution from FEniCS
    output:
    -------
        disp_cent: centroid of displaced triangualtions (in the mesh)
        u_disp: displacement vectors of each triangle 
    """

    dim = np.shape(cell_cent)[1]
    u_disp = np.zeros((len(cell_cent), dim), dtype=float)

    for i, cell in enumerate(cell_cent):
        u_disp[i] = u_fe(cell_cent[i]) #this is expensive 

    disp_cent = cell_cent + u_disp

    return disp_cent, u_disp


def solve_fenic_bar(mesh, cell_cent, npts=15, material='steel', plot_ = False, force=-5e8):
    """
    solves the case for a 2D steel plate loaded statically under various loads

    input:
    ------
        mesh : FEniCS mesh
        npts : number of discretization points for 2D plate with circular hole
        material: 
        plot_: boolean for showing plots of FE solution
        force: value of force we want to apply

    output:
    -------
        disp_cent : centroids of elements after FE solution
        u_disp    : displacement of each centroid
    """
    L = 3.
    H = 1.
    
#   mesh = rectangle_mesh_with_hole(npts=npts)
    #mesh = rectangle_mesh(Point(0,0), Point(3,1), numptsX=20, numptsY=10)

    
    def eps(v):
        return sym(grad(v))
    
    def sigma(v):
        return bulk*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)
    
    class LeftEdge(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and abs(x[0]) < FENICS_EPS*1e3)
    
    class RightEdge(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[0] - L) < FENICS_EPS*1e3
    
    class BottomEdge(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]) < FENICS_EPS*1e3
    
    class TopEdge(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[1] - H) < FENICS_EPS*1e3
    
    ## separate edges
    left_edge   = LeftEdge()
    right_edge  = RightEdge()
    bottom_edge = BottomEdge()
    top_edge    = TopEdge()
    
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)
    
    right_edge.mark(sub_domains, 5)
    left_edge.mark(sub_domains, 2)
    bottom_edge.mark(sub_domains, 3)
    top_edge.mark(sub_domains, 4)
    
    ds = Measure("ds")(subdomain_data=sub_domains)
        
    # Material Constants
    E, nu, rho, mu, bulk, gamma = get_steel_properties(dim=2)
    ## Variational Formulation 
    
    #################################
    #incase we wish to apply a body force
    rho_g = 9e-3
    f = Constant((0.0,-rho_g))
    #################################
    V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
    u = TrialFunction(V)
    d = u.geometric_dimension()
    v = TestFunction(V)
    a = inner(sigma(u), eps(v))*dx
    #l = inner(f, v)*dx  
    
    #Neumann Boundary condition for traction force
    g = inner(Constant((0,force)),v) #
    l = g*ds(5)
        
    #Applying bc and solving
    #bc = DirichletBC(V.sub(0), Constant(0.), left_edge)
    bc = DirichletBC(V, Constant((0., 0.)), left_edge)
    u_fe = Function(V, name="Displacement")
    solve(a == l, u_fe, bc)
    
    disp_cent, u_disp = get_displaced_cell_centroids(mesh, u_fe, cell_cent) 
    
    if plot_ is True:
        plt.figure()
        plot(mesh)
        plot(20*u_fe, mode="displacement")
        plt.xlim(-0.5,3.5)
        plt.ylim(-0.6,1.5)
        plt.show(block=False)

    return disp_cent, u_disp
