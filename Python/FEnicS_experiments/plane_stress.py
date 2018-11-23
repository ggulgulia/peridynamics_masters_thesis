from __future__ import print_function
from fenics import *
import mshr
import matplotlib.pyplot as plt

L = 2.
H = 1.
Nx = 25
Ny = 10

mesh = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny)
#Router = mshr.Rectangle(Point(0,0), Point(L,H))
#Rinner = mshr.Circle(Point(0+0.5*L, 0+0.5*H), 0.25*H)
#domain = Router - Rinner
#mesh = mshr.generate_mesh(domain, 15)

#plt.figure()
#plot(mesh)

def eps(v):
    return sym(grad(v))

def left(x, on_boundary):
    return on_boundary and abs(x[0]) < FENICS_EPS*1e3

def right(x, on_boundary):
    return on_boundary and abs(x[0] - L) < FENICS_EPS*1e3
    
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

right_edge.mark(sub_domains, 1)
left_edge.mark(sub_domains, 2)
bottom_edge.mark(sub_domains, 3)
top_edge.mark(sub_domains, 4)

ds = Measure("ds")[sub_domains]
    
# Material Constants
E = Constant(200e9); nu = Constant(0.3)
model = "plane_stress"
mu = E/2/(1+nu)
bulk = E*nu/(1+nu)/(1-2*nu)

if model == "plane_stress":
    bulk = 2*mu*bulk/(bulk+2*mu)


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
g = inner(Constant((100000000,0)),v) #
l = g*ds(1)
    
#Applying bc and solving
bc = DirichletBC(V.sub(0), Constant(0.), left)
u = Function(V, name="Displacement")
solve(a == l, u, bc)

plt.figure()
plot(1e4*u, mode="displacement")

plt.show(block=False)

# Plot stress
#s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
#von_Mises = sqrt(3./2*inner(s, s))
#V = FunctionSpace(mesh, 'P', 1)
#von_Mises = project(von_Mises, V)
#plt.figure()
#plot(von_Mises, title='Stress intensity')
#plt.show(block=False)

# Compute magnitude of displacement
#u_magnitude = sqrt(dot(u, u))
#u_magnitude = project(u_magnitude, V)
#plot(u_magnitude, 'Displacement magnitude')
#print('min/max u:',
#      u_magnitude.vector().get_local().min(),
#      u_magnitude.vector().get_local().max())
