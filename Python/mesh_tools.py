import meshpy.triangle as tri
import numpy as np
import meshpy.geometry as geo
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from six.moves import range
import meshpy.geometry as geo
from meshpy.tools import uniform_refine_triangles

def plot_(mesh, interactive=False):
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
    plt.show(block=interactive)

def add_points(points, segments):
    new_points = []
    for i in range(len(points)):
        point = points[i]
        new_pnts = np.linspace(point[0], point[1], segments+1)

        for i in range(segments):
            new_points.append((new_pnts[i], new_pnts[i+1]))
    
    return new_points

def needs_refinement(vertices, area):
    vert_origin, vert_destination, vert_apex = vertices
    bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
    bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

    dist_center = math.sqrt(bary_x**2 + bary_y**2)
    max_area = 100*(math.fabs(0.002 * (dist_center-0.5)) + 0.0001)
    return area > max_area


def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i+1))
    result.append((end, start))
    return result



def rectangle_mesh(point1=(0,0), point2 = (1,1), subdiv=(5,5)):

    points, facets, _, _ = geo.make_box(point1, point2, subdivisions=subdiv)
    builder = geo.GeometryBuilder()
    mp = geo.Marker.FIRST_USER_MARKER
    builder.add_geometry(points = points, facets=facets, facet_markers = mp)
    mi = tri.MeshInfo()
    builder.set(mi)
    mesh = tri.build(mi, max_volume=1e-1, min_angle=20)
    
    pold = np.array(mesh.points).tolist()
    elemsold = np.array(mesh.elements).tolist()
    return mesh

def unit_square_mesh(subdiv=(10,10)):
    return rectangle_mesh((0.0,0.0),(0.0,0.0), subdiv)


def get_mesh_points(mesh):
    return np.array(mesh.points)

def get_mesh_elements(mesh):
    return np.array(mesh.elements)

def get_elem_centroid(mesh):
    elems = get_mesh_elements(mesh)
    points = get_mesh_points(mesh)
    elems_centroid = []

    for a, b,c in elems:
        [a_pt,b_pt,c_pt] = [points[idx] for idx in [a,b,c]]
        loc_elem_cent = [0.0, 0.0]

        loc_elem_cent[0] = (a_pt[0] + b_pt[0] + c_pt[0])/3
        loc_elem_cent[1] = (a_pt[1] + b_pt[1] + c_pt[1])/3

        elems_centroid.append(loc_elem_cent)
    
    return elems_centroid

def get_elem_areas(mesh):
    element_areas = []
    elems = get_mesh_elements(mesh)
    points = get_mesh_points(mesh)

    for a, b, c in elems:
        a_pt, b_pt, c_pt = [points[idx] for idx in [a,b,c]]

        loc_matrix = np.column_stack([[a_pt[0], a_pt[1], 1.0], [b_pt[0], b_pt[1], 1.0],[c_pt[0], c_pt[1], 1.0] ])
        loc_area = 0.5*abs(la.det(loc_matrix))
        element_areas.append(loc_area)

    return element_areas
