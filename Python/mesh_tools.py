import meshpy.triangle as tri
import numpy as np
import meshpy.geometry as geo
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from six.moves import range

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
    mesh = tri.build(mi, max_volume=1e-1, generate_faces=True, min_angle=30,
            mesh_order=None, generate_neighbor_lists=True)
    
    pold = np.array(mesh.points).tolist()
    elemsold = np.array(mesh.elements).tolist()
    return mesh

def unit_square_mesh(subdiv=(5,5)):
    return rectangle_mesh((0.0,0.0),(1.0,1.0), subdiv)


def get_elem_centroid(mesh):
    """
    TODO
    input : 
        mesh:  meshpy.triangle.MeshInfo
               2D truangular mesh
        
    output:
        elems_centroid : double list
                         list of element centroid x,y cell_ID            
    """
    elems = np.array(mesh.elements) 
    points = np.array(mesh.points)
    elems_centroid = []

    elem_id = -1
    for a, b,c in elems:
        [a_pt,b_pt,c_pt] = [points[idx] for idx in [a,b,c]]
        loc_elem_cent = [0.0, 0.0, elem_id]

        loc_elem_cent[0] = (a_pt[0] + b_pt[0] + c_pt[0])/3
        loc_elem_cent[1] = (a_pt[1] + b_pt[1] + c_pt[1])/3

        elems_centroid.append(loc_elem_cent)
        elem_id +=1
    
    return elems_centroid

def get_elem_areas(mesh):
    element_areas = []
    elems = np.array(mesh.elements) 
    points = np.array(mesh.points)

    for a, b, c in elems:
        a_pt, b_pt, c_pt = [points[idx] for idx in [a,b,c]]

        loc_matrix = np.column_stack([[a_pt[0], a_pt[1], 1.0], [b_pt[0], b_pt[1], 1.0],[c_pt[0], c_pt[1], 1.0] ])
        loc_area = 0.5*abs(la.det(loc_matrix))
        element_areas.append(loc_area)

    return element_areas

def get_edge_lengths(mesh):
    """
    TODO
    input : 
        mesh: meshpy.triangle.MeshInfo
              2D triangular mesh
    
    output:
        edge_lengths : list of doubles
                     List of edge lengths of all the edges (internal and external)
                     in the mesh
    """
    faces = np.array(mesh.faces)
    points = np.array(mesh.points)
    edge_lengths = []
    for a, b in faces:
        a_pt , b_pt = [points[idx] for idx in [a,b]]
        loc_len = math.sqrt((a_pt[0] - b_pt[0])**2 + (a_pt[1] - b_pt[1])**2)
        edge_lengths.append(loc_len)

    return edge_lengths
    
 
def uniform_refine_triangles(mesh, factor=2):
    """
    TODO
    input : 
        mesh:  meshpy.triangle.MeshInfo
               2D triangular mesh
        factor: int
                Factor by which the input mesh has to be refined
    output:
        mesh : meshpy.triangle.MeshInfo()
               Refined mesh 
    """
    points = np.array(mesh.points).tolist()
    elements = np.array(mesh.elements).tolist()
    new_points = points[:]
    new_elements = []
    new_face = []
    old_face_to_new_faces = {}
    face_point_dict = {}

    points_per_edge = factor+1

    def get_refined_face(a, b):
        if a > b:
            a, b = b, a
            flipped = True
        else:
            flipped = False

        try:
            face_points = face_point_dict[a, b]
        except KeyError:
            a_pt, b_pt = [points[idx] for idx in [a, b]]
            dx = [(b_pt[0] - a_pt[0])/factor, (b_pt[1] - a_pt[1])/factor ]

            # build subdivided facet
            face_points = [a]

            for i in range(1, points_per_edge-1):
                face_points.append(len(new_points))
                new_points.append([(a_pt[0] + dx[0]*i), (a_pt[1] + dx[1]*i)])

            face_points.append(b)

            face_point_dict[a, b] = face_points

            for i in range(factor):
                new_face.append([face_points[i], face_points[i+1]])

        if flipped:
            return face_points[::-1]
        else:
            return face_points


    for a, b, c in elements:
        a_pt, b_pt, c_pt = [points[idx] for idx in [a, b, c]]
        dr = [(b_pt[0] - a_pt[0])/factor, (b_pt[1] - a_pt[1])/factor]
        ds = [(c_pt[0] - a_pt[0])/factor, (c_pt[1] - a_pt[1])/factor]

        ab_refined, bc_refined, ac_refined = [
                get_refined_face(*pt_indices)
                for pt_indices in [(a, b), (b, c), (a, c)]]

        el_point_dict = {}

        # fill out edges of el_point_dict
        for i in range(points_per_edge):
            el_point_dict[i, 0] = ab_refined[i]
            el_point_dict[0, i] = ac_refined[i]
            el_point_dict[points_per_edge-1-i, i] = bc_refined[i]

        # fill out interior of el_point_dict
        for i in range(1, points_per_edge-1):
            for j in range(1, points_per_edge-1-i):
                el_point_dict[i, j] = len(new_points)
                new_points.append([(a_pt[0] + dr[0]*i + ds[0]*j), (a_pt[1] + dr[1]*i + ds[1]*j)])

        # generate elements
        for i in range(0, points_per_edge-1):
            for j in range(0, points_per_edge-1-i):
                new_elements.append((
                    el_point_dict[i, j],
                    el_point_dict[i+1, j],
                    el_point_dict[i, j+1],
                    ))
                if i+1+j+1 <= factor:
                    new_elements.append((
                        el_point_dict[i+1, j+1],
                        el_point_dict[i+1, j],
                        el_point_dict[i, j+1],
                        ))

    new_mesh = tri.MeshInfo()
    new_mesh.set_points(new_points)
    new_mesh.elements.resize(len(new_elements))
    new_mesh.faces.resize(len(new_face))
    for i, el in enumerate(new_elements):
        new_mesh.elements[i] = el

    for i, f in enumerate(new_face):
        new_mesh.faces[i] = new_face[i]

    return new_mesh 

