import meshpy.triangle as tri
import numpy as np
import meshpy.geometry as geo
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from helper import *
from collections import OrderedDict as od
from six.moves import range

def plot_(mesh, annotate=False):
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    elems_cent = get_elem_centroid(mesh)

    x,y = np.array(elems_cent).T
    plt.scatter(x,y, color='r', marker='o')

    #TODO FIX: annotate the element centroid with element number
    #if annotate is True:
    #    for i in range(len(elems_cent)):
    #        x,y = np.array(elems_cent).T
    #        plt.annotate(i, xy=(x,y), xytext=(x-0.001, y+0.001), xycoords='data')

    plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
    plt.show(block=False)

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
    mesh = tri.build(mi, max_volume=1e-2, generate_faces=True, min_angle=35,
            mesh_order=None, generate_neighbor_lists=True)
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
    dim = np.shape(points[0])[0]
    elems_centroid = np.zeros((0,dim), dtype=float)

    for a, b,c in elems:
        [a_pt,b_pt,c_pt] = [points[idx] for idx in [a,b,c]]
        loc_elem_cent = np.zeros((1,dim), dtype=float)
        loc_elem_cent = (a_pt + b_pt + c_pt)/3

        elems_centroid = np.append(elems_centroid, loc_elem_cent.reshape(1,dim), axis=0)
    
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
    edge_lengths = np.zeros(0, dtype=float)
    for a, b in faces:
        a_pt , b_pt = [points[idx] for idx in [a,b]]
        loc_len = mod(a_pt - b_pt)
        edge_lengths = np.append(edge_lengths, loc_len)

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
    if factor == 1:
        return mesh
    else:
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


def get_peridym_mesh_bounds(mesh):
    """returns list of elements and a list
    of the centroids of the corresponding
    elements that lie next to the boundary 

    TODO: this works only for unit square mesh
    at the moement, generalize it to arbitrary
    shape

    input:
    -----
        mesh: meshpy.MeshInfo mesh object
        
    returns:
    -------
        ## see comments just before the return statement below ##

    """
    elems = np.array(mesh.elements).tolist()
    elem_cent = get_elem_centroid(mesh)
    pts = np.array(mesh.points)
    #assign element id to centroid
    elem_dict = {}
    for i in range(len(elems)):
        elem_dict[i] = elem_cent[i]

    corners = geo.bounding_box(pts)

    ll = corners[0][0]
    rr = corners[1][0]
    bb = corners[0][1]
    tt = corners[1][1]

    lft_bnd_elems = {}; rit_bnd_elems = {}; btm_bnd_elems = {}; top_bnd_elems = {} 
    lft_elem_cent = {}; rit_elem_cent = {}; btm_elem_cent = {}; top_elem_cent = {} 

    j = 0
    for a, b, c in elems:
        if (pts[a][0] == ll) or(pts[b][0]==ll) or (pts[c][0]==ll):
            lft_bnd_elems[j] = elem_dict[j]
            lft_elem_cent[j] = elem_dict[j]
        j +=1
        
    j = 0
    for a, b, c in elems:
        if (pts[a][0] == rr) or(pts[b][0]==rr) or (pts[c][0]==rr):
            rit_bnd_elems[j] = elem_dict[j]
            rit_elem_cent[j] = elem_dict[j]
        j +=1
    
    j = 0
    for a, b, c in elems:
        if (pts[a][1] == bb) or(pts[b][1]==bb) or (pts[c][1]==bb):
            btm_bnd_elems[j] = elem_dict[j]
            btm_elem_cent[j] = elem_dict[j]
        j +=1

    j = 0
    for a, b, c in elems:
        if (pts[a][1] == tt) or(pts[b][1]==tt) or (pts[c][1]==tt):
            top_bnd_elems[j] = elem_dict[j]
            top_elem_cent[j] = elem_dict[j]
        j +=1
    
    lft_elem_cent = od(sorted(lft_elem_cent.items()))
    rit_elem_cent = od(sorted(rit_elem_cent.items()))
    btm_elem_cent = od(sorted(btm_elem_cent.items()))
    top_elem_cent = od(sorted(top_elem_cent.items()))
    #returns 
    # 1. list of all node numbers that forms the trinagles along the boundary
    #    (above list is in the form of the list of list where the inner list 
    #     represents node set belonging to a triangle on the boundary )
    # 2. list of centroids of all such triangles
    return {"left":lft_bnd_elems, "right":rit_bnd_elems, "bottom":btm_bnd_elems, "top":top_bnd_elems}, \
            {"left":lft_elem_cent, "right":rit_elem_cent, "bottom": btm_elem_cent, "top":top_elem_cent}
