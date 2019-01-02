import numpy as np
from fenics_mesh_tools import *


class TreeNode:
    """
    a quad tree node that serves 
    as either a child or a root of 
    the quad tree 

    expecting the key to be a 2D array
    """
    def __init__(self, extents=None):
        self.end = True
        self.leaves = []
        self.extents = extents


class QuadTree:

    """
    this creates a hierearchal data 
    structure for 2D euclidian data 
    which is intended to speed up the 
    neighbor search algorithm
    """

    def __init__(self):
        self.root = None
        self.depth = 0
        self.horizon = 0.0 
        self.dim = None

    def depth(self):
        return self.depth

    def __depth__(self):
        return self.depth

    def __iter__(self):
        return self.root.__iter__()

    def compute_sub_domains(self, extents, horizon):
        """
        computes the four nodes of the quad tree
        taking the horizon into account as the ghost layer 
        of each node in the quad tree 
        """
        dim = len(extents[0])
        num_leaves = 2**dim

        midPoint = 0.5*sum(extents[:,])
        midVer = np.row_stack(np.broadcast(midPoint[0], extents[:,1]))
        midHor = np.row_stack(np.broadcast(extents[:,0], midPoint[1]))

        extents_new = np.zeros((num_leaves,2,dim), dtype=float)
        extents_new[0] = np.vstack((midPoint, extents[1]))
        extents_new[1] = np.vstack((midHor[0], midVer[1]))
        extents_new[2] = np.vstack((midVer[0], midHor[1]))
        extents_new[3] = np.vstack((extents[0], midPoint))

        return extents_new

    def iterate_quad_tree(self, currNode, depth, dim):
        """
        public member function that iterates down the tree from
        currNode till depth provided to it. 
        
        If the depth exceeds the valid depth from currNode, 
        then the function returns the extents till the valid 
        depth down the currNode 

        For 2D this is a Quadtree and for 3D the tree is an
        Octree
        
        input:
        -----
            self:
            currNode: TreeNode at an arbitrary level
            depth:    Depth that needs to be traversed
            dim:      Dimension of the problem

        """
        num_leaves = 2**dim
        extents = np.zeros((num_leaves**depth,2,dim), dtype=float)
        self._iterate(currNode, depth, extents, num_leaves)

        return extents

    def _iterate(self, currNode, depth, extent_array, num_leaves):

        """
        private member function that is internally called by iterate_quad_tree
        method 

        """
        if(currNode.end==True):
            extent_array[:] = currNode.extents
        else:
            size = int(num_leaves**(depth-1))
            for i in range(4):
                self._iterate(currNode.leaves[i], depth-1, extent_array[int(i*size):int((i+1)*size)], num_leaves)

    def put(self, extents, horizon):

        """
        public method that recursively decomposes a rectangle domain
        and adds tree node child to the tree root until the child subomain
        edge length is less than or equal to horizon

        input:
        ------
            self:
            extents: extents of bounding box
            horizon: peridynamic horizon
        output:
        ------
            None
        """
        dim = len(extents[0])
        num_leaves = 2**dim
        #el: edge_length
        el = min(abs(extents[0] - extents[1]))
        self.horizon = horizon
        self.dim = dim

        while(el > horizon):
            if self.root:
                self._put(self.root, horizon, num_leaves)
                self.depth += 1
                ext_arry = self.iterate_quad_tree(self.root, self.depth, dim)
                ee = ext_arry[0]
                el = abs(min(ee[0] - ee[1]))
            else:
                self.root = TreeNode(extents)
    
    def _put(self, currNode, horizon, num_leaves):

        if(currNode.end==False):
            for i in range(num_leaves):
                self._put(currNode.leaves[i], horizon, num_leaves)
        else:
            extents1 = self.compute_sub_domains(currNode.extents, horizon) 
            for i in range(num_leaves):
                currNode.leaves.append(TreeNode(extents1[i]))

            currNode.end = False

