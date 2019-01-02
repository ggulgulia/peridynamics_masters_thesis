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
        
        dim = len(extents[0])
        num_leaves = int(2**dim)

        if len(extents) == num_leaves:
            for i in range(num_leavs):
                self.leaves.append(TreeNode(extents[i]))
        else:
            self.extents = extents

    def hasOneChild(self):
        return (self.leaves[0])
    def hasTwoChild(self):
        return (self.leaves[1])
    def hasThreeChild(self):
        return (self.leaves[2])
    def hasFourChild(self):
        return (self.leaves[3])
    

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
        midX = 0.5*sum(extents[:,0])
        midY = 0.5*sum(extents[:,1])
        
        midXBottom = np.array((midX, extents[0][1]))
        midXTop    = np.array((midX, extents[1][1]))
        midYLeft   = np.array((extents[0][0], midY))
        midYRight  = np.array((extents[1][0], midY))
        midPoint   = np.array((midX, midY))

        extentL = np.vstack((extents[0], midXTop))
        extentR = np.vstack((midXBottom, extents[1]))

        extent1 = np.vstack((midPoint, extents[1]))
        extent2 = np.vstack((midYLeft, midXTop))
        extent3 = np.vstack((midXBottom, midYRight))
        extent4 = np.vstack((extents[0], midPoint))

        extents = np.zeros((4,2,2), dtype=float)
        extents[0] = extent1; extents[1] = extent2;
        extents[2] = extent3; extents[3] = extent4 


        return extents

    def iterate_quad_tree(self, currNode, depth):
        extents = np.zeros((4**depth,2,2), dtype=float)
        self._iterate(currNode, depth, extents)

        return extents

    def _iterate(self, currNode, depth, extent_array):

        if(currNode.end==True):
            extent_array[:] = currNode.extents
        else:
            size = int(4**(depth-1))
            for i in range(4):
                self._iterate(currNode.leaves[i], depth-1, extent_array[int(i*size):int((i+1)*size)])

    def put(self, extents, horizon):

        el = abs(extents[0][0] - extents[1][0])
        while(el > horizon):
            if self.root:
                self._put(self.root, horizon)
                self.depth += 1
                ee_array = self.iterate_quad_tree(self.root, self.depth)
                ee = ee_array[0]
                el = abs(ee[0][0] - ee[1][0])
            else:
                self.root = TreeNode(extents)
    
    def _put(self, currNode, horizon):

        if(currNode.end==False):

            for i in range(4):
                self._put(currNode.leaves[i], horizon)
        else:
            extents1 = self.compute_sub_domains(currNode.extents, horizon) 

            for i in range(4):
                currNode.leaves.append(TreeNode(extents1[i]))

            currNode.end = False
