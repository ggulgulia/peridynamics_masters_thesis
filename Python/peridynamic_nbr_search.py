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
        
        if len(extents) == 4:

            self.one = TreeNode(extents[0])
            self.two = TreeNode(extents[1])
            self.three = TreeNode(extents[2])
            self.four = TreeNode(extents[3])
            self.end = False
        else:
            self.extents = extents
            self.one = None 
            self.two = None 
            self.three = None
            self.four = None 

    def hasOneChild(self):
        return (self.one)
    def hasTwoChild(self):
        return (self.two)
    def hasThreeChild(self):
        return (self.three)
    def hasFourChild(self):
        return (self.four)
    

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
        self.extents = None

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

    def put(self, extents, horizon):

        #while((0.5*np.sum(extents,axis=0) > 1.1*horizon).all()):
        while(True):
            if self.root:
                self._put(self.root, horizon)
                self.depth += 1
            else:
                self.root = TreeNode(extents)
    
    def _put(self, currNod, horizon):

        if(currNode.end==False):
            self._put(currNode.one, horizon)
            self._put(currNode.two, horizon)
            self._put(currNode.three, horizon)
            self._put(currNode.four, horizon)
        else:
             ee = currNode.extents
             ll = sqrt((ee[0][0]-ee[1][0])**2 + (ee[0][1]-ee[1][1])**2)
             if(ll< horizon):
                extents1 = self.compute_sub_domains(currNode.extents, horizon)
                currNode.one   = TreeNode(extents1[0])
                currNod.two    = TreeNode(extents1[1])
                currNode.three = TreeNode(extents1[2])
                currNode.four  = TreeNode(extents1[3])
                currNode.end = False
             else:
                 break
        
