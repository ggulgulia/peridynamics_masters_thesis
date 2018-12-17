import numpy as np
from fenics_mesh_tools import *


class TreeNode:
    """
    a quad tree node that serves 
    as either a child or a root of 
    the quad tree 

    expecting the key to be a 2D array
    """
    def __init__(self, cents, one=None, two=None, three=None, 
                 four=None, extents=None):
        self.end = True
        
        if(one is not None):
            self.one = TreeNode(one, extents = extents[0])
            self.end = False
        else:
            self.one = one

        if(two is not None):
            self.two = TreeNode(two, extents = extents[1])
            self.end = False
        else:
            self.two = two 

        if(three is not None):
            self.three = TreeNode(three, extents = extents[2])
            self.end = False
        else:
            self.three = three 
        if(four is not None):
            self.four = TreeNode(four, extents = extents[3])
            self.end = False
        else:
            self.four = four

        self.cents = None
        self.extents = None
        
        if self.end == True:
            self.cents = cents
            self.extents = extents

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
        self.coords = None 
        self.depth = 0
        self.horizon = 0.0 
        self.extents = None

    def depth(self):
        return self.depth

    def __depth__(self):
        return self.depth

    def __iter__(self):
        return self.root.__iter__()

    def compute_sub_domains(self, coords, extents, horizon):
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

        k = 1.1
        # divide along x-axis 
        rr_id = np.where(coords[:,0] >= midX)
        ll_id = np.where(coords[:,0] <= midX)
        rr = coords[rr_id]
        ll = coords[ll_id]

        #divide along y-axis
        one_id   = np.where(rr[:,1] >= midY)
        two_id   = np.where(ll[:,1] >= midY)
        three_id = np.where(rr[:,1] <= midY)
        four_id  = np.where(ll[:,1] <= midY)

        one   = rr[one_id];   two  = ll[two_id]
        three = rr[three_id]; four = ll[four_id]

        return one, two, three, four, extents

    def put(self, coords, extents, horizon):

        #while((0.5*np.sum(extents,axis=0) > 1.1*horizon).all()):
        while(True):
            if self.root:
                self._put(self.root, self.root.extents, horizon)
                self.depth += 1
            else:
                self.extents = extents
                self.coords = coords
                one, two, three, four, extents = self.compute_sub_domains(coords, extents, horizon)
                self.root = TreeNode(coords, one, two, three, four, extents)
                self.depth  += 1
    
    def _put(self, currNode, extents, horizon):

        if(currNode.hasOneChild()):
            self._put(currNode.one, currNode.one.extents, horizon)
        else:
            sub11, sub21, sub31, sub41, extents1 = self.compute_sub_domains(currNode.cents, extents, horizon)
            currNode.one   = TreeNode(currNode.cents, sub11, sub21, sub31, sub41, extents1)
        
        if(currNode.hasTwoChild()):
            self._put(currNode.two, currNode.two.extents, horizon)
        else:
            sub12, sub22, sub32, sub42, extents2 = self.compute_sub_domains(currNode.cents, extents, horizon)
            currNode.two   = TreeNode(currNode.cents, sub12, sub22, sub32, sub42, extents2)
        
        if(currNode.hasThreeChild()):
            self._put(currNode.three, currNode.three.extents, horizon)
        else:
            sub13, sub23, sub33, sub43, extents3 = self.compute_sub_domains(currNode.cents, extents, horizon)
            currNode.three = TreeNode(currNode.cents, sub13, sub23, sub33, sub43, extents3)
        
        if(currNode.hasFourChild()):
            self._put(currNode.four, currNode.four.extents, horizon)
        else:
            sub14, sub24, sub34, sub44, extents4 = self.compute_sub_domains(currNode.cents, extents, horizon)
            currNode.four  = TreeNode(currNode.cents, sub14, sub24, sub34, sub44, extents4)

