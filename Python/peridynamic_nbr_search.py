import numpy as np
import copy as cpy 
from fenics_mesh_tools import *


class TreeNode:
    """
    a quad tree node that serves 
    as either a child or a root of 
    the quad tree 

    expecting the key to be a 2D array
    """
    def __init__(self, extents=None, level=None):
        self.end = True
        self.leaves = []
        self.extents = extents
        self.level = level
        self.loc_code=None
        self.nx = None 
        self.ny = None
        self.bin_loc_code = None
        self.is_boundary=None


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
        if(depth==0):
            extent_array[:] = currNode.extents
        else:
            size = int(num_leaves**(depth-1))
            for i in range(num_leaves):
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
                self._put(self.root, horizon, num_leaves, self.depth+1)
                self.depth += 1
                ext_arry = self.iterate_quad_tree(self.root, self.depth, dim)
                ee = ext_arry[0]
                el = abs(min(ee[0] - ee[1]))
            else:
                self.root = TreeNode(extents, 0)

        self.organize_tree_nodes()
        return
    
    def _put(self, currNode, horizon, num_leaves, depth):

        if(currNode.end==False):
            for i in range(num_leaves):
                self._put(currNode.leaves[i], horizon, num_leaves, depth)
        else:
            extents1 = self.compute_sub_domains(currNode.extents, horizon) 
            for i in range(num_leaves):
                currNode.leaves.append(TreeNode(extents1[i], depth))

            currNode.end = False
        return


    def _assign_node_loc(self, loc_code, new_loc, at_loc):
        """
        private helper function to mutate the location_code 
        of a node at a given level 
    
        input:
        ------
            loc_code: string 
            new_loc      : single digit int, new location code
            at_loc:      : single digit int indicating where in 
                           location_code the new_loc goes
        """ 
        str_list = list(loc_code)
        str_list[at_loc] = str(new_loc)
        
        return ''.join(str_list)


    def _assign_bin_loc_code(self, treeNode):
        """
        private method accessd by _organize_tree_nodes

        creates the binary location code by combining 
        nx and ny. Refer to page 2 of Constant Time Neighbor
        Finding by Aizawa et. al. 

        """
        assert(len(treeNode.nx)==len(treeNode.ny))
        ll = len(treeNode.nx)
        lst1 = list(treeNode.nx); lst2 = list(treeNode.ny);
        bin_loc_code = []
        for i in range(ll):
            bin_loc_code.append(lst2[i])
            bin_loc_code.append(lst1[i])
        
        treeNode.bin_loc_code =  ''.join(bin_loc_code)
        return


    def organize_tree_nodes(self):
        """
        public method that organizes additional 
        structure in the tree nodes starting at root
        till its depth as described
        in the paper : Constant Time Neighbor Finding 
        in Quadtrees by Aizawa et. al.

        this method makes a call to private method
        with its root node and corresponding sibling 
        index which is zero for all root

        input:
        ------
            NONE
        output:
        ------
            NONE
        """

        tree_depth = self.depth
        loc_code = str(0).zfill(tree_depth)
        root = self.root 
        root.loc_code = str(0) #root is always zero
        nx = loc_code 
        ny = loc_code 
        bounding_box = root.extents
        root.is_boundary = True

        num_leaves = len(root.leaves)
        for i in range(num_leaves):
            self._organize_tree_nodes(root.leaves[i], i, 
                                     loc_code, nx, ny, bounding_box)


    def _organize_tree_nodes(self, currNode, sibInd, 
                             loc_code, nx, ny, bounding_box):
        """
        private method that is called by public method
        organize_tree_nodes and recursively organizes 
        the TreeNode member variable location_code
        at each level of the tree. 

        For more details on naming conventions for inputs 
        to the function refer to paper: Constant Time Neighbor
        Search in Quad tree by Aizawa et. al.

        input:
        -----
            self:
            currNode: TreeNode object, (grand..) child of root
            sibInd:   int, refers to the one of the four child of its ancestor
            loc_code: str, quartenary location code, num of chars = depth
            nx      : str, binary location code in x direction
            ny      : str, binary location code in y direction
            bounding_box: extents of the bounding box
        """

        level = currNode.level 
        num_leaves = len(currNode.leaves)
        currNode.loc_code = self._assign_node_loc(loc_code, sibInd, level-1)

        if(sibInd == 0):
            currNode.nx = nx 
            currNode.ny = ny 

        if(sibInd == 1):
            currNode.nx = self._assign_node_loc(nx, 1, level-1)
            currNode.ny = ny 

        if(sibInd == 2):
            currNode.nx = nx
            currNode.ny = self._assign_node_loc(ny, 1, level-1)

        if(sibInd == 3):
            currNode.nx = self._assign_node_loc(nx, 1, level-1)
            currNode.ny = self._assign_node_loc(ny, 1, level-1)
        
        self._assign_bin_loc_code(currNode)
        loc_code = currNode.loc_code
        nx = currNode.nx
        ny = currNode.ny; 
        currNode.is_boundary = (bounding_box == currNode.extents).any()
        
        if(currNode.end==False):
            for i in range(num_leaves):
                self._organize_tree_nodes(currNode.leaves[i], i, 
                                          loc_code, nx, ny, bounding_box )
        else:
            return

    def get_linear_tree(self):
        """
        public method that collects immediate 
        neighbor extents for each node in the tree

        in 2D there can be upto 8 neighbors
        in 3D there can be upto 26 neighbors
        in ND there can be up to 3^N-1 neighbors
        
        TODO: implement
        """
        linear_tree = []

        return "TODO :P sorry, I've not had the time"

        ### END OF CLASS QuadTree ###

def find_one_nbr(nq, delNi):

    """
    refer to equation 2 in the paper:
    Constant Time Neighbor Search by Aizawa et. al
    
    currently handles only 2D case
    TODO: generalize for 3D
    """

    tx = ['0','1']; ty = ['1', '0']
    dim = 2 #hard coded :(
    depth = len(nq)/dim
    
    tx = ''.join(tx*depth)
    ty = ''.join(ty*depth)
    nbr_int = (((int(nq,2)|int(tx,2))+(int(delNi,2)&int(ty,2)))&int(ty,2))|(((int(nq,2)|int(ty,2))+(int(delNi,2)&int(tx,2)))&int(tx,2))
    nbr_bin = bin(nbr)

    return nbr_int, nbr_bin[2:]

def get_binary_direction(dir1, dir2):
    """
    refer to page 2 of the paper: Constant Time Neighbor Search
    by Aizawa et.al.

    This function creates a combination of two directions
    lying on different axis like NE, NW, SE or SW
    Works only for 2D

                  N
              NW  |   NE
                  |
             W----|----E
                  |
              SW  |   SE
                  S
    this doesn't work for dir1==dir2, or when dir1, dir2 lie on
    same axis (like dir1 = North, and dir2 = South)
    input:
    ------
        dir1 : binary string direction1
        dir2 : binary string direction2
        length: length of resulting binary string
        
    output:
    -------
        returns the binary string direction as a combination of
        dir1 and dir2. For eg, if dir1 = North, dir2 = East,
        the function returns North-East binary dirction
    """
    #some sanity checks on directions
    assert(len(dir1)==len(dir2)) #directions represents same tree depth
    assert(dir1 != dir2)         # not same directions
    assert(dir1[-2:] != dir2[-2:])# directions not on same axis

    return bin((int(dir1,2)+int(dir2,2)))[2:].zfill(len(dir1))

def find_all_nbrs(nq):
    """
    Finds all the neighbors according to equation 2
    of the paper mentioned several times in this script
    currently works only for 2D
    for 2D there are 8 nbrs and for 3D there are 26
    
    TODO: generalize for 3D 

    """
    dim = 2; ll = len(nq)
    depth = int(len(nq)/ll)
    east =  ['0', '1']
    north = ['1', '0']

    east *= depth
    north *= depth
    west = ['0']*len(nq); west[-1] = '1'
    south = ['0']*len(nq); south[-2] = '1'
    
    EE = ''.join(east)
    WW = ''.join(west)
    NN = ''.join(north)
    SS = ''.join(south)
    
    SW = get_binary_direction(SS, WW, ll)
    SE = get_binary_direction(SS, EE, ll)
    NW = get_binary_direction(NN, WW, ll)
    NE = get_binary_direction(NN, EE, ll)

    delN = [WW, SW, SS, SE, EE, NE, NN, NW]
    nbr_int = []
    nbr_bin = []
    
    for delNi in delN:
        nbr_int_i, nbr_bin_i = find_one_nbr(nq, delNi)
        nbr_int.append(nbr_int_i)
        nbr_bin.append(nbr_bin_i)

    return nbr_int, nbr_bin 
