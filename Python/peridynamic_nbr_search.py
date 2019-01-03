import numpy as np
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
        self.location_code=None
        self.nx = None 
        self.ny = None
        self.bin_code = None
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
    
    def _put(self, currNode, horizon, num_leaves, depth):

        if(currNode.end==False):
            for i in range(num_leaves):
                self._put(currNode.leaves[i], horizon, num_leaves, depth)
        else:
            extents1 = self.compute_sub_domains(currNode.extents, horizon) 
            for i in range(num_leaves):
                currNode.leaves.append(TreeNode(extents1[i], depth))

            currNode.end = False

    def _mutate_string_at(self, location_code, new_loc, at_loc):
        """
        private helper function to mutate the location_code 
        of a node at a given level 
    
        input:
        ------
            location_code: string 
            new_loc      : single digit int, new location code
            at_loc:      : single digit int indicating where in 
                           location_code the new_loc goes
        """ 
        str_list = list(location_code)
        str_list[at_loc] = str(new_loc)
        
        return ''.join(str_list)

    def organize_tree_nodes(self):
        """
        public method that organizes additional 
        structure in the tree nodes starting at root
        till its depth as described
        in the paper : Constant Time Neighbor Finding 
        in Quadtrees by Aizwa et al

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
        location_code = str(0).zfill(tree_depth)
        root = self.root 
        root.location_code = str(0) #root is always zero
        nx = location_code 
        ny = location_code 
        bounding_box = root.extents
        root.is_boundary = True

        num_leaves = len(root.leaves)
        for i in range(num_leaves):
            self._organize_tree_nodes(root.leaves[i], tree_depth, i, 
                                     location_code, nx, ny, bounding_box)


    def _organize_tree_nodes(self, currNode, tree_depth, sibling_index, 
                             location_code, nx, ny, bounding_box):
        """
        private method that is called by public method
        organize_tree_nodes and recursively organizes 
        the TreeNode member variable location_code
        at each level of the tree

        input:
        -----
            self:
            currNode:   TreeNode member
            tree_depth:     
            sibling_index:  
        """
        level = currNode.level 
        num_leaves = len(currNode.leaves)
        currNode.location_code = self._mutate_string_at(location_code, sibling_index, level-1)

        if(sibling_index == 0):
            currNode.nx = nx 
            currNode.ny = ny 

        if(sibling_index == 1):
            currNode.nx = self._mutate_string_at(nx, 1, level-1)
            currNode.ny = ny 

        if(sibling_index == 2):
            currNode.nx = nx
            currNode.ny = self._mutate_string_at(ny, 1, level-1)

        if(sibling_index == 3):
            currNode.nx = self._mutate_string_at(nx, 1, level-1)
            currNode.ny = self._mutate_string_at(ny, 1, level-1)
        
        location_code = currNode.location_code
        nx = currNode.nx
        ny = currNode.ny; 
        currNode.is_boundary = (bounding_box == currNode.extents).any()
        
        if(currNode.end==False):
            for i in range(num_leaves):
                self._organize_tree_nodes(currNode.leaves[i], tree_depth, i, 
                                          location_code, nx, ny, bounding_box )
        else:
            return

    def organize_node_nbrs(self):
        """
        public method that collects immediate 
        neighbor extents for each node in the tree

        in 2D there can be upto 8 neighbors
        in 3D there can be upto 26 neighbors
        in ND there can be up to 3^N-1 neighbors
        """




