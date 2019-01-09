import numpy as np
import copy as cpy 
from fenics_mesh_tools import *
import timeit as tm


class TreeNode:
    """
    a quad tree node that serves  as either a child or a root of 
    the quad tree expecting the key to be a 2D array

    for literature refer: Constant Time Neighbor Search in Quadtree
    by Aizawa et.al.

    members
    -------
        end          :bool,whether the tree node has child
        leaves       :empty list to hold child TreeNode
        extents      :np.array, subdomain bounding box
        level        :int, level in tree to which the TreeNode refers
        loc_code     :Quartenary/Octal location code acc. to literature
        nx           :binary location code along x-direction acc. to literature
        ny           :binary location code along y-direction acc. to literature
        bin_loc_code :binary location code acc. to literature
        has_bounds   :if the extents in Tree Node forms a part
                      of the boundary
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
        self.has_bounds=None


class QuadTree:

    """
    this creates a hierearchal data 
    structure for 2D euclidian data 
    which is intended to speed up the 
    neighbor search algorithm

    members:
    -------
        root    :TreeNode object serving as the root of the QuadTree
        depth   :int, denotes the depth of the tree
        horizon :peridynamic horizon
        dim     :geometric dimension of the problem 

    NOTE: this currently handles only 2D
    TODO: extend to 3D : OctTree
    """

    def __init__(self):
        self.root = None
        self.depth = 0
        self.horizon = 0.0 
        self.dim = None

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

        while(True):
            if self.root:
                ext_arry = self.iterate_quad_tree(self.root, self.depth, dim)
                ee = ext_arry[0]
                el = np.zeros(self.dim, dtype=float)
                for d in range(dim):
                    el[d] = abs(np.diff(ee[:,d]))
                if ((el < horizon).any()):
                    break

                self._put(self.root, horizon, num_leaves, self.depth+1)
                self.depth += 1
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
            loc_code     : binary string 
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
        root.has_bounds = True

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
        currNode.has_bounds = (bounding_box == currNode.extents).any()
        
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
        depth = self.depth
        root = self.root
        dim = self.dim
        horizon = self.horizon
        linear_tree = {}
        num_leaves = len(root.leaves)

        for i in range(num_leaves):
            self._get_bin_loc_code_and_extents(depth, root, horizon, linear_tree)

        return linear_tree

    def _get_bin_loc_code_and_extents(self, treeDepth, currNode, horizon, linear_tree):
        
        if(currNode.level != treeDepth):
            num_leaves = len(currNode.leaves)
            for i in range(num_leaves):
                 self._get_bin_loc_code_and_extents(treeDepth, currNode.leaves[i], horizon, linear_tree)
        else:
            linear_tree[currNode.bin_loc_code] = currNode.extents

            return linear_tree

        ### END OF CLASS QuadTree ###
