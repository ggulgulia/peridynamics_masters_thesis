from peridynamic_quad_tree import *
import timeit as tm

def find_one_nbr(nq, delNi):

    """
    refer to equation 2 in the paper:
    Constant Time Neighbor Search by Aizawa et. al
    
    currently handles only 2D case
    TODO: generalize for 3D
    """
    
    ll = len(nq)
    if(delNi == bin(0)[2:].zfill(ll)):
        #sanity check, no need to compute self
        #if delNi = '00...0'
        return 

    tx = ['0','1']; ty = ['1', '0']
    dim = 2 #hard coded :(
    depth = int(len(nq)/dim)
    
    tx = ''.join(tx*depth)
    ty = ''.join(ty*depth)
    #equation 2 in paper Constant Time Neighbor finding in Quad Tree
    nbr_int = (((int(nq,2)|int(tx,2))+(int(delNi,2)&int(ty,2)))&int(ty,2))|(((int(nq,2)|int(ty,2))+(int(delNi,2)&int(tx,2)))&int(tx,2))
    nbr_bin = bin(nbr_int)[2:].zfill(len(nq))

    return nbr_int, nbr_bin

def get_binary_direction(dir1, dir2):
    """
    refer to page 2 of the paper: Constant Time Neighbor Search
    by Aizawa et.al.

    This function creates a combination of two directions
    lying on different axis like NE, NW, SE or SW
    Works only for 2D
    input:
    ------
        dir1 : binary string direction1
        dir2 : binary string direction2
        length: length of resulting binary string
        
    output:
    -------
        returns the binary string direction as a combination of
        dir1 and dir2. 
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
    
                  NN
              NW  |   NE
                  |
            WW----|----EE
                  |
              SW  |   SE
                  SS

    desired delN order: [WW, SW, SS, SE, EE, NE, NN, NW]

    this is the order with which we desire the neighbors
    binary location code

    TODO: generalize for 3D 
    """
    dim = 2
    ll = len(nq)
    depth = int(ll/dim)
    east =  ['0', '1']
    north = ['1', '0']

    east *= depth
    north *= depth
    west = ['0']*len(nq); west[-1] = '1'
    south = ['0']*len(nq); south[-2] = '1'
    
    tempCard = [] #Cardinal directions
    corner0 = ''.join(['0']*depth)
    corner1 = ''.join(['1']*depth)
    #see direction nomenclature in docstring
    if(nq[1::2]==corner1):
        #no west direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(west))

    if(nq[0::2]==corner1):
        #no south direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(south))

    if(nq[1::2]==corner0):
        #no east direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(east))

    if(nq[0::2]==corner0):
        #no north direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(north))
    #final order infinal order in tempCard = [WW, SS, EE, NN]

    tempComb = [] #combination of carinal directions
    for tt in tempCard[1::2]: #SS or NN
        for vv in tempCard[0::2]: #WW or EE
            if((tt) and (vv)):
                tempComb.append(get_binary_direction(tt,vv))
            else:
                tempComb.append(None)
    # current order tempComb = [SW, SE, NW, NE]
    
    ## swap NW and NE positions in tempComb
    ## i know  this is ugly but I cannot
    ## think of a better way at the moment
    a = tempComb[-1]; b = tempComb[-2]
    tempComb[-1]=b;   tempComb[-2]=a
    #final (and desired) order tempComb = [SW, SE, NE, NW]

    delN = []
    assert(len(tempCard) == len(tempComb))
    for i in range(len(tempCard)):
        if tempCard[i]:
            delN.append(tempCard[i])
        if(tempComb[i]):
            delN.append(tempComb[i])
    #delN order = [WW, SW, SS, SE, EE, NE, NN, NW]
    # above delN order is circular (see doc string graphic)

    nbr_int = []
    nbr_bin = []
    for delNi in delN:
        nbr_int_i, nbr_bin_i = find_one_nbr(nq, delNi)
        nbr_int.append(nbr_int_i)
        nbr_bin.append(nbr_bin_i)

    return nbr_int, nbr_bin 

def tree_nbr_search(linear_tree, cell_cents, horizon, vol_corr=True, struct_grd=False):
    """
    linear tree nbr search

    input:
    ------
        cent_linear_tree : dictonary of binary_location code of 
                       all subdomains
        cell_cents        : numpy array of cell centroid
        horizon          : pridynamic horizon
        vol_corr    : boolean whether we need volume correction or not
        struct_grd  : boolean, whether using triangulation or squared lattice

    output:
    -------
        nbr_lst 
    """


    print("computing the neighbor list of the mesh with horizon size of %4.6f"%horizon)
    start = tm.default_timer()
    kk = linear_tree.keys()
    nbr_lst = []
    nbr_beta_lst = []
    for k in kk:
        _, bin_nbrs = find_all_nbrs(k)
        nbr_cells = np.empty(0,int)
        nbr_lst_k, nbr_beta_lst_k =compute_single_nbr_lst(linear_tree, k, horizon, cell_cents, vol_corr, struct_grd)
        nbr_lst = nbr_lst + nbr_lst_k
        nbr_beta_lst = nbr_beta_lst + nbr_beta_lst_k
    
    """ 
    both nbr_lst and nbr_beta_lst have first entry of the array
    as the parent cell index which points to the corresponding 
    nbr lst for that cell. nbr_lst at this point is unsorted. 
    Hence the nbr_lst & nbr_beta_lst is first sorted based on the 
    first entry (which is the cell index) and then this cell
    index is removed from both the list
    """

    ### SORT based on cell idx
    nbr_lst = sorted(nbr_lst, key=lambda x: x[0])
    nbr_beta_lst = sorted(nbr_beta_lst, key=lambda x: x[0])

    ## REMOVE the cell idx
    nbr_lst_mod = [np.delete(ll, 0) for ll in nbr_lst]
    nbr_beta_lst_mod = [np.delete(ll, 0) for ll in nbr_beta_lst]
    end = tm.default_timer()
    print("time taken to compute tree neighbor list= %4.5f sec"%(end-start))

    return nbr_lst_mod, nbr_beta_lst_mod

def compute_nbr_sub_domain_cells(linear_tree, bin_code, horizon, cell_cents):
    """
    given a linear tree and a binary location code 
    belonging to the tree, this method computes
    the cell centroid ids of all the cells lying
    in the neighborhood subdomains of the corres-
    ponding subdomain reffered to by the binary
    location code

    input:
    ------
        linear_tree : dictonary, key=binary location code
                     of subdomains, values= extents of subdomains
        bin_code    : binary location code of subdomain whose nbrs
                      we're interested to find
        horizon     : float, peridynamic horizon
        cell_cents  : np.array, centroids of cells in mesh

    output:
    -------
        nbr_cells: np.array of all nbr cell cent ids 
                   from the nbr subdomains corresponding
                   to the subdomain reffered to by bin code
    """
    nbr_cells = np.empty(0, int)
    _, bin_nbrs = find_all_nbrs(bin_code)
    
    dim = len(cell_cents[0])
    ee_curr = cpy.deepcopy(linear_tree[bin_code])
    el = np.zeros(dim, dtype=float) #array for edge lengths
    for d in range(dim):
        el[d] = np.asscalar(np.diff(ee_curr[:,d]))
    
    idx = np.argmin(el, axis=0)
    delta = 1.04*horizon - el[idx]
    
    for bb in bin_nbrs:
        ee = cpy.deepcopy(linear_tree[bb])

        """
        if the edge length is different along 
        one of the axis, and is smaller than horizon
        then, we need to expand it along that axis by
        the amount it is smaller than horizon

        done by the two if statements below
        """
        if(ee[0][idx] < ee_curr[0][idx]):
            ee[0][idx] -= delta
        if(ee[1][idx] > ee_curr[1][idx]):
            ee[1][idx] += delta

        nn_cc = get_cell_centroid2(cell_cents, ee)
        nbr_cells = np.append(nbr_cells, nn_cc)

    return np.unique(nbr_cells)

def compute_single_nbr_lst(linear_tree, bin_code, horizon, cell_cents, vol_corr, struct_grd):
    """
    given a binary location code for a sub domain and a linear quad tree
    that the subdomain is associated with, this method computes the 
    neighbor list of all the cell centroid ids located in the subdomain
    in consideration and the corresponding neighbor volume fraction list

    The cells may not completely lie within the spherical horizon and 
    those that lie partially within the horizon are also accounted for 
    in the nbr_beta_lst, beta here referring to the volume fraction of
    the cell that lie in the spherical horizon

    input:
    ------
        linear_tree : dictonary of linear tree, key= binary location code, 
                      values = subdomain extents
        bin_code    : binary location code of subdomain under consideration
        horizon     : peridynamic horizon
        cell_cents  : np array of all cell centroids
        vol_corr    : boolean wheather we need volume correction or not
        struct_grd  : boolean, wheather using triangulation(unstructured grid) or not

    output:
    -------
        nbr_lst     : nbr lst of all cell cents lying in the subdomain reffered
                      to by the bin_code
        nbr_beta_lst: list of volume factions of all the cells that are in the 
                      nbr_lst. 
    """
    nbr_lst = []
    nbr_beta_lst = []
    nbr_cells = compute_nbr_sub_domain_cells(linear_tree, bin_code, horizon, cell_cents)
    curr_cells = get_cell_centroid2(cell_cents, linear_tree[bin_code])

    for i, c  in enumerate(curr_cells):
        temp_nbr_cents = cpy.deepcopy(nbr_cells)
        curr_nbrs = []
        temp_nbr_cents = np.append(temp_nbr_cents, curr_cells[0:i])
        temp_nbr_cents = np.append(temp_nbr_cents, curr_cells[i+1:])
        temp_nbr_cents = np.unique(temp_nbr_cents)
        if(vol_corr==True):
            #compute correction factor range
            if(struct_grd):
                fract = 0.5 # for square lattices
                el = np.max(np.diff(cell_cents[0:2], axis=0))
            else:
                fract = (1.0/3.0) #for triangulations 
                el =abs(np.max(np.diff(cell_cents[0:3:2],axis=0))) 

            delta_plus = horizon + fract*el
            delta_minus = horizon - fract*el
            curr_beta = []
            for j in temp_nbr_cents:
                xi = la.norm((cell_cents[j]-cell_cents[c]))
                if(la.norm((cell_cents[j]-cell_cents[c]),2)<= delta_minus):
                    curr_beta.append(1.0)
                    curr_nbrs.append(j)
                elif((delta_minus < xi) and (delta_plus >= xi)):
                    beta = (horizon + fract*el - xi)/el
                    curr_beta.append(beta)
                    curr_nbrs.append(j)

        elif(vol_corr==False):
            for j in temp_nbr_cents:
                if(la.norm((cell_cents[j]-cell_cents[c]),2)<=horizon):
                    curr_nbrs.append(j)
            curr_beta = [1]*len(curr_nbrs)

        nbr_lst.append(np.array([c]+curr_nbrs))
        nbr_beta_lst.append(np.array([c]+curr_beta))

    return nbr_lst, nbr_beta_lst


def test_nbr_lst(nbr_lst_tree, nbr_lst_naive):

    """
    this is a helper function and checks if the 
    nbr_lst from tree data structure is exactly 
    same as the one obtained from naive linear 
    search. 
    
    The function force exits if at any 
    index of the cell centroid, the nbr list 
    for that cell doesn't match

    NOTE: the order of input parameters to method
          doesn't matter, which means the user
          has to be honest to put the nbr_lsts
          generated by two methods ;r, which means the user
          has to be honest to put the nbr_lsts
          generated by two methods ;)
    """
    import sys
    for i in range(len(nbr_lst_naive)):
        if((nbr_lst_tree[i] == nbr_lst_naive[i]).all()):
            #print(i,'true')
            pass
        else:
                sys.exit("cell id %i doesnt have a matching nbr_lst compared to naive nbr_lst:" %i)

    print("Hurrayy, the nbr_lst from tree is correct")
