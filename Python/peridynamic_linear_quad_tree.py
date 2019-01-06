from peridynamic_quad_tree import *

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
    #see direction nomenclature in docstring
    if(nq[1::2]=='111'):
        #no west direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(west))

    if(nq[0::2]=='111'):
        #no south direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(south))

    if(nq[1::2]=='000'):
        #no east direction
        tempCard.append(None)
    else:
        tempCard.append(''.join(east))

    if(nq[0::2]=='000'):
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
