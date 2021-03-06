from peridynamic_quad_tree import *
from peridynamic_linear_quad_tree import *
from peridynamic_neighbor_data import *


def generate_test_mesh(factor):

    baseX = 10; baseY = 5
    numX = factor*baseX
    numY = factor*baseY

    return RectangleMesh(Point(0,0), Point(2,1), numX, numY)


def main():

    num_tests = 10
    
    import timeit as tm
    print("Starting to test the quad tree neighbor search algo")
    start_global = tm.default_timer()

    tree_time  = np.zeros(num_tests, float)
    naive_time = np.zeros(num_tests, float)
    mesh_size = np.zeros(num_tests, int)

    for i in range(num_tests):
        print("***************************************")
        print("Starting test number %i" %(i+1))
        mm = generate_test_mesh(i+1)
        print("num cells: %i" %mm.num_cells())
        cell_cents = get_cell_centroids(mm)
        mesh_size[i] = mm.num_cells()
        extents = get_domain_bounding_box(mm)
        horizon = 3*mm.hmax()
        tree = QuadTree()
        tree.put(extents, horizon)

        start_loc = tm.default_timer()
        tree_nbr, _ = tree_nbr_search(tree.get_linear_tree(), cell_cents, horizon, vol_corr=False)
        tree_time[i] = tm.default_timer() - start_loc

        start_loc = tm.default_timer()
        naive_nbr = peridym_compute_neighbors(mm, horizon)
        naive_time[i] = tm.default_timer() - start_loc

        test_nbr_lst(tree_nbr, naive_nbr)
        print("*************end of test***************")
        print("***************************************\n")

    plt.figure()
    plt.scatter(mesh_size, tree_time, marker='o', color='g', s=30, label='tree time')
    plt.scatter(mesh_size, naive_time, marker='o', color='r', s=30, label='naive time')
    plt.legend()
    plt.title("plot of time for nbr list using quad tree and naive method")
    plt.show(block=False)

    print("Time for test: %4.3f seconds" %(tm.default_timer()-start_global)) 

    return tree_time, naive_time, mesh_size
