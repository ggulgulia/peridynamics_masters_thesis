from peridynamic_plane_stress import *
from fenics_plane_stress import *
from peridynamic_infl_fun import *


def compare_PD_horizons_with_FE(horizons, mesh, npts=15, material='steel', plot_=False, force=-5e8, struct_grd=False):
    """
    compares the FE and PD solution for a simple 2D case 

    :horizons: TODO
    :mesh: TODO
    :npts: TODO
    :material: TODO
    :'plot_: TODO
    :returns: TODO

    """
    if(struct_grd):
        cell_cent = structured_cell_centroids(mesh)
        base_horizon = 3*np.diff(cell_cent[0:2][:,0])[0]
        el = np.diff(cell_cent[0:2][:,0])[0]
    else:
        cell_cent = get_cell_centroids(mesh)
        base_horizon = 3*mesh.hmax()
        el = mesh.hmax()
    
    print("*********************************************************")
    print("*********************************************************")
    print('solving using FEniCS:')
    disp_cent_FE, u_disp_FE = solve_fenic_bar(mesh, cell_cent, npts, material, plot_=plot_, force=force)

    dim = np.shape(cell_cent)[1]
    
    disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
    u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)

    infl_fun = gaussian_infl_fun2

    print("*********************************************************")
    print("*********************************************************")
    print('solving using Peridynamics:')
    for i in range(len(horizons)):
        _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizons[i], mesh, npts=npts,material=material,
                                                           omega_fun=infl_fun, plot_=plot_, force=force,struct_grd=struct_grd)

        disp_cent_PD_array[i] = disp_cent_i
        u_disp_PD_array[i]    = u_disp_i 
        print("*********************************************************")
        print("*********************************************************")
    
    top_els = np.ravel(np.argwhere(cell_cent[:,1] == np.max(cell_cent[:,1])))

    u_top_fe = u_disp_FE[top_els]
    cell_cent_top = cell_cent[top_els]
    
    plt.figure()
    plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, label='FE Solution')

    for i in range(len(horizons)):
        mm = int(horizons[i]/el)
        u_disp_pd_top = u_disp_PD_array[i][top_els]
        plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], linewidth=2.0, label='Horizon='+ str(mm) +'*'+str(el))

    plt.legend()
    plt.title('displacement of top centroids mesh size: %i, el: %4.3f'%(len(cell_cent), el))
    plt.show(block=False)
    plt.savefig('FE_vs_PD_displacements')

    return disp_cent_FE, u_disp_FE, disp_cent_PD_array, u_disp_PD_array



def compare_PD_infl_funs_with_FE(horizon, mesh, npts=15, material='steel', plot_=False, force=-5e8, struct_grd=False):
    """
    compares the FE and PD solution for a simple 2D case 

    :horizons: TODO
    :mesh: TODO
    :npts: TODO
    :material: TODO
    :'plot_: TODO
    :returns: TODO

    """
    print("*********************************************************")
    print("*********************************************************")
    print('solving using FEniCS:')
    disp_cent_FE, u_disp_FE = solve_fenic_bar(mesh, material, force=force)

    if(struct_grd):
        cell_cent = structured_cell_centroids(mesh)

    else:
        cell_cent = get_cell_centroids(mesh)
    
    dim = np.shape(cell_cent)[1]
    #dictonary of all influence functions
    infl_fun_dict = {'omega1':gaussian_infl_fun1,
                      'omega2':gaussian_infl_fun2,
                      'omega3':parabolic_infl_fun1,
                      'omega4':parabolic_infl_fun2}

    #horizon = 0.175
    keys = infl_fun_dict.keys()

    disp_cent_PD_array = {} 
    u_disp_PD_array = {}

    print("*********************************************************")
    print("*********************************************************")
    print('solving using Peridynamics:')
    tree = QuadTree()
    extents = get_domain_bounding_box(mesh)
    tree.put(extents, horizon)
    nbr_lst, nbr_beta_lst = tree_nbr_search(tree.get_linear_tree(), cell_cent, horizon, struct_grd)
    for kk in keys:
        infl_fun = infl_fun_dict[kk]
        _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizon, mesh,nbr_lst=nbr_lst, nbr_beta_lst=nbr_beta_lst, material=material,omega_fun=infl_fun, force=force, struct_grd=struct_grd)

        disp_cent_PD_array[kk] = disp_cent_i
        u_disp_PD_array[kk]    = u_disp_i 
        print("*********************************************************")
        print("*********************************************************")
    
    top_els = np.ravel(np.argwhere(cell_cent[:,1] == np.max(cell_cent[:,1])))

    u_top_fe = u_disp_FE[top_els]
    cell_cent_top = cell_cent[top_els]
    
    plt.figure()
    plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, label='FE Solution')

    for kk in keys:
        u_disp_pd_top = u_disp_PD_array[kk][top_els]
        plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], linewidth=2.0, label='infl_fun: ' + kk)

    plt.legend()
    plt.title('displacement of top centroids, horizon='+str(horizon))
    plt.show(block=False)
    plt.savefig('FE_vs_PD_displacements_delta'+str(horizon)+".png")

    return disp_cent_FE, u_disp_FE, disp_cent_PD_array, u_disp_PD_array
