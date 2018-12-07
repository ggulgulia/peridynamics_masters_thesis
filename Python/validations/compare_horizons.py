from peridynamic_plane_stress import *
from fenics_plane_stress import *
from peridynamic_infl_fun import *
from fenics import *
from fenics_mesh_tools import *


def compare_FE_and_PD(horizons, mesh, npts=15, material='steel', plot_=False, force=-5e8):
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
    disp_cent_FE, u_disp_FE = solve_fenic_bar(mesh,npts,material, plot_=plot_, force=force)

    cell_cent = get_cell_centroids(mesh)
    dim = np.shape(cell_cent)[1]
    
    disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
    u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)

    infl_fun = gaussian_infl_fun2

    print("*********************************************************")
    print("*********************************************************")
    print('solving using Peridynamics:')
    for i in range(len(horizons)):
        _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizons[i], mesh, npts=npts,material=material,
                                                           omega_fun=infl_fun, plot_=plot_, force=force)

        disp_cent_PD_array[i] = disp_cent_i
        u_disp_PD_array[i]    = u_disp_i 
    
    flag = 799 #top right corner element of the bar that we wish to monitor
    top_els = np.arange(723, 803, 4, dtype=int) #centroids of all top elements


    u_top_fe = u_disp_FE[top_els]
    cell_cent_top = cell_cent[top_els]
    
    plt.figure()
    plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, label='FE Solution')

    for i in range(len(horizons)):
        u_disp_pd_top = u_disp_PD_array[i][top_els]
        plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], linewidth=2.0, label='PD Solution, delta='+str(horizons[i]))

    plt.legend()
    plt.title('displacement of top centroids')
    plt.show(block=False)
    plt.savefig('FE_vs_PD_displacements')
