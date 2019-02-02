from peridynamic_plane_stress import *
from fenics_convergence import *
from peridynamic_infl_fun import *

def run_comparisons():
    print("*********************************************************")
    print("*********************************************************")
    print('solving using Finite Elements:')
    tol = 1e-2#tolerance for convergence of FE solution
    _, _, mesh_lst, u_fe_conv = fenics_mesh_convergence(tol=tol, plot_=False)
    print("Number of cells in FEniCS mesh on which the FE solution converged %i" %mesh_lst[-1].num_cells())
    print("*********************************************************")
    print("*********************************************************")


    #######################################################################################################
    ##                              CREATE A USEFUL MESH LIST OF MANAGABLE SIZE                         ##
    #######################################################################################################

    ##remove very coarse meshes since they are
    ##anywy useless for peridynamic computations
    grd_fact = (1+int(struct_grd))
    slice_idx = 0
    #for idx, mm in enumerate(mesh_lst):
    #    num_cells = mm.num_cells()
    #    if(int(num_cells/grd_fact) <1600):
    #        print("removing the mesh at index %i due to low cell count (%i) for peridynamic calculations"%(idx, int(num_cells/grd_fact)))
    #        slice_idx = idx

    #mesh_lst = mesh_lst[slice_idx+1:]
    #        
    ###we don't want to wait forever for computations
    ###on this serial code so we slice the list to a managable size 
    #if(len(mesh_lst)> 5):
    #    print("Too many meshes in the list, resizing to managable size")
    #    mesh_lst = mesh_lst[0:5]

    #######################################################################################################
    #######################################################################################################


    ##triangular mesh, horizon studies
    result1_wi_vol_corr = compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, vol_corr=True,  struct_grd=False)
    result1_wo_vol_corr = compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, vol_corr=False, struct_grd=False)

    ##square mesh, horizon studies
    result2_wi_vol_corr = compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, vol_corr=True,  struct_grd=True)
    result2_wo_vol_corr = compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, vol_corr=False, struct_grd=True)
    
    ##triangular mesh, influence function studies
    result3_wi_vol_corr = compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, vol_corr=True,  struct_grd=False)
    result3_wo_vol_corr = compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, vol_corr=False, struct_grd=False)

    ##square mesh, influence function studies
    result4_wi_vol_corr = compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, vol_corr=True,  struct_grd=True)
    result4_wo_vol_corr = compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, vol_corr=False, struct_grd=True)

    return result1_wo_vol_corr, result2_wo_vol_corr, result3_wo_vol_corr, result4_wo_vol_corr, result1_wi_vol_corr, result2_wi_vol_corr, result3_wi_vol_corr, result4_wi_vol_corr

def compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, npts=15, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
    """
    compares the FE and PD solution with varying horizon 
    for a simple 2D case 

    input:
    ------
        horizons : TODO
        npts     : TODO
        material : TODO
        plot_    : TODO
    output:
    -------
        
    """

    dim = mesh_lst[0].topology().dim()
    print("Number of cells in FEniCS mesh on which the FE solution converged %i" %mesh_lst[-1].num_cells())

    ## Empty global lists to store data for each mesh in mesh_lst
    cell_cent_top_lst =      [] #top cell centroids of each mesh
    u_top_fe_conv_lst =      [] #solution on the finest mesh interpolated at top cell centroids of each mesh
    u_disp_PD_array_lst =    [] #displaced cell centroids for each mesh from peridynamic theory
    disp_cent_PD_array_lst = [] #cell centroids after adding displacements to cell centroids in original config
    
    #solve for each mesh in mesh_lst  
    print("*********************************************************")
    print("*********************************************************")
    print('solving using Peridynamics:')
    for curr_mesh in mesh_lst:
        if(struct_grd):
            cell_cent = structured_cell_centroids(curr_mesh)
            el = get_peridym_edge_length(cell_cent, struct_grd)
            print("(STRUCTURED) grid size of eqivalent Peridynamics grid: %i" %len(cell_cent))
        else:
            cell_cent = get_cell_centroids(curr_mesh)
            horizons = np.arange(0.8, 1.6, 0.2)*5.001*curr_mesh.hmax()
            el = get_peridym_edge_length(cell_cent, struct_grd)
            print("(UNSTRUCTURED) grid size of equivalent Peridynamic grid: %i" %len(cell_cent))

        horizons = np.array([0.2400048, 0.300006, 0.3600072, 0.4200048])
        horizons = np.array([0.2400048])
        #declare empty storage for each horizon in 'horizons' array and curr_mesh  
        infl_fun = gaussian_infl_fun2
        disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
        u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)

        for i in range(len(horizons)):
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizons[i], curr_mesh, npts=npts,material=material, omega_fun=infl_fun, plot_=plot_, force=force, vol_corr=vol_corr, struct_grd=struct_grd)

            disp_cent_PD_array[i] = disp_cent_i
            u_disp_PD_array[i]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
        
        #interpolate FE solution from the finest mesh to topmost cell centroids
        # in the curr_mesh
        top_els = np.ravel(np.argwhere(cell_cent[:,1] == np.max(cell_cent[:,1])))
        cell_cent_top = cell_cent[top_els]
        u_top_fe = np.zeros((len(cell_cent_top),dim), dtype=float)
        for i, cc in enumerate(cell_cent_top):
            u_top_fe[i] = u_fe_conv(cc)
       
        # plot the results of the curr_mesh
        colors = get_colors(len(horizons)+1)
        plt.figure()
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], color=colors[0], linewidth=2.0, label='FE Solution')

        colors.pop(0)
        for i in range(len(horizons)):
            u_disp_pd_top = u_disp_PD_array[i][top_els]
            plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], color=colors[i], linewidth=2.0, label='Horizon='+str(horizons[i]))

        plt.legend()
        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title('displacement of top centroids mesh size: %i, el: %4.3f, struc_grd: %s, vol_corr: %s'%(len(cell_cent), el[0], str_struct_grd, str_vol_corr))
        plt.show(block=False)
        plt.savefig('FE_vs_PD_displacements')
        
        # save the data for the curr_mesh to global lists
        cell_cent_top_lst.append(cell_cent_top) 
        u_top_fe_conv_lst.append(u_top_fe)
        u_disp_PD_array_lst.append(u_disp_PD_array)
        disp_cent_PD_array_lst.append(disp_cent_PD_array)

    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst



def compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, npts=15, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
    """
    compares the FE and PD solution for a simple 2D case 
    
    input:
    ------
    horizons : array of peridynamic horizon 
    mesh     : fenics mesh
    npts     : TODO
    material : TODO
    plot_    : TODO
    returns  : TODO

    """
    
    infl_fun_dict = {'omega1':gaussian_infl_fun1,
                      'omega2':gaussian_infl_fun2,
                      'omega3':parabolic_infl_fun1,
                      'omega4':parabolic_infl_fun2}
    infl_fun_dict = {'omega1':gaussian_infl_fun1}

    horizon = 0.2400048
    dim = mesh_lst[0].topology().dim()
    ## Empty global lists to store data for each mesh in mesh_lst
    cell_cent_top_lst =      [] #top cell centroids of each mesh
    u_top_fe_conv_lst =      [] #solution on the finest mesh interpolated at top cell centroids of each mesh
    u_disp_PD_array_lst =    [] #displaced cell centroids for each mesh from peridynamic theory
    disp_cent_PD_array_lst = [] #cell centroids after adding displacements to cell centroids in original config
    
    #solve for each mesh in mesh_lst  
    print("*********************************************************")
    print("*********************************************************")
    print('solving using Peridynamics:')



    ##########################################################################
    for curr_mesh in mesh_lst:
        if(struct_grd):
            cell_cent = structured_cell_centroids(curr_mesh)
            el = get_peridym_edge_length(cell_cent, struct_grd)
            print("(STRUCTURED) grid size of eqivalent Peridynamics grid: %i" %len(cell_cent))
        else:
            cell_cent = get_cell_centroids(curr_mesh)
            el = get_peridym_edge_length(cell_cent, struct_grd)
            print("(UNSTRUCTURED) grid size of equivalent Peridynamic grid: %i" %len(cell_cent))

        extents = compute_modified_extents(cell_cent, el, struct_grd)
        keys = infl_fun_dict.keys()
        disp_cent_PD_array = {} 
        u_disp_PD_array = {}

        print("*********************************************************")
        print("*********************************************************")
        print('solving using Peridynamics:')
        for kk in keys:
            infl_fun = infl_fun_dict[kk]
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizon, curr_mesh, material=material,omega_fun=infl_fun, force=force, vol_corr=vol_corr,struct_grd=struct_grd)

            disp_cent_PD_array[kk] = disp_cent_i
            u_disp_PD_array[kk]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
        
        top_els = np.ravel(np.argwhere(cell_cent[:,1] == np.max(cell_cent[:,1])))
        cell_cent_top = cell_cent[top_els]
        u_top_fe = np.zeros((len(cell_cent_top),dim), dtype=float)
        for i, cc in enumerate(cell_cent_top):
            u_top_fe[i] = u_fe_conv(cc)


        colors = get_colors(len(keys)+1)
        plt.figure()
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, color=colors[0], label='FE Solution')
        colors.pop(0)
        for i, kk in enumerate(keys):
            u_disp_pd_top = u_disp_PD_array[kk][top_els]
            plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], linewidth=2.0, color=colors[i], label='infl_fun: ' + kk)

        plt.legend()
        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title('displacement of top centroids, horizon='+str(horizon)+' struct_grd: '+str_struct_grd + ' vol_corr: '+str_vol_corr )
        plt.show(block=False)
        plt.savefig('FE_vs_PD_displacements_delta'+str(horizon)+".png")

        # save the data for the curr_mesh to global lists
        cell_cent_top_lst.append(cell_cent_top) 
        u_top_fe_conv_lst.append(u_top_fe)
        u_disp_PD_array_lst.append(u_disp_PD_array)
        disp_cent_PD_array_lst.append(disp_cent_PD_array)
    
    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst
