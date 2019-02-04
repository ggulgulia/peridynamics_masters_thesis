from peridynamic_plane_stress import *
from fenics_convergence import *
from peridynamic_infl_fun import *
from os import path, getcwd

def write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, file_path, file_name):
    """
    writes our data of interest to the csv file
    """
    import csv
    def _write_(abs_file_path, arr):
        with open(abs_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerows(arr)


    num_data = len(cell_cent_top_lst)
    for i in range(num_data):
        cel_cnt_file_name = path.join(file_path, file_name + str(i).zfill(2) + '_cel_cnt_top.csv')
        ufe_file_name =     path.join(file_path, file_name + str(i).zfill(2) + '_ufe_top.csv')
        dsc_file_name =     path.join(file_path, file_name + str(i).zfill(2) + '_dsp_cnt.csv')
        u_dsp_file_name =   path.join(file_path, file_name + str(i).zfill(2) + '_u_dsp.csv')

        _write_(cel_cnt_file_name, cell_cent_top_lst[i])
        _write_(ufe_file_name, u_top_fe_conv_lst[i])
        _write_(dsc_file_name, disp_cent_PD_array_lst[i])
        _write_(u_dsp_file_name, u_disp_PD_array_lst[i])

    return


def managable_mesh_list(mesh_lst, struct_grd=False):
    """
    CREATE A USEFUL MESH LIST OF MANAGABLE SIZE              
    
    Step1:
    ------
    remove very coarse meshes since they are
    anywy useless for peridynamic computations

    Step2:
    we don't want to wait forever for computations
    on this serial code so we slice the list to a managable size 
    
    """
    #step1
    grd_fact = (1+int(struct_grd))
    slice_idx = 0
    for idx, mm in enumerate(mesh_lst):
        num_cells = mm.num_cells()
        if(int(num_cells/grd_fact) <1600):
            print("removing the mesh at index %i due to low cell count (%i) for peridynamic calculations"%(idx, int(num_cells/grd_fact)))
            slice_idx = idx

    mesh_lst = mesh_lst[slice_idx+1:]
            
    #Step2
    if(len(mesh_lst)> 5):
        print("Too many meshes in the list, resizing to managable size")
    return mesh_lst[0:3]
    


def run_comparisons():

    from datetime import datetime as dttm
    print("*********************************************************")
    print("*********************************************************")
    print('solving using Finite Elements:')
    tol = 1e-5#tolerance for convergence of FE solution
    _, _, mesh_lst, u_fe_conv = fenics_mesh_convergence(tol=tol, plot_=False)
    print("Number of cells in FEniCS mesh on which the FE solution converged %i" %mesh_lst[-1].num_cells())
    print("*********************************************************")
    print("*********************************************************")

    

    pwd = getcwd()
    hori = path.join(getcwd(), 'horizon')
    omga = path.join(getcwd(), 'inflfun')
    
    tri_msh_lst = managable_mesh_list(mesh_lst, struct_grd=False)
    ##triangular mesh, horizon studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(tri_msh_lst, u_fe_conv, vol_corr=False,  struct_grd=False)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('hrz_wo_srct_gr_wo_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, hori, file_name)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(tri_msh_lst, u_fe_conv, vol_corr=True, struct_grd=False)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('hrz_wo_srct_gr_wi_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, hori, file_name)

    ##triangular mesh, influence function studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(tri_msh_lst, u_fe_conv, vol_corr=False,  struct_grd=False)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('omg_wo_srct_gr_wo_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, omga, file_name)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(tri_msh_lst, u_fe_conv, vol_corr=True, struct_grd=False)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('omg_wo_srct_gr_wi_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, omga, file_name)

    sqr_msh_lst = managable_mesh_list(mesh_lst, struct_grd=False)
    ##square mesh, horizon studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(sqr_msh_lst, u_fe_conv, vol_corr=False,  struct_grd=True)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('hrz_wi_srct_gr_wo_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, hori, file_name)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(sqr_msh_lst, u_fe_conv, vol_corr=True, struct_grd=True)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('hrz_wi_srct_gr_wi_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, hori, file_name)
    
    ##square mesh, influence function studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(sqr_msh_lst, u_fe_conv, vol_corr=False,  struct_grd=True)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('omg_wi_srct_gr_wo_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, omga, file_name)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(sqr_msh_lst, u_fe_conv, vol_corr=True, struct_grd=True)
    now = dttm.now().strftime("%Y%m%d%H%M%S%f")
    file_name = now + str('omg_wi_srct_gr_wi_vol_corr')
    write_data_to_csv(cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, omga, file_name)

    print("SUCCESSFULLY FINISHED THE STUDIES\nGOODLUCK ANALYZING ERRORS FROM THE TON OF FILES THAT HAVE BEEN WRITTEN\n")
    return 


def compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
    """
    compares the FE and PD solution with varying horizon 
    for a simple 2D case 

    input:
    ------
        mesh_lst : TODO
        material : TODO
        plot_    : TODO
    output:
    -------
        
    """

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
        #declare empty storage for each horizon in 'horizons' array and curr_mesh  
        infl_fun = gaussian_infl_fun2
        disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
        u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)

        for i in range(len(horizons)):
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizons[i], curr_mesh, material=material, omega_fun=infl_fun, plot_=plot_, force=force, vol_corr=vol_corr, struct_grd=struct_grd)

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



def compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
    """
    compares the FE and PD solution for a simple 2D case 
    
    input:
    ------
    horizons : array of peridynamic horizon 
    mesh     : fenics mesh
    material : TODO
    plot_    : TODO
    returns  : TODO

    """
    
    #infl_fun_dict = {'omega1':gaussian_infl_fun1,
    #                  'omega2':gaussian_infl_fun2,
    #                  'omega3':parabolic_infl_fun1,
    #                  'omega4':parabolic_infl_fun2}
    infl_fun_lst = [gaussian_infl_fun1, gaussian_infl_fun2, parabolic_infl_fun1, parabolic_infl_fun2]
    infl_fun_name = ['narrow gaussian', 'standard gaussian', 'peridigm parabola' 'standard parabola']

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
        disp_cent_PD_array = np.zeros((len(infl_fun_lst), len(cell_cent), dim), dtype=float)
        u_disp_PD_array = np.zeros((len(infl_fun_lst), len(cell_cent), dim), dtype=float)

        print("*********************************************************")
        print("*********************************************************")
        print('solving using Peridynamics:')
        for i, infl_fun in enumerate(infl_fun_lst):
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizon, curr_mesh, material=material,omega_fun=infl_fun, force=force, vol_corr=vol_corr,struct_grd=struct_grd)

            disp_cent_PD_array[i] = disp_cent_i
            u_disp_PD_array[i]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
        
        top_els = np.ravel(np.argwhere(cell_cent[:,1] == np.max(cell_cent[:,1])))
        cell_cent_top = cell_cent[top_els]
        u_top_fe = np.zeros((len(cell_cent_top),dim), dtype=float)
        for i, cc in enumerate(cell_cent_top):
            u_top_fe[i] = u_fe_conv(cc)


        colors = get_colors(len(infl_fun_lst)+1)
        plt.figure()
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, color=colors[0], label='FE Solution')
        colors.pop(0)
        for i in range(len(infl_fun_lst)):
            kk = infl_fun_name[i]
            u_disp_pd_top = u_disp_PD_array[i][top_els]
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
