from peridynamic_plane_stress import *
from fenics_convergence import *
from peridynamic_infl_fun import *
from datetime import datetime as dttm
from os import path, getcwd, mkdir

err_fig_num = 1;
disp_fig_num= 1;
dpi = 10
axis_font = {'size': str(int(15*dpi))}
title_font = {'size': str(18*dpi)}
legend_size = {'size': str(12*dpi)}
tick_size = 12*dpi
marker_size=100*3.5*dpi

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

def generate_figure_path(fig_path, fig_counter, mesh_size, metric1='err', metric2='horizon', struct_grd=False, vol_corr=False):
    """
    this method generates a suitable/sensible name for the figure to be saved
    NOTE: name here includes the absolute path where the figure is intended to be saved

    fig_path: absolute directory where figure is saved
    fig_counter: some counter to arrange the figure 
    mesh_size: num cells in mesh 
    meric1: string indicating what the polt measures eg : 'err' or 'disp'
    metric2: string indicating what the polt measures eg : 'horizon' or 'infl_fun'
    struct_grd: boolean, 
    vol_corr: boolean 

    output:
    -------
        new_fig_path : absolute path (including the name and format) where 
                       the plot is to be saved

    """
    strct_grd_str = '_wi_str_grd' if(struct_grd) else '_wo_strgrd'
    vol_corr_str  = '_wi_vc.png' if(vol_corr) else '_wo_vc.png'
    new_fig_path = metric1 + '_' + str(fig_counter).zfill(3) + '_'+ metric2 + '_msh'+ str(mesh_size) + strct_grd_str + vol_corr_str 
    new_fig_path = path.join(fig_path, new_fig_path)
    return new_fig_path


def run_comparisons():

    print("*********************************************************")
    print("*********************************************************")
    print('solving using Finite Elements:')
    tol = 1e-5 #tolerance for convergence of FE solution
    _, _, mesh_lst, u_fe_conv = fenics_mesh_convergence(tol=tol, plot_=False)
    print("Number of cells in FEniCS mesh on which the FE solution converged %i" %mesh_lst[-1].num_cells())
    print("*********************************************************")
    print("*********************************************************")

    

    pwd = getcwd()
    today = dttm.now().strftime("%Y%m%d")
    data_dir = path.join(pwd, 'validation_test_on_'+today)
    hori = path.join(data_dir, 'horizon')
    omga = path.join(data_dir, 'inflfun')
    mkdir(data_dir); mkdir(hori); mkdir(omga)
    
    tri_msh_lst = managable_mesh_list(mesh_lst, struct_grd=False)
    #tri_msh_lst = mesh_lst
    ##triangular mesh, horizon studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(tri_msh_lst, u_fe_conv, data_path=hori, vol_corr=False,  struct_grd=False)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(tri_msh_lst, u_fe_conv, data_path=hori, vol_corr=True, struct_grd=False)

    ##triangular mesh, influence function studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(tri_msh_lst, u_fe_conv, data_path=omga, vol_corr=False,  struct_grd=False)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(tri_msh_lst, u_fe_conv, data_path=omga, vol_corr=True, struct_grd=False)

    sqr_msh_lst = managable_mesh_list(mesh_lst, struct_grd=False)
    #sqr_msh_lst = mesh_lst 
    ##square mesh, horizon studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(sqr_msh_lst, u_fe_conv, data_path=hori, vol_corr=False,  struct_grd=True)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(sqr_msh_lst, u_fe_conv, data_path=hori, vol_corr=True, struct_grd=True)
    
    ##square mesh, influence function studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(sqr_msh_lst, u_fe_conv, data_path=omga, vol_corr=False,  struct_grd=True)

    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(sqr_msh_lst, u_fe_conv, data_path=omga, vol_corr=True, struct_grd=True)

    print("SUCCESSFULLY FINISHED THE STUDIES\nGOODLUCK ANALYZING ERRORS FROM THE TON OF FILES THAT HAVE BEEN WRITTEN\n")
    return 


def compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
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

    global disp_fig_num
    global err_fig_num 
    global dpi 
    global axis_font 
    global title_font 
    global legend_size 
    global tick_size 
    global marker_size

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

        #horizons = np.array([0.2400048, 0.300006, 0.3600072, 0.4200048])
        horizons = np.array([0.1, 0.15])
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
       
        ###### PLOT PD and FE solns #####
        colors = get_colors(len(horizons)+1)
        plt.figure(figsize=(240, 128), dpi=dpi)
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], color=colors[0], linewidth=2*dpi, label='FE Solution')
        end_cell_y_disp_fe = u_top_fe[-1][1]

        colors.pop(0)
        error = np.zeros(len(horizons), dtype=float)
        rel_error = np.zeros(len(horizons), dtype=float)
        for i in range(len(horizons)):
            u_disp_pd_top = u_disp_PD_array[i][top_els]
            end_cell_y_disp_pd = u_disp_pd_top[-1][1]
            error[i] = abs(end_cell_y_disp_pd - end_cell_y_disp_fe)
            rel_error[i] = error[i]/abs(end_cell_y_disp_fe)*100.0
            plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], color=colors[i], linewidth=2*dpi, label='Horizon='+str(horizons[i]))

        plt.legend(prop=legend_size)
        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title('displacement of top centroids mesh size: %i, el: %4.3f, struc_grd: %s, vol_corr: %s'%(len(cell_cent), el[0], str_struct_grd, str_vol_corr), **title_font)
        plt.ylim( -0.035, 0.01)
        plt.xlabel('x coordinates of centroids [m]', fontsize=15*dpi, **axis_font)
        plt.ylabel('y-displacement [m]', fontsize=15*dpi, **axis_font)
        plt.xticks(fontsize=10*dpi); plt.yticks(fontsize=10*dpi)
        fig_path = generate_figure_path(data_path, disp_fig_num, len(cell_cent), 'disp', 'horizon', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        disp_fig_num += 1

        ###### PLOT diff b/w PD and FE solns #####
        plt.figure(figsize=(200, 160), dpi=dpi)
        plt.suptitle('displacement of top centroids mesh size: %i, el: %4.3f, struc_grd: %s, vol_corr: %s'%(len(cell_cent), el[0], str_struct_grd, str_vol_corr), **title_font)
        plt.subplot(1,2,1)
        plt.scatter(horizons, error, marker='o', color='b', s=marker_size)
        err_min, err_max = np.min(error), np.max(error)
        plt.ylim(err_min - 0.2*err_min, err_max + 0.2*err_max)
        plt.xlim(horizons[0]-horizons[0]/5, horizons[-1] + horizons[-1]/5)
        plt.xlabel('horizon [m]', fontsize=15*dpi, **axis_font)
        plt.ylabel('abs difference in displacement', fontsize=15*dpi, **axis_font)
        plt.xticks(fontsize=tick_size); plt.yticks(fontsize=tick_size)
        plt.title('abs error b/w PD and FE vs Horizon size', **title_font)

        plt.subplot(1,2,2)
        plt.scatter(horizons, rel_error, marker='o', color='g', s=marker_size)
        plt.ylim(0.0, 20)
        plt.xlim(horizons[0]-horizons[0]/5, horizons[-1] + horizons[-1]/5)
        plt.xlabel('horizon [m]', fontsize= 15*dpi, **axis_font)
        plt.ylabel('rel difference in displacement [%]', fontsize=15*dpi, **axis_font)
        plt.xticks(fontsize=tick_size); plt.yticks(fontsize=tick_size)
        plt.title('rel error(%) b/w PD and FE vs Horizon size', **title_font)
        fig_path = generate_figure_path(data_path, err_fig_num, len(cell_cent), 'err', 'horizon', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        err_fig_num +=1

        # save the data for the curr_mesh to global lists
        cell_cent_top_lst.append(cell_cent_top) 
        u_top_fe_conv_lst.append(u_top_fe)
        u_disp_PD_array_lst.append(u_disp_PD_array)
        disp_cent_PD_array_lst.append(disp_cent_PD_array)

    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst



def compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
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
    global disp_fig_num
    global err_fig_num 
    global dpi 
    global axis_font 
    global title_font 
    global legend_size 
    global tick_size
    global marker_size

    #infl_fun_lst = [gaussian_infl_fun1, gaussian_infl_fun2,parabolic_infl_fun1,parabolic_infl_fun2]
    infl_fun_lst = [gaussian_infl_fun1, gaussian_infl_fun2]
    infl_fun_name = ['narrow gaussian', 'standard gaussian', 'peridigm parabola', 'standard parabola']

    #horizon = 0.2400048
    horizon = 0.1
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
            print("RUNNING TEST WITH INFLUENCE FUNCTION: %s "%(infl_fun_name[i]))
            print("*********************************************************\n")
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


        error = np.zeros(len(infl_fun_lst), dtype=float)
        rel_error = np.zeros(len(infl_fun_lst), dtype=float)
        colors = get_colors(len(infl_fun_lst)+1)
        plt.figure(figsize=(240, 180), dpi=dpi)
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, color=colors[0], label='FE Solution')
        end_cell_y_disp_fe = u_top_fe[-1][1]
        colors.pop(0)
        for i in range(len(infl_fun_lst)):
            kk = infl_fun_name[i]
            u_disp_pd_top = u_disp_PD_array[i][top_els]
            end_cell_y_disp_pd = u_disp_pd_top[-1][1]
            error[i] = abs(end_cell_y_disp_pd - end_cell_y_disp_fe)
            rel_error[i] = error[i]/abs(end_cell_y_disp_fe)*100.00
            plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], linewidth=2*dpi, color=colors[i], label='infl_fun: ' + kk)

        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title('displacement of top centroids mesh size: %i, el: %4.3f, horizon: %4.3f, struc_grd: %s, vol_corr: %s'%(len(cell_cent), el[0], horizon, str_struct_grd, str_vol_corr), **title_font)
        plt.ylim( -0.035, 0.01)
        plt.ylabel('y-displacement [m]', fontsize=15*dpi, **axis_font)
        plt.xlabel('x-coordinates of particles [m]', fontsize=15*dpi, **axis_font)
        plt.xticks(fontsize=tick_size); plt.yticks(fontsize=tick_size)
        plt.legend(prop = legend_size)
        fig_path = generate_figure_path(data_path, err_fig_num, len(cell_cent), 'disp', 'infl_fun', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        disp_fig_num += 1


        #### Plot diff (error) b/w PD and FE soln ####
        plt.figure(figsize=(240, 180), dpi=dpi)
        plt.suptitle('displacement of top centroids mesh size: %i, el: %4.3f, horizon: %4.3f, struc_grd: %s, vol_corr: %s'%(len(cell_cent), el[0], horizon, str_struct_grd, str_vol_corr))
        x_ax = np.arange(1, len(infl_fun_lst)+1, 1, dtype=float)
        plt.subplot(1,2,1)
        for i in range(len(infl_fun_lst)):
                kk = infl_fun_name[i]
                plt.scatter(x_ax[i], error[i], marker='o', color=colors[i], s=marker_size, label='omega: ' + kk)
        err_min, err_max = np.min(error), np.max(error)
        plt.ylim(err_min - 0.2*err_min,err_max + 0.2*err_max)
        plt.xlim(0, len(infl_fun_lst)+1)
        plt.xlabel('omega', **axis_font)
        plt.ylabel('abs difference', **axis_font)
        plt.xticks(fontsize=tick_size); plt.yticks(fontsize=tick_size)
        plt.legend(prop=legend_size)
        plt.title('abs difference b/w PD and FE vs Horizon size', **title_font)

        plt.subplot(1,2,2)
        for i in range(len(infl_fun_lst)):
                kk = infl_fun_name[i]
                plt.scatter(x_ax[i], rel_error[i], marker='o', color=colors[i], s=marker_size, label='omega: ' + kk)
        plt.ylim(np.min(rel_error)-5, np.max(rel_error)+5)
        plt.xlim(0, len(infl_fun_lst)+1)
        plt.xlabel('omega', fontsize=15*dpi, **axis_font)
        plt.ylabel('rel difference [%]', fontsize=15*dpi,**axis_font)
        plt.xticks(fontsize=tick_size); plt.yticks(fontsize=tick_size)
        plt.legend(prop=legend_size)
        plt.title('Rel difference b/w PD and FE soln vs Horizon size', **title_font)
        fig_path = generate_figure_path(data_path, err_fig_num, len(cell_cent), 'err', 'infl_fun', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        err_fig_num += 1

        # save the data for the curr_mesh to global lists
        cell_cent_top_lst.append(cell_cent_top) 
        u_top_fe_conv_lst.append(u_top_fe)
        u_disp_PD_array_lst.append(u_disp_PD_array)
        disp_cent_PD_array_lst.append(disp_cent_PD_array)
    
    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst
