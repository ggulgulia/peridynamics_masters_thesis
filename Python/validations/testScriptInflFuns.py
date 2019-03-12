from testHelper import*

dpi = 10
axis_font = {'size': str(int(15*dpi))}
title_font = {'size': str(18*dpi)}
legend_size = {'size': str(12*dpi)}
tick_size = 12*dpi
marker_size=100*3.5*dpi


def compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
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

    plotting_param_dict = global_plotting_parameters()
    dpi         = plotting_param_dict['dpi']  
    axis_font   = plotting_param_dict['axis_font']  
    title_font  = plotting_param_dict['title_font'] 
    legend_size = plotting_param_dict['legend_size']  
    tick_size   = plotting_param_dict['tick_size'] 
    marker_size = plotting_param_dict['marker_size'] 

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
        fig_path = generate_figure_path(data_path, fig_cnt.err_fig_num, len(cell_cent), 'disp', 'infl_fun', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        fig_cnt.disp_fig_num += 1


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
        fig_path = generate_figure_path(data_path, fig_cnt.err_fig_num, len(cell_cent), 'err', 'infl_fun', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        fig_cnt.err_fig_num += 1
        
        # save the data for the curr_mesh to global lists
        cell_cent_top_lst.append(cell_cent_top) 
        u_top_fe_conv_lst.append(u_top_fe)
        u_disp_PD_array_lst.append(u_disp_PD_array)
        disp_cent_PD_array_lst.append(disp_cent_PD_array)
    
    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst
