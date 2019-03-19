from testHelper import*

def compare_PD_material_models(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
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
    plotting_param_dict = global_plotting_parameters()
    dpi         = plotting_param_dict['dpi'] 
    axis_font   = plotting_param_dict['axis_font']
    title_font  = plotting_param_dict['title_font']
    legend_size = plotting_param_dict['legend_size']
    tick_size   = plotting_param_dict['tick_size']
    marker_size = plotting_param_dict['marker_size']

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
