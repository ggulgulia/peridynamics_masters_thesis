from testHelper import*

def compare_PD_material_models(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
    """
    compares the FE and PD solution with varying horizon 
    for a simple 2D case 

    Note: Default settings in the script is to call correspondance material model
    To change to response=LPS in the appropriate line 

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
    abs_error_end_particle_lst = {}
    rel_error_end_particle_lst = {}
    
    #solve for each mesh in mesh_lst  
    print("*********************************************************")
    print("*********************************************************")
    print('solving using Peridynamics:')

    for curr_mesh in mesh_lst:
        if(struct_grd):
            cell_cent = structured_cell_centroids(curr_mesh)
            print("(STRUCTURED) grid size of eqivalent Peridynamics grid: %i" %len(cell_cent))
        else:
            cell_cent = get_cell_centroids(curr_mesh)
            #horizons = np.arange(0.8, 1.6, 0.2)*5.001*curr_mesh.hmax()
            print("(UNSTRUCTURED) grid size of equivalent Peridynamic grid: %i" %len(cell_cent))

        el = get_peridym_edge_length(cell_cent, struct_grd)

        #edge lengths are currently [0.05555556, 0.04545455, 0.03846154, 0.03333333, 0.02941176]
        # need to select horizon approporiately : min horizon is 2.0001 times max edge length in the coarsest grid
        #expected particle count in uniform[square/triangle] grid: 648, 900, 1300, 1800
        horizons = np.array([0.11111667555600001, 0.166672235556, 0.22222779555599997, 0.277783355556], dtype=float)
        #horizons = np.array([0.0611111667555600001])
        #declare empty storage for each horizon in 'horizons' array and curr_mesh  
        infl_fun = gaussian_infl_fun2
        disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
        u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)

        for i in range(len(horizons)):
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizons[i], curr_mesh, material=material, omega_fun=infl_fun, plot_=False, force=force, vol_corr=vol_corr, struct_grd=struct_grd, response='correspondance')

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
       
        numParticles = len(cell_cent)
        ###### PLOT PD and FE solns #####
        colors = get_colors(len(horizons)+1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], color=colors[0], linewidth=2, label='FE')
        end_cell_y_disp_fe = u_top_fe[-1][1]

        colors.pop(0)

        ### collect end cell errors
        abs_error_end_particle = np.zeros(len(horizons), dtype=float)
        rel_error_end_particle = np.zeros(len(horizons), dtype=float)
        for i in range(len(horizons)):
            u_disp_pd_top = u_disp_PD_array[i][top_els]
            end_cell_y_disp_pd = u_disp_pd_top[-1][1]
            abs_error_end_particle[i] = abs(end_cell_y_disp_pd - end_cell_y_disp_fe)
            rel_error_end_particle[i] = abs_error_end_particle[i]/abs(end_cell_y_disp_fe)*100.0
            plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], color=colors[i], linewidth=2, label='$\delta:$'+format(horizons[i], '.2E'))


        ## append data to global list
        abs_error_end_particle_lst[numParticles] = abs_error_end_particle
        rel_error_end_particle_lst[numParticles] = rel_error_end_particle

        plt.legend(loc='lower left', fancybox=True, framealpha=0.0, fontsize=14)
        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title("Horizon studies, Num Cells = %i vol_corr=%s, vol"%(numParticles,str_vol_corr))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
        plt.ylim( -0.012, 0.002)
        plt.xlim(-0.001, 2.001)
        plt.xlabel('x coordinates of centroids [m]', fontsize=18)
        plt.ylabel('y-displacement [m]', fontsize=18 )
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        #fig_path = generate_figure_path(data_path, fig_cnt.disp_fig_num, len(cell_cent), 'disp', 'horizon', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        fig_cnt.disp_fig_num += 1

        ###### PLOT diff b/w PD and FE solns #####
        #plt.suptitle('displacement of top centroids mesh size: %i, el: %4.3f, struc_grd: %s, vol_corr: %s'%(len(cell_cent), el[0], str_struct_grd, str_vol_corr), **title_font)
        #plt.subplot(1,2,1)

    kk = abs_error_end_particle_lst.keys()

    markers = get_markers(len(horizons)+1)
    colors = get_colors(len(horizons)+1)

    ############## Plot ABS and REL ERRORS #########
    plt.figure()
    for i, k in enumerate(kk):
        error = abs_error_end_particle_lst[k]
        plt.plot(horizons, error, linewidth=2, marker=markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlim(horizons[0]-horizons[0]/5, horizons[-1] + horizons[-1]/5)
    plt.xlabel('Horizon, $\delta$ [m]', fontsize=16)
    plt.ylabel('abs difference in displacement', fontsize=16)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.legend(loc='center right', fontsize=14)
    plt.title('abs error b/w PD and FE vs Horizon size',  fontsize=18)

    plt.figure()
    for i, k in enumerate(kk):
        rel_error = rel_error_end_particle_lst[k]
        plt.plot(horizons, rel_error, linewidth=2, marker=markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlim(horizons[0]-horizons[0]/5, horizons[-1] + horizons[-1]/5)
    plt.xlabel('Horizon $\delta$ [m]', fontsize= 16)
    plt.ylabel('rel difference in displacement [%]', fontsize=16)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.legend(loc='center right', fontsize=14)
    plt.title('rel error b/w PD and FE vs Horizon size', fontsize=18)
    #fig_path = generate_figure_path(data_path, fig_cnt.err_fig_num, len(cell_cent), 'err', 'horizon', struct_grd, vol_corr)
    #plt.savefig(fig_path, dpi=dpi)
    plt.show(block=False)
    fig_cnt.err_fig_num +=1

    # save the data for the curr_mesh to global lists
    cell_cent_top_lst.append(cell_cent_top) 
    u_top_fe_conv_lst.append(u_top_fe)
    u_disp_PD_array_lst.append(u_disp_PD_array)
    disp_cent_PD_array_lst.append(disp_cent_PD_array)

    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, abs_error_end_particle_lst, rel_error_end_particle_lst 
