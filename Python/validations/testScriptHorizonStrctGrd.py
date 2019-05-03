from testHelper import*


def compare_PD_horizons_with_FE_StrctGrd(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False):
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
    centerline_cells_lst =   [] #top cell centroids of each mesh
    u_cntline_fe_conv_lst =  [] #solution on the finest mesh interpolated at top cell centroids of each mesh
    u_disp_PD_array_lst =    {} #displaced cell centroids for each mesh from peridynamic theory
    disp_cent_PD_array_lst = [] #cell centroids after adding displacements to cell centroids in original config
    abs_error_end_particle_lst = {}
    rel_error_end_particle_lst = {}
    abs_error_avg_cntlineY_lst = {}
    rel_error_avg_cntlineY_lst = {}
    
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

        #edge lengths are currently [0.05555556, 0.04545455, 0.03846154, 0.03333333, 0.02941176]
        # need to select horizon approporiately : min horizon is 2.0001 times max edge length
        #expected particle count in uniform[square/triangle] grid: 648, 900, 1300, 1800
        horizons = np.array([0.11111667555600001, 0.166672235556, 0.22222779555599997, 0.277783355556], dtype=float)
        #horizons = np.array([0.20611111667555600001, 0.3001])
        #declare empty storage for each horizon in 'horizons' array and curr_mesh  
        infl_fun = gaussian_infl_fun2
        disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
        u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)

        numParticles = len(cell_cent)

        for i in range(len(horizons)):
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizons[i], curr_mesh, material=material, omega_fun=infl_fun, plot_=False, force=force, vol_corr=vol_corr, struct_grd=struct_grd)

            disp_cent_PD_array[i] = disp_cent_i
            u_disp_PD_array[i]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
        u_disp_PD_array_lst[numParticles] = u_disp_PD_array
        

         ###### POST PROCESSING #######
        ### computation to get domain range where we want to monitor results ###
        el = get_peridym_edge_length(cell_cent, struct_grd)
        obs_extents = get_obs_extent(el, curr_mesh)
        centerline_cells, idx, cent_X_axis, cent_Y_axis = get_centerline_cells_and_idx(cell_cent, obs_extents)
        cent_Y_lyrs, idx_lst = separate_centrline_lyers_by_y_coordinates(cell_cent, centerline_cells)

        num_cnt_lyrs = len(cent_Y_lyrs)
        u_cntline_fe = np.zeros((len(cent_Y_lyrs[0]), dim), dtype=float) #assume the particles in each layers are equal
        u_cntline_PD = np.zeros((len(cent_Y_lyrs[0]), dim), dtype=float) #assume the particles in each layers are equal
        for i in range(num_cnt_lyrs):
            for j, cc in enumerate(cent_Y_lyrs[i]):
                 u_cntline_fe[j] += u_fe_conv(cc)
        u_cntline_fe = u_cntline_fe/num_cnt_lyrs

        ###### PLOT PD and FE solns along centerline #####
        colors = get_colors(len(horizons)+1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(cent_X_axis, u_cntline_fe[:,1], color=colors[0], linewidth=2, label='FE')
        plt.title("N=%i,vol correction=%s, horizon studies"%(numParticles,str(vol_corr)), fontsize=14)
        #compute difference b/w PD and FE at centereline
        abs_error_avg_cntlineY = np.zeros(len(horizons), dtype=float)
        rel_error_avg_cntlineY = np.zeros(len(horizons), dtype=float)

        colors.pop(0)
        for i in range(len(horizons)):
            u_disp_PD_i =  u_disp_PD_array[i]
            for lyr_idx in idx_lst:
                u_cntline_PD += u_disp_PD_i[lyr_idx]

            u_cntline_PD = u_cntline_PD/num_cnt_lyrs
            #compute avg y displacement
            avg_Y_cntline_dispPD = np.average(u_cntline_PD[:,1])
            avg_Y_cntline_dispFE = np.average(u_cntline_fe[:,1])
            abs_error_avg_cntlineY[i] = abs(avg_Y_cntline_dispPD - avg_Y_cntline_dispFE)
            rel_error_avg_cntlineY[i] = abs_error_avg_cntlineY[i]/abs(avg_Y_cntline_dispFE)*100.0
            plt.plot(cent_X_axis, u_cntline_PD[:,1], color=colors[i], linewidth=2, label='$\delta:$'+format(horizons[i], '.2E'))
        abs_error_avg_cntlineY_lst[numParticles] = abs_error_avg_cntlineY
        rel_error_avg_cntlineY_lst[numParticles] = rel_error_avg_cntlineY

        plt.xlabel('x-coordinates of particles [m]', fontsize=16)
        plt.ylabel('y-displacement [m]', fontsize=16)
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
        plt.xticks(fontsize=10); plt.yticks(fontsize=10)
        plt.legend(loc='lower left', fontsize=12)


    kk = abs_error_avg_cntlineY_lst.keys()
    markers= get_markers(len(horizons)+1)
    ############## Plot ABS and REL ERRORS #########
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, k in enumerate(kk):
        error = abs_error_avg_cntlineY_lst[k]
        plt.plot(horizons, error, linewidth=2, marker=markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.xlabel('Horizon, $\delta$ [m]', fontsize=16)
    plt.ylabel('abs difference in displacement', fontsize=16)
    ax.set_xticks(horizons)
    ax.set_xticklabels(horizons, fontsize=12)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.legend(loc='upper left',fancybox=True, framealpha=0.0, fontsize=12)
    plt.title('abs error b/w PD and FE vs Horizon size',  fontsize=18)

    plt.show(block=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, k in enumerate(kk):
        rel_error = rel_error_avg_cntlineY_lst[k]
        plt.plot(horizons, rel_error, linewidth=2, marker=markers[i], color='k', markersize=8, label='N = '+str(k))
    fig_cnt.err_fig_num +=1
    plt.xlabel('Horizon, $\delta$ [m]', fontsize=16)
    plt.ylabel('rel difference in displacement', fontsize=16)
    ax.set_xticks(horizons)
    ax.set_xticklabels(horizons, fontsize=12)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0E'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.legend(loc='upper left',fancybox=True, framealpha=0.0, fontsize=12)
    plt.title('rel error b/w PD and FE vs Horizon size',  fontsize=18)

    # save the data for the curr_mesh to global lists

    return u_disp_PD_array_lst, abs_error_avg_cntlineY_lst, rel_error_avg_cntlineY_lst 




    ### to get additional legends
    #ax.legend(fancybox=True, framealpha=0.5)
    #h = [plt.plot([],[], color=colors[i], marker=markers[i], ms=8, ls="")[0] for i in range(5)]
    #leg1 = plt.legend(handles=h, labels=[648, 968, 1352, 1800, 2312],loc='upper right', title="Particle Count", fontsize=10)
    #ax.add_artist(leg1)

    #h2 = [plt.plot([],[], color='k', linewidth=1.5, ms=8, ls=ls[i])[0] for i in range(2)]
    #plt.legend(handles=h2, labels=['w/i Volume Correction', 'w/o Volume Correction'],loc='lower center', fontsize=10)
    #ax.add_artist(leg2)
