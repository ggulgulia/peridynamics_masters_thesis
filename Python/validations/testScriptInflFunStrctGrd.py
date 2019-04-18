from testHelper import*



def compare_PD_infl_funs_with_FE_StrctGrd(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=True):
    """
    compares the FE and PD solution for a simple 2D case 
    the comparison is carried out at the centerline of the domain
    
    input:
    ------
    horizons : array of peridynamic horizon 
    mesh     : fenics mesh
    material : TODO
    plot_    : TODO
    returns  : TODO

    """


    #infl_fun_lst = [gaussian_infl_fun1, gaussian_infl_fun2,parabolic_infl_fun1,parabolic_infl_fun2]

    """
    Terminology of infl_fun_lst:
        1. omega1 : standard gaussian influence function
        2. omega2 : narrow gaussian influence function
        3. omega3 : standard parabola
        4. omega4 : peridigm like parabola
    """
    infl_fun_ordered_lst = ['omega1', 'omega2', 'omega3', 'omega4']
    infl_fun_lst = {'omega1':gaussian_infl_fun2, 'omega2':gaussian_infl_fun1, 'omega3': parabolic_infl_fun2, 'omega4':parabolic_infl_fun1, }

    infl_fun_keys = infl_fun_lst.keys()
    infl_fun_name = {'omega1':'STANDARD GAUSSIAN', 'omega2': 'NARROW GAUSSIAN', 'omega3': 'STANDARD PARABOLA', 'omega4': 'PERIDIGM PARABOLA'}
    infl_fun_symbols = get_influence_function_symbol()

    horizon = 0.166672235556
    #horizon = 0.2
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
        numParticles = len(cell_cent)

        print("*********************************************************")
        print("*********************************************************")
        print('solving using Peridynamics:')
        for i, key in enumerate(infl_fun_ordered_lst):
            infl_fun = infl_fun_lst[key]
            print("RUNNING TEST WITH INFLUENCE FUNCTION: %s "%(infl_fun_name[key]))
            print("*********************************************************\n")
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar(horizon, curr_mesh, material=material,omega_fun=infl_fun, plot_=plot_, force=force, vol_corr=vol_corr,struct_grd=struct_grd)

            disp_cent_PD_array[i] = disp_cent_i
            u_disp_PD_array[i]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
        
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
        colors = get_colors(len(infl_fun_lst)+1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(cent_X_axis, u_cntline_fe[:,1], color=colors[0], linewidth=2, label='FE')
        plt.title(r"N=%i,vol correction=%s, $\Omega\langle\xi\rangle$ studies"%(numParticles,str(vol_corr)), fontsize=14)
        #compute difference b/w PD and FE at centereline
        abs_error_avg_cntlineY = np.zeros(len(infl_fun_lst), dtype=float)
        rel_error_avg_cntlineY = np.zeros(len(infl_fun_lst), dtype=float)

        colors.pop(0)
        for i, key in enumerate(infl_fun_ordered_lst):
            kk = infl_fun_name[key]
            u_disp_PD_i =  u_disp_PD_array[i]
            for lyr_idx in idx_lst:
                u_cntline_PD += u_disp_PD_i[lyr_idx]
            u_cntline_PD = u_cntline_PD/num_cnt_lyrs
            kk = infl_fun_name[key]
            #compute avg y displacement
            avg_Y_cntline_dispPD = np.average(u_cntline_PD[:,1])
            avg_Y_cntline_dispFE = np.average(u_cntline_fe[:,1])
            abs_error_avg_cntlineY[i] = abs(avg_Y_cntline_dispPD - avg_Y_cntline_dispFE)
            rel_error_avg_cntlineY[i] = abs_error_avg_cntlineY[i]/abs(avg_Y_cntline_dispFE)*100.0
            symbol = infl_fun_symbols[key]
            plt.plot(cent_X_axis, u_cntline_PD[:,1], color=colors[i], linewidth=2, label=symbol)
        abs_error_avg_cntlineY_lst[numParticles] = abs_error_avg_cntlineY
        rel_error_avg_cntlineY_lst[numParticles] = rel_error_avg_cntlineY

        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title('Omega tests, Num Cells = %i, vol_corr= %s'%(numParticles,str_vol_corr), fontsize=20)
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
        plt.ylim( -0.012, 0.002)
        plt.ylabel('y-displacement [m]', fontsize=14)
        plt.xlabel('x-coordinates of particles [m]', fontsize=14)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        plt.legend(loc='center left', fontsize=14)
        #fig_path = generate_figure_path(data_path, fig_cnt.err_fig_num, len(cell_cent), 'disp', 'infl_fun', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)


    ### Some work to make pretty plots ###
    #### Plot diff (error) b/w PD and FE soln ####
    kk = abs_error_avg_cntlineY_lst.keys()
    plt.rc('text', usetex = True)
    markers  = get_markers(len(infl_fun_lst)+1)
    #collecting latex symbols to plot on x-axis
    xtick_labels = []
    for key in infl_fun_ordered_lst:
        xtick_labels.append(infl_fun_symbols[key])
    xtick_labels = np.array(xtick_labels)
    x_ax = np.arange(1, len(infl_fun_lst)+1, 1, dtype=float)

    ### PLOTTING ABS ERROR ###
    ## Note kk is dictonary keys of mesh size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, k in enumerate(kk):
        error = abs_error_avg_cntlineY_lst[k]
        plt.plot(x_ax, error, linewidth=2, marker = markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlim(0, len(infl_fun_lst)+1)
    plt.xlabel('influence function '+r'$\omega_i\langle\xi\rangle$',fontsize=15)
    plt.ylabel('abs difference', fontsize=16)
    ax.set_xticks(x_ax)
    ax.set_xticklabels(xtick_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("omega tests, absloute error, vol corr= %s"%str_vol_corr)
    plt.legend(loc='center right', fontsize=16)
    
    
    ### PLOTTING REL ERROR ###
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## Note kk is dictonary keys of mesh size
    for i, k in enumerate(kk):
        rel_error = rel_error_avg_cntlineY_lst[k]
        plt.plot(x_ax, rel_error, linewidth=2, marker = markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.xlim(0, len(infl_fun_lst)+1)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlabel(r'$\omega_i\langle\xi\rangle$', fontsize=16)
    plt.ylabel('rel difference [%]', fontsize=16)
    ax.set_xticks(x_ax)
    ax.set_xticklabels(xtick_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='center right', fontsize=16)
    #fig_path = generate_figure_path(data_path, fig_cnt.err_fig_num, len(cell_cent), 'err', 'infl_fun', struct_grd, vol_corr)
    plt.title("omega tests, relerror, vol corr= %s"%str_vol_corr)
    plt.show(block=False)
    fig_cnt.err_fig_num += 1

    
    return u_disp_PD_array_lst, abs_error_avg_cntlineY_lst, rel_error_avg_cntlineY_lst 
