from testHelper import*
from peridynamics_global_test_script import solve_peridynamic_bar_problem

def interpolate_fe_pd_soln_at_boundary(u_fe, cell_cent, bName='top'):
    """
    depending on the problem we are solving, we might want to obtain the
    PD solution at the  boundary particle location of 
    a given peridynamic discretization. for this purpose the
    method finds the appropriate boudnary

    NOTE: the method works only for 2d  rectangular geometry 
    and is done to make my life easier while doing tests 
    my masters for thesis

    :u_disp_pd: PD solution at nodal locations(from FENICS)
    :bName: name of the boundary where solution is desired
    :cell_cent: peridynamic discretization of domain 
    :returns: u_fe_atBoundary

    """
    #cardinal search dimension for boundary locations
    ## for left/right we look along x-dim(index 0)
    ## for top/bottom we look along y-dim(index 1)
    searchDimDict = {'top':1,'bottom':1, 'left':0, 'right':0}
    sd = searchDimDict[bName]

    if bName == 'top' or bName == 'right':
        boundElIds= np.ravel((np.argwhere(cell_cent[:,sd] == np.max(cell_cent[:,sd]))))
    if bName == 'bottom' or bName == 'left':
        boundElIds= np.ravel((np.argwhere(cell_cent[:,sd] == np.min(cell_cent[:,sd]))))

    cell_cent_bound = cell_cent[boundElIds]
   
    #place holder for interpolated fe solution at boundary of 
    # corresponding peridynamic mesh
    u_fe_bnd_cent = np.zeros((len(cell_cent_bound), 2), dtype=float)
    for idx, cc in enumerate(cell_cent_bound):
        u_fe_bnd_cent[idx] = u_fe(cc)
        
    return u_fe_bnd_cent, cell_cent_bound, boundElIds

def compare_PD_infl_funs_with_FE(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=False, problem='transverseTraction'):
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
    _, _, _, u_fe_conv, _ = fenics_mesh_convergence(struct_grd=struct_grd, problem=problem, tol=1e-5, plot_=plot_, force=force)

    boundLocationDict = {'transverseTraction': 'top', 'patchTest':'right', 'axialLoad':'right'}
    bName    =           boundLocationDict[problem]

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

    #infl_fun_ordered_lst = ['omega1']
    #infl_fun_lst = {'omega1':gaussian_infl_fun2}

    infl_fun_keys = infl_fun_lst.keys()
    infl_fun_name = {'omega1':'STANDARD GAUSSIAN', 'omega2': 'NARROW GAUSSIAN', 'omega3': 'STANDARD PARABOLA', 'omega4': 'PERIDIGM PARABOLA'}
    infl_fun_symbols = get_influence_function_symbol()

    horizon = 0.2222278
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
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar_problem(horizon, curr_mesh, material=material,omega_fun=infl_fun, plot_=plot_, force=force, vol_corr=vol_corr,struct_grd=struct_grd, problem=problem)

            disp_cent_PD_array[i] = disp_cent_i
            u_disp_PD_array[i]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
        
        top_els = np.ravel(np.argwhere(cell_cent[:,1] == np.max(cell_cent[:,1])))
        cell_cent_top = cell_cent[top_els]
        u_top_fe = np.zeros((len(cell_cent_top),dim), dtype=float)
        for i, cc in enumerate(cell_cent_top):
            u_top_fe[i] = u_fe_conv(cc)

        ### Store errors in a dictonary ##
        abs_error_end_particle = np.zeros(len(infl_fun_lst), dtype=float)
        rel_error_end_particle = np.zeros(len(infl_fun_lst), dtype=float)
    
        colors = get_colors(len(infl_fun_lst)+1)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ###### PLOT FE Soln ####
        plt.plot(cell_cent_top[:,0], u_top_fe[:,1], linewidth=2.0, color=colors[0], label='FE')
        end_cell_y_disp_fe = u_top_fe[-1][1]
        colors.pop(0)
        for i, key in enumerate(infl_fun_ordered_lst):
            kk = infl_fun_name[key]
            u_disp_pd_top = u_disp_PD_array[i][top_els]
            end_cell_y_disp_pd = u_disp_pd_top[-1][1]
            abs_error_end_particle[i] = abs(end_cell_y_disp_pd - end_cell_y_disp_fe)
            rel_error_end_particle[i] = abs_error_end_particle[i]/abs(end_cell_y_disp_fe)*100.00
            #### PLOT PD SOLN ####
            symbol = infl_fun_symbols[key]
            plt.plot(cell_cent_top[:,0], u_disp_pd_top[:,1], linewidth=2, color=colors[i], label=symbol)

        ###### STORE ERRORS ##### 
        abs_error_end_particle_lst[numParticles] = abs_error_end_particle
        rel_error_end_particle_lst[numParticles] = rel_error_end_particle

        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        #plt.title("Omega tests, Num Cells = %i, vol_corr= %s"%(int(numParticles),str_vol_corr), fontsize=16)
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
        plt.ylim( -0.012, 0.002)
        plt.ylabel('y-displacement', fontsize=14)
        plt.xlabel('x-coordinates of particles strct grd:'+str(struct_grd), fontsize=14)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        ax.legend(fancybox=True, framealpha=0.0)
        plt.legend(loc='lower left', title='N = '+str(numParticles), fontsize=16)
        fig_path = generate_figure_path(data_path, fig_cnt.err_fig_num, len(cell_cent), 'disp', 'infl_fun', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        fig_cnt.disp_fig_num += 1


    ### Some work to make pretty plots ###
    #### Plot diff (error) b/w PD and FE soln ####
    kk = abs_error_end_particle_lst.keys()
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
        error = abs_error_end_particle_lst[k]
        plt.plot(x_ax, error, linewidth=2, marker = markers[i], color='k', markersize=8, label='N = '+str(k))
    #err_min, err_max = np.min(error), np.max(error)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlim(0, len(infl_fun_lst)+1)
    plt.xlabel('influence function '+r'$\omega_i\langle\xi\rangle$',fontsize=15)
    plt.ylabel('absolute error [m]', fontsize=14)
    ax.set_xticks(x_ax)
    ax.set_xticklabels(xtick_labels, fontsize=14)
    ax.legend(fancybox=True, framealpha=0.0, loc='upper right', fontsize=12)
    plt.yticks(fontsize=14)
    plt.title("omega tests, absloute error, vol corr= %s"%str_vol_corr)
    
    
    ### PLOTTING REL ERROR ###
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## Note kk is dictonary keys of mesh size
    for i, k in enumerate(kk):
        rel_error = rel_error_end_particle_lst[k]
        plt.plot(x_ax, rel_error, linewidth=2, marker = markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.xlim(0, len(infl_fun_lst)+1)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlabel(r'$\omega_i\langle\xi\rangle$', fontsize=16)
    plt.ylabel('relative error [%]', fontsize=14)
    ax.set_xticks(x_ax)
    ax.set_xticklabels(xtick_labels, fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(fancybox=True, framealpha=0.0, loc='upper right', fontsize=12)
    #plt.title("omega tests, relerror, vol corr= %s"%str_vol_corr)
    plt.show(block=False)
#    fig_cnt.err_fig_num += 1

    
    # save the data for the curr_mesh to global lists
    cell_cent_top_lst.append(cell_cent_top) 
    u_top_fe_conv_lst.append(u_top_fe)
    u_disp_PD_array_lst.append(u_disp_PD_array)
    disp_cent_PD_array_lst.append(disp_cent_PD_array)
    
    return cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, abs_error_end_particle_lst, rel_error_end_particle_lst
