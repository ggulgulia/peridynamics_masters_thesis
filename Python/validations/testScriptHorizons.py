from testHelper import*
from peridynamics_global_test_script import solve_peridynamic_bar_problem  

def plot_for_transverse_load(horizons, abs_error_end_particle_lst, rel_error_end_particle_lst):

    ############## Plot ABS and REL ERRORS #########
    kk = abs_error_end_particle_lst.keys()
    markers = get_markers(len(horizons)+1)
    plt.figure()
    for i, k in enumerate(kk):
        error = abs_error_end_particle_lst[k]
        plt.plot(horizons, error, linewidth=2, marker=markers[i], color='k', markersize=8, label='N = '+str(k))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
    plt.xlim(horizons[0]-horizons[0]/5, horizons[-1] + horizons[-1]/5)
    plt.xlabel('Horizon, $\delta$ [m]', fontsize=14)
    plt.ylabel('abs difference in displacement', fontsize=14)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.legend(loc='center right', fontsize=14)
    plt.title('abs error b/w PD and FE vs Horizon size',  fontsize=16)

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
    plt.show(block=False)

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
    
 
def compare_PD_horizons_with_FE(mesh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=True, force=-5e8, vol_corr=True, struct_grd=False, problem='transverseTraction'):
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

    _, _, _, u_fe_conv, _ = fenics_mesh_convergence(struct_grd=struct_grd, problem=problem, tol=1e-5, plot_=plot_, force=force)

    boundLocationDict = {'transverseTraction': 'top', 'patchTest':'right', 'axialLoad':'right'}
    bName    =           boundLocationDict[problem]
    
    dim = mesh_lst[0].topology().dim()
    ## Empty global lists to store data for each mesh in mesh_lst
    cell_cent_bound_lst =      [] #top cell centroids of each mesh
    u_bnd_fe_conv_lst =      [] #solution on the finest mesh interpolated at top cell centroids of each mesh
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
        # need to select horizon approporiately : min horizon is 2.0001 times max edge length
        #expected particle count in uniform[square/triangle] grid: 648, 900, 1300, 1800
        horizons = np.array([0.11111667555600001, 0.166672235556, 0.22222779555599997, 0.277783355556], dtype=float)
        #horizons = np.array([0.251111667555600001])
        #declare empty storage for each horizon in 'horizons' array and curr_mesh  
        infl_fun = gaussian_infl_fun2
        disp_cent_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)
        u_disp_PD_array = np.zeros((len(horizons), len(cell_cent), dim), dtype=float)


        for i in range(len(horizons)):
            _,_, disp_cent_i, u_disp_i = solve_peridynamic_bar_problem(horizons[i], curr_mesh, material=material, omega_fun=infl_fun, plot_=plot_, force=force, vol_corr=vol_corr, struct_grd=struct_grd, problem=problem)

            disp_cent_PD_array[i] = disp_cent_i
            u_disp_PD_array[i]    = u_disp_i 
            print("*********************************************************")
            print("*********************************************************")
            
        #interpolate FE solution from the finest mesh to topmost cell centroids
        # in the curr_mesh
        u_bnd_fe, cell_cent_bound, boundElIds = interpolate_fe_pd_soln_at_boundary(u_fe_conv, cell_cent, bName=bName)
       
        numParticles = len(cell_cent)
        abs_error_end_particle = np.zeros(len(horizons), dtype=float)
        rel_error_end_particle = np.zeros(len(horizons), dtype=float)
        ###### PLOT PD and FE solns #####
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title(problem+': vol correction = ' +str(vol_corr)+ ' , horizon studies', fontsize=14)

        if problem =='transverseTraction':
            colors = get_colors(len(horizons)+1)
            plt.plot(cell_cent_bound[:,0], u_bnd_fe[:,1], color=colors[0], linewidth=2, label='FE')
            end_cell_y_disp_fe = u_bnd_fe[-1][1]
            colors.pop(0)
            ### collect end cell errors
            for i in range(len(horizons)):
                u_disp_pd_bnd = u_disp_PD_array[i][boundElIds]
                end_cell_y_disp_pd = u_disp_pd_bnd[-1][1]
                abs_error_end_particle[i] = abs(end_cell_y_disp_pd - end_cell_y_disp_fe)
                rel_error_end_particle[i] = abs_error_end_particle[i]/abs(end_cell_y_disp_fe)*100.0
                plt.plot(cell_cent_bound[:,0], u_disp_pd_bnd[:,1], color=colors[i], linewidth=2, label='$\delta:$'+format(horizons[i], '.2E'))
            plt.xlabel('x coordinates of centroids [m]', fontsize=14)
            plt.ylabel('y-displacement [m]', fontsize=14 )
        
        elif problem == 'patchTest' or problem == 'axialLoad':
            markers = get_markers(len(horizons)+1)
            avg_x_disp_fe = np.average(u_bnd_fe[:,0])
            x_avg_loc = np.average(cell_cent_bound[:,0])
            plt.plot(0, avg_x_disp_fe, color='k', marker=markers[0], linewidth=2, label='FE')
            markers.pop(0)

            for i in range(len(horizons)):
                u_disp_pd_bnd = u_disp_PD_array[i][boundElIds]
                avg_x_disp_pd = np.average(u_disp_pd_bnd[:,0])
                abs_error_end_particle[i] = abs(avg_x_disp_fe - avg_x_disp_pd)
                rel_error_end_particle[i] = abs_error_end_particle[i]/abs(avg_x_disp_fe )*100.0
                plt.plot(i+1, avg_x_disp_pd, color='k',marker=markers[0], linewidth=2, label=str(i+1)+':'+r'$\delta:$'+format(horizons[i], '.2E'))
            plt.xticks(np.arange(0, len(horizons)+1,1), fontsize=12)
            plt.xlabel(r'index of $\delta$', fontsize=14)
            plt.ylabel('avg x-displacement [m]', fontsize=12 )


        ## append data to global list
        abs_error_end_particle_lst[numParticles] = abs_error_end_particle
        rel_error_end_particle_lst[numParticles] = rel_error_end_particle

        plt.legend(loc='center left', fontsize=14)
        str_struct_grd = str(bool(struct_grd))
        str_vol_corr   = str(bool(vol_corr))
        plt.title("Num Cells = %i"%numParticles)
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
        plt.ylim( -0.012, 0.002)
        plt.xlim(-0.001, 2.001)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        fig_path = generate_figure_path(data_path, fig_cnt.disp_fig_num, len(cell_cent), 'disp', 'horizon', struct_grd, vol_corr)
        #plt.savefig(fig_path, dpi=dpi)
        plt.show(block=False)
        fig_cnt.disp_fig_num += 1

    ###### PLOT diff b/w PD and FE solns FOR CORRECT LOADING CONDITIONS #####
    plot_for_transverse_load(horizons, abs_error_end_particle_lst, rel_error_end_particle_lst)

    # save the data for the curr_mesh to global lists
    cell_cent_bound_lst.append(cell_cent_bound) 
    u_bnd_fe_conv_lst.append(u_bnd_fe)
    u_disp_PD_array_lst.append(u_disp_PD_array)
    disp_cent_PD_array_lst.append(disp_cent_PD_array)

    return cell_cent_bound_lst, u_bnd_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, abs_error_end_particle_lst, rel_error_end_particle_lst 
