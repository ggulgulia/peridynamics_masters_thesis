from peridynamic_patch_test import solve_peridynamic_patch_test
from peridynamic_plane_stress import solve_peridynamic_bar_transverse
from peridynamic_axial_stress import solve_peridynamic_bar_axial

def solve_peridynamic_bar_problem(horizon, m=None, nbr_lst=None, nbr_beta_lst=None, material='steel', omega_fun=None, plot_=False, force=-5e8, vol_corr=True, struct_grd=False, response='LPS', problem='transverseTraction'):
#def solve_peridynamic_test_problem(problem='transverseTraction'):
    """
    global file that calls other appropriate test scripts

    :problem: TODO
    :returns: TODO

    """

    pd_solution_method = {'patchTest': solve_peridynamic_patch_test, 'transverseTraction': solve_peridynamic_bar_transverse, 'axialLoad':solve_peridynamic_bar_axial}
   
    boundLocationDict = {'transverseTraction': 'top', 'patchTest':'right', 'axialLoad':'center'}

    solve_pd = pd_solution_method[problem]
    
    K, K_bound, disp_cent, u_disp = solve_pd(horizon, m=m, nbr_lst=nbr_lst, nbr_beta_lst=nbr_beta_lst, material=material, omega_fun = omega_fun, plot_=plot_, force=force, vol_corr=vol_corr, struct_grd=struct_grd, response=response)

    return  K, K_bound, disp_cent, u_disp
