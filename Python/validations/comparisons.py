from testScriptInflFuns import *
from testScriptHorizons import *
from testScriptMaterials import *
from testScriptInflFunStrctGrd import compare_PD_infl_funs_with_FE_StrctGrd
from testScriptHorizonStrctGrd import compare_PD_horizons_with_FE_StrctGrd
#from testScriptMaterials import *

def run_comparisons():

    print("*********************************************************")
    print("*********************************************************")
    print('solving using Finite Elements:')
    tol = 1e-5 #tolerance for convergence of FE solution
    _, _, mesh_lst, u_fe_conv, err_norm_lst = fenics_mesh_convergence(tol=tol, plot_=False)
    print("Number of cells in FEniCS mesh on which the FE solution converged %i" %mesh_lst[-1].num_cells())
    print("*********************************************************")
    print("*********************************************************")

    

    pwd = getcwd()
    today = dttm.now().strftime("%Y%m%d%S")
    data_dir = path.join(pwd, 'validation_test_on_'+today)
    hori = path.join(data_dir, 'horizon')
    omga = path.join(data_dir, 'inflfun')
    mkdir(data_dir); mkdir(hori); mkdir(omga)
    fig_cnt = my_fig_num_counter(0)

    unstr_msh_lst, strct_msh_lst = generate_struct_mesh_list_for_pd_tests()
    ##triangular mesh, horizon studies
    #cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(unstr_msh_lst, u_fe_conv, fig_cnt, data_path=hori, plot_=False, vol_corr=False,  struct_grd=False)

    #cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(unstr_msh_lst, u_fe_conv, fig_cnt, data_path=hori, plot_=False, vol_corr=True, struct_grd=False)

#    ##triangular mesh, influence function studies
#    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(unstr_msh_lst, u_fe_conv, fig_cnt, data_path=omga, plot_=False, vol_corr=False,  struct_grd=False)
#
#    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(unstr_msh_lst, u_fe_conv, fig_cnt, data_path=omga, plot_=False, vol_corr=True, struct_grd=False)
#
        
    ######## the method in the script calls correspondence constitutive response model by default ##########################
    cell_cent_top_lst_csp, u_top_fe_conv_lst_csp, disp_cent_PD_array_lst_csp, u_disp_PD_array_lst_csp, abs_error_end_particle_lst_csp, rel_error_end_particle_lst_csp = compare_PD_material_models(strct_msh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=True)
#    sqr_msh_lst = managable_mesh_list(mesh_lst, struct_grd=False)
    #sqr_msh_lst = mesh_lst 
    ##square mesh, horizon studies
    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, abs_error_end_particle_lst, rel_error_end_particle_lst = compare_PD_horizons_with_FE(strct_msh_lst, u_fe_conv, fig_cnt, data_path=hori, plot_=False, vol_corr=False,  struct_grd=True)

    #cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_horizons_with_FE(strct_msh_lst, u_fe_conv, fig_cnt, data_path=hori, plot_=False, vol_corr=True, struct_grd=True)
    
    ##square mesh, influence function studies
#    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst, abs_error_end_particle_lst, rel_error_end_particle_lst = compare_PD_infl_funs_with_FE(strct_msh_lst, u_fe_conv, fig_cnt, data_path=omga, plot_=False, vol_corr=False,  struct_grd=True)
#
#    cell_cent_top_lst, u_top_fe_conv_lst, disp_cent_PD_array_lst, u_disp_PD_array_lst = compare_PD_infl_funs_with_FE(strct_msh_lst, u_fe_conv, fig_cnt, data_path=omga, plot_=False, vol_corr=True, struct_grd=True)
#
    #### Influence function studies with new scripts ####
    u_disp_PD_array_lst, abs_error_avg_cntlineY_lst, rel_error_avg_cntlineY_lst = compare_PD_infl_funs_with_FE_StrctGrd(strct_msh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=True)

    ##### Horizon studies with new scripts ########
    u_disp_PD_array_lst, abs_error_avg_cntlineY_lst, rel_error_avg_cntlineY_lst = compare_PD_horizons_with_FE_StrctGrd(strct_msh_lst, u_fe_conv, fig_cnt, data_path=None, material='steel', plot_=False, force=-5e8, vol_corr=True, struct_grd=True)
    print("SUCCESSFULLY FINISHED THE STUDIES\nGOODLUCK ANALYZING ERRORS FROM THE TON OF FILES THAT HAVE BEEN WRITTEN\n")
    return 
