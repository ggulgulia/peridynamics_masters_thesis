from peridynamic_plane_stress import *
from fenics_convergence import *
from peridynamic_infl_fun import *
from datetime import datetime as dttm
from os import path, getcwd, mkdir

def global_plotting_parameters():
    """
    returns the plotting paramters like legend size, text font size, etc
    See the return object
    :returns: dictonary of plotting paramters 

    """
    
    dpi = 10
    plotting_param_dict = {'dpi':dpi, 'axis_font':{'size': str(int(15*dpi))}, 'title_font':{'size': str(18*dpi)}, 
                            'legend_size':{'size': str(12*dpi)}, 'tick_size': 12*dpi, 'marker_size':100*3.5*dpi}
    return plotting_param_dict


class my_fig_num_counter():

    """
    simple object to keep track of
    plot numbers
    """
    def __init__(self, init_val):
        """a very trivial constructor """
        self.err_fig_num = init_val
        self.disp_fig_num = init_val
        

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
        if(int(num_cells/grd_fact) <600):
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
