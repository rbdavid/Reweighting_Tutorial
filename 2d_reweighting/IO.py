import sys
import matplotlib.pyplot as plt
import numpy as np

########################
def config_parser(config_file,parameters):	# Function to take config file and create/fill the parameter dictionary 
    
    necessary_parameters = ['emus_meta_file','num_biased_dimensions','temperature','k_B','unbiased_data_files_naming','nWindows','xMin','xMax','yMin','yMax','xBins','yBins','x_axis_label','y_axis_label','reweighted_x_axis_prob_density_plot_name','reweighted_y_axis_prob_density_plot_name','reweighted_x_axis_unstitched_fe_plot_name','reweighted_y_axis_unstitched_fe_plot_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','reweighted_2d_heatmap_plot_name','reweighted_2d_heatmap_data_file_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','output_directory']

    all_parameters       = ['emus_meta_file','num_biased_dimensions','temperature','k_B','unbiased_data_files_naming','nWindows','xMin','xMax','yMin','yMax','xBins','yBins','x_axis_label','y_axis_label','reweighted_x_axis_prob_density_plot_name','reweighted_y_axis_prob_density_plot_name','reweighted_x_axis_unstitched_fe_plot_name','reweighted_y_axis_unstitched_fe_plot_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','reweighted_2d_heatmap_plot_name','reweighted_2d_heatmap_data_file_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','output_directory','x_axis_line_functions','y_axis_line_functions','td_line_function','bootstrap_bool','nIterations']

    for i in range(len(necessary_parameters)):
        parameters[necessary_parameters[i]] = ''
        
    # SETTING DEFAULT PARAMETERS FOR OPTIONAL PARAMETERS:
    parameters['x_axis_line_functions'] = None
    parameters['y_axis_line_functions'] = None
    parameters['2d_function'] = None
    parameters['bootstrap_bool'] = False
    parameters['nIterations'] = 100
    
    # GRABBING PARAMETER VALUES FROM THE CONFIG FILE:
    with open(config_file) as f:
        exec(compile(f.read(),config_file,'exec'),parameters)

    for key, value in list(parameters.items()):
        if value == '':
            print('%s has not been assigned a value. This variable is necessary for the script to run. Please declare this variable within the config file.' %(key))
            sys.exit()
            
########################
def summary(file_name,arguments,parameters):
    with open(file_name,'w') as f:
        f.write('To recreate this analysis, run this line:\n')
        for i in range(len(sys.argv)):
            f.write('%s ' %(arguments[i]))
        f.write('\n\n')
        f.write('Parameters used:\n')
        for key, value in list(parameters.items()):
            if key == '__builtins__':
                continue
            if type(value) == int or type(value) == float:
                f.write('%s = %s\n' %(key,value))
            else:
                f.write("%s = '%s'\n" %(key,value))
        
########################
def read_meta(meta_file):
    file_list = []
    r0_k_list = []
    with open(meta_file,'r') as f:
        for line in f:
            if not line.startswith('#'):
                temp = line.split()
                file_list.append(temp[0])
                r0_k_list.append([float(temp[1]),float(temp[2])])
                
    return np.array(r0_k_list), file_list

########################
def finish_plot(plot_id, figure_name, xlabel, ylabel, xlim=None, ylim=None, font_size=14):
    plt.figure(plot_id)
    plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
    plt.ylabel(ylabel,size=font_size)
    plt.xlabel(xlabel,size=font_size)
    if type(xlim) == tuple and len(xlim) == 2:
        plt.xlim(xlim)
    if type(ylim) == tuple and len(ylim) == 2:
        plt.ylim(ylim)
    plt.savefig(figure_name,dpi=600,transparent=True)

