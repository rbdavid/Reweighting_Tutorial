#!/home/rbdavid/bin/python
# ----------------------------------------
# USAGE:
# assuming 1D US simulations are being reweighted into a 2d collective variable space

# ----------------------------------------
# PREAMBLE:

import sys
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import NullFormatter
from emus import emus,avar,usutils
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

nullfmt = NullFormatter()
free_energy_cmap = plt.cm.get_cmap('Blues_r')
four_pi = 4.*np.pi

config_file = sys.argv[1]

# ----------------------------------------
# FUNCTIONS:

necessary_parameters = ['emus_meta_file','num_biased_dimensions','temperature','k_B','projected_data_files_naming','cv_data_files_naming','start_production_number','end_production_number','nWindows','xMin','xMax','yMin','yMax','xBins','yBins','x_axis_label','y_axis_label','reweighted_x_axis_prob_density_plot_name','reweighted_y_axis_prob_density_plot_name','reweighted_x_axis_unstitched_fe_plot_name','reweighted_y_axis_unstitched_fe_plot_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','reweighted_2d_heatmap_plot_name','reweighted_2d_heatmap_data_file_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','output_directory']

all_parameters = ['emus_meta_file','num_biased_dimensions','temperature','k_B','projected_data_files_naming','cv_data_files_naming','start_production_number','end_production_number','nWindows','xMin','xMax','yMin','yMax','xBins','yBins','x_axis_label','y_axis_label','reweighted_x_axis_prob_density_plot_name','reweighted_y_axis_prob_density_plot_name','reweighted_x_axis_unstitched_fe_plot_name','reweighted_y_axis_unstitched_fe_plot_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','reweighted_2d_heatmap_plot_name','reweighted_2d_heatmap_data_file_name','reweighted_x_axis_stitched_fe_plot_name','reweighted_y_axis_stitched_fe_plot_name','output_directory','x_axis_line_functions','y_axis_line_functions','td_line_function','bootstrap_bool','nIterations']
def config_parser(config_file):	# Function to take config file and create/fill the parameter dictionary 
	for i in range(len(necessary_parameters)):
		parameters[necessary_parameters[i]] = ''

	# SETTING DEFAULT PARAMETERS FOR OPTIONAL PARAMETERS:
        parameters['x_axis_line_functions'] = None
        parameters['y_axis_line_functions'] = None
        parameters['2d_function'] = None
        parameters['bootstrap_bool'] = False
        parameters['nIterations'] = 100

	# GRABBING PARAMETER VALUES FROM THE CONFIG FILE:
	execfile(config_file,parameters)
	for key, value in parameters.iteritems():
		if value == '':
			print '%s has not been assigned a value. This variable is necessary for the script to run. Please declare this variable within the config file.' %(key)
			sys.exit()

def summary():
	with open(parameters['output_directory']+parameters['summary_output_filename'],'w') as f:
        	f.write('To recreate this analysis, run this line:\n')
        	for i in range(len(sys.argv)):
        		f.write('%s ' %(sys.argv[i]))
        	f.write('\n\n')
		f.write('Parameters used:\n')
		for i in all_parameters:
			f.write('%s = %s \n' %(i,parameters[i]))
		f.write('\n\n')

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

def detect_local_minima(arr):
        # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        Takes an array and detects the troughs using the local minimum filter.
        Returns a boolean mask of the troughs (i.e. 1 when the pixel's value is the neighborhood minimum, 0 otherwise)
        """
        # define an connected neighborhood
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
        neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
        # apply the local minimum filter; all locations of minimum value in their neighborhood are set to 1
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
        local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
        # local_min is a mask that contains the peaks we are looking for, but also the background. In order to isolate the peaks we must remove the background from the mask.
         
        # we create the mask of the background
        background = (arr == np.take(arr,0))    # assumes that the first element in the arr array corresponds to the background value; this is a bug if the first element does not match the background value!
         
        # a little technicality: we must erode the background in order to successfully subtract it from local_min, otherwise a line will appear along the background border (artifact of the local minimum filter)
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
        eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
         
        # we obtain the final mask, containing only peaks, by removing the background from the local_min mask
        detected_minima = local_min.astype(np.int) - eroded_background.astype(np.int)
        return np.where(detected_minima)    # returns a tuple of n arrays where n is the number of dimensions of the arr array. The length of each array corresponds to the number of found local minima

def main():
        
        kT = parameters['k_B']*parameters['temperature']
        beta = 1./kT
        
        # ----------------------------------------
        # load data
        print 'Starting EMUS analysis'
        psis, cv_trajs, neighbors = usutils.data_from_meta(parameters['emus_meta_file'],parameters['num_biased_dimensions'],T=parameters['temperature'],k_B=parameters['k_B']) # psis is a list of arrays containing the relative weights of each window's cv values in all windows. cv_trajs is a list of arrays containing the raw cv values for each window. neighbors is an 2D array containing indices for neighboring windows to be used. 
        # calculate the partition function for each window
        z, F = emus.calculate_zs(psis,neighbors=neighbors)  # z is an 1D array that contains the normalization constant for each window. F is an 2D array (num windows by num windows) that contains the eigenvalues of the iterature EMUS process.

        r0_k_data, file_list = read_meta(parameters['emus_meta_file'])    # file_list not being saved...

        # ----------------------------------------
        # FOR REWEIGHTING, ONLY NEED THE Z ARRAY
        del psis
        del cv_trajs
        del neighbors
        del F
        del file_list

        # ----------------------------------------
        # load in unbiased cv data
        print 'Starting to load in cv data (original and new)'
        nWindows_range = range(int(parameters['nWindows']))
        nProductions_range = range(int(parameters['start_production_number']),int(parameters['end_production_number'])+1)
        data = ['' for i in nWindows_range]
        for i in nWindows_range:
                print 'loading window', i
                temp_projected_data = np.loadtxt(parameters['projected_data_files_naming']%(i,i))[:,:2]     # only grabbing projected data of PC1 and PC2
                temp_data = np.zeros((len(temp_projected_data),3))
                temp_data[:,1] = temp_projected_data[:,0]   # PC1 projected data is index 1
                temp_data[:,2] = temp_projected_data[:,1]   # PC2 projected data is index 2
                count = 0
                for j in nProductions_range:
                        temp_cv_data = np.loadtxt(parameters['cv_data_files_naming']%(j,i,j),skiprows=1)
                        temp_data[count:count+len(temp_cv_data),0] = temp_cv_data[:,1]  # biased CV data is index 0
                        count += len(temp_cv_data)
                data[i] = temp_data

                if count != len(temp_projected_data):
                        print 'projected data file', i, 'has different number of lines than the combined cv data from the respective window. This should not happen.', count, len(temp_projected_data)
                        sys.exit()

                #print 'Window', i, ': First frame data ', data[i][0], 'Last frame data', data[i][-1], 'Do these numbers match up to the expected values?'
        
        # ----------------------------------------
        # prep 2d histogram arrays
        xMin = float(parameters['xMin']) 
        xMax = float(parameters['xMax']) 
        yMin = float(parameters['yMin']) 
        yMax = float(parameters['yMax']) 
        xBins = int(parameters['xBins'])
        yBins = int(parameters['yBins'])
        delta_x = (xMax - xMin)/xBins
        delta_y = (yMax - yMin)/yBins
        print 'Bin widths:', delta_x, delta_y
        x_half_bins = np.array([xMin + delta_x*(i+0.5) for i in range(xBins)])
        y_half_bins = np.array([yMin + delta_y*(i+0.5) for i in range(yBins)])
        x_edges = np.array([xMin + delta_x*i for i in range(xBins+1)])
        y_edges = np.array([yMin + delta_y*i for i in range(yBins+1)])
        
        # ----------------------------------------
        # reweighting 1D US results into a 2D collective variable space
        nValues_total = 0
        x_total_fe_counts = np.zeros(xBins,dtype=np.float32)
        y_total_fe_counts = np.zeros(yBins,dtype=np.float32)
        td_total_fe_counts = np.zeros((xBins,yBins),dtype=np.float32)
        for i in nWindows_range:
                nValues = len(data[i])
                nValues_total += nValues

                x_window_counts = np.zeros(xBins)
                y_window_counts = np.zeros(yBins)
                x_window_fe_counts = np.zeros(xBins)
                y_window_fe_counts = np.zeros(yBins)

                with open(parameters['output_directory'] + 'window%03d.prods_%d_%d.frame_weights.dat'%(i,int(parameters['start_production_number']),int(parameters['end_production_number'])),'w') as W:
                        for j in range(nValues):
                                
                                # ----------------------------------------
		                # HISTOGRAMMING DATA POINT
                                x_index = int((data[i][j][1] - xMin)/delta_x)
                                y_index = int((data[i][j][2] - yMin)/delta_y)
		                
                                if x_index < 0 or x_index > xBins:
                                        print '!!! 0 > x_index >= xBins ...', data[i][j][0], x_index, i, 'Histogram bounds are not wide enough. Job failed.'
                                        sys.exit()
                                elif x_index == xBins:
                                        x_index = -1
                               
                                if y_index < 0 or y_index > yBins:
                                        print '!!! 0 > y_index >= yBins ...', data[i][j][0], y_index, i, 'Histogram bounds are not wide enough. Job failed.'
                                        sys.exit()
                                elif y_index == yBins:
                                        y_index = -1

                                # ----------------------------------------
		                # ANALYZING DATA POINT IN CONSIDERATION OF CURRENT WINDOW
                                w = np.exp((-beta*r0_k_data[i][1]/2.)*(data[i][j][0] - r0_k_data[i][0])**2)/z[i]     # exp((-k/2*k_B*T)(r-r0)^2)/z; no volume correction...
                                #w = (data[i][j][0]**2)*np.exp((-beta*r0_k_data[i][1]/2.)*(data[i][j][0] - r0_k_data[i][0])**2)/z[i]     # r^2 * exp((-k/2*k_B*T)(r-r0)^2)/z; 
                                
                                x_window_counts[x_index] += 1
                                x_window_fe_counts[x_index] += 1/w
                                y_window_counts[y_index] += 1
                                y_window_fe_counts[y_index] += 1/w

                                # ----------------------------------------
		                # ANALYZING DATA POINT IN CONSIDERATION OF ALL WINDOWS
                                w = 0
                                for k in nWindows_range:
                                        w+= np.exp((-beta*r0_k_data[k][1]/2.)*(data[i][j][0] - r0_k_data[k][0])**2)/z[k]       # exp((-k/2*k_B*T)(r-r0)^2)/z; no volume correction...
                                        #w+= (data[i][j][0]**2)*np.exp((-beta*r0_k_data[k][1]/2.)*(data[i][j][0] - r0_k_data[k][0])**2)/z[k]       # r^2 * exp((-k/2*k_B*T)(r-r0)^2)/z; 

		                w /= parameters['nWindows'] # <r^2 * exp((-k/2*k_B*T)(r-r0)^2)/z>; average reciprocal boltzmann weight of data point in all possible windows;
                                weight = 1./w

                                W.write('%15d %15f\n'%(j,weight))
                                x_total_fe_counts[x_index] += weight
                                y_total_fe_counts[y_index] += weight
                                td_total_fe_counts[x_index][y_index] += weight
                
                # ----------------------------------------
		# FINISHING ANALYSIS OF THE REWEIGHTED PROB. DENSITY OF EACH INDIVIDUAL WINDOW - XDATA
                x_window_prob_density = x_window_counts/(nValues*delta_x)
                plt.figure(1)
                plt.plot(x_half_bins[:],x_window_prob_density[:],zorder=3)
                
                # ----------------------------------------
		# FINISHING ANALYSIS OF THE REWEIGHTED FREE ENERGY OF EACH INDIVIDUAL WINDOW - XDATA
                x_window_fe_counts = -kT*np.log(x_window_fe_counts/(nValues*delta_x))  # no volume correction
                #x_window_fe_counts = np.array([-kT*np.log(x_window_fe_counts[j]/(nValues*delta_x*four_pi)) for j in range(xBins)])
                plt.figure(2)
                plt.plot(x_half_bins[x_window_counts > 10.], x_window_fe_counts[x_window_counts > 10],zorder=3)
                
                # ----------------------------------------
		# FINISHING ANALYSIS OF THE REWEIGHTED PROB. DENSITY OF EACH INDIVIDUAL WINDOW - YDATA
                y_window_prob_density = y_window_counts/(nValues*delta_y)
                plt.figure(3)
                plt.plot(y_half_bins[:],y_window_prob_density[:],zorder=3)
                
                # ----------------------------------------
		# FINISHING ANALYSIS OF THE REWEIGHTED FREE ENERGY OF EACH INDIVIDUAL WINDOW - YDATA
                y_window_fe_counts = -kT*np.log(y_window_fe_counts/(nValues*delta_y))  # no volume correction
                #y_window_fe_counts = np.array([-kT*np.log(y_window_fe_counts[j]/(nValues*delta_y*four_pi)) for j in range(yBins)])
                plt.figure(4)
                plt.plot(y_half_bins[y_window_counts > 10.], y_window_fe_counts[y_window_counts > 10],zorder=3)
                
                print 'Done with window', i

        # ----------------------------------------
        # FINISHED REWEIGHTING, RUNNING BOOTSTRAP ANALYSIS TO GET ERROR BARS
        if parameters['bootstrap_bool']:
                print 'Beginning bootstrap analysis to approximate error in reweighted CVs'
                original_data = np.empty((0,3))
                for i in nWindows_range:
                        original_data = np.concatenate((original_data,np.array(data[i])))

                if original_data.shape != (nValues_total,3):
                        print original_data.shape, nValues_total, 'something went wrong during bootstrapping'
                        sys.exit()

                x_bootstrap_results = []
                y_bootstrap_results = []
                td_bootstrap_results = []
                for i in range(parameters['nIterations']):
                        print 'Starting Step %d of %d steps in bootstrap analysis'%(i,parameters['nIterations'])
                        # create bootstrap data
                        sample_data = original_data[np.random.randint(nValues_total,size=nValues_total)]
                        x_total_fe_bootstrap = np.zeros(xBins)
                        y_total_fe_bootstrap = np.zeros(yBins)
                        td_total_fe_bootstrap = np.zeros((xBins,yBins))

                        # analyze new dataset to get reweighted FE of each bin
                        for j in range(nValues_total):
                                
                                # ----------------------------------------
		                # HISTOGRAMMING DATA POINT
                                x_index = int((sample_data[j,1] - xMin)/delta_x)
                                y_index = int((sample_data[j,2] - yMin)/delta_y)
                                
                                if x_index == xBins:
                                        x_index = -1
                                if y_index == yBins:
                                        y_index = -1

                                w = 0
                                for k in nWindows_range:
                                        w+= np.exp((-beta*r0_k_data[k][1]/2.)*(sample_data[j,0] - r0_k_data[k][0])**2)/z[k]       # exp((-k/2*k_B*T)(r-r0)^2)/z; no volume correction...
		                
                                w /= parameters['nWindows'] # <r^2 * exp((-k/2*k_B*T)(r-r0)^2)/z>; average reciprocal boltzmann weight of data point in all possible windows;
                                
                                x_total_fe_bootstrap[x_index] += 1/w
                                y_total_fe_bootstrap[y_index] += 1/w
                                td_total_fe_bootstrap[x_index][y_index] += 1/w
		        
                        x_total_fe_bootstrap /= delta_x*nValues_total
                        x_total_fe_bootstrap = -kT*np.log(x_total_fe_bootstrap)  # no volume correction
                        x_total_fe_bootstrap -= np.ndarray.min(x_total_fe_bootstrap)
                        x_bootstrap_results.append(x_total_fe_bootstrap)

                        y_total_fe_bootstrap /= delta_y*nValues_total
                        y_total_fe_bootstrap = -kT*np.log(y_total_fe_bootstrap)  # no volume correction
                        y_total_fe_bootstrap -= np.ndarray.min(y_total_fe_bootstrap)
                        y_bootstrap_results.append(y_total_fe_bootstrap)
                        
                        td_total_fe_bootstrap /= delta_x*delta_y*nValues_total     # currently a stitched probability density; no volume correction
                        td_total_fe_bootstrap = -kT*np.log(td_total_fe_bootstrap)
                        td_total_fe_bootstrap -= np.ndarray.min(td_total_fe_bootstrap)
                        td_bootstrap_results.append(td_total_fe_bootstrap)

                ### NOTE: CALCS THE STANDARD ERROR OF THE MEAN
                #x_std_error = np.std(np.array(x_bootstrap_results),axis=0)/np.sqrt(parameters['nIterations'])
                #y_std_error = np.std(np.array(y_bootstrap_results),axis=0)/np.sqrt(parameters['nIterations'])
                #td_std_error = np.std(np.array(td_bootstrap_results),axis=0)/np.sqrt(parameters['nIterations'])
       
                ### NOTE: CALCS THE STANDARD DEVIATION OF THE BOOTSTRAPPED DATA
                x_std_error = np.std(np.array(x_bootstrap_results),axis=0)
                y_std_error = np.std(np.array(y_bootstrap_results),axis=0)
                td_std_error = np.std(np.array(td_bootstrap_results),axis=0)
       
                np.savetxt(parameters['output_directory'] + 'x_axis_error_analysis.dat', x_std_error, fmt='%.10f')
                np.savetxt(parameters['output_directory'] + 'y_axis_error_analysis.dat', y_std_error, fmt='%.10f')
                np.savetxt(parameters['output_directory'] + 'td_axis_error_analysis.dat', td_std_error, fmt='%.10f')
                del sample_data
                del x_total_fe_bootstrap
                del y_total_fe_bootstrap
                del td_total_fe_bootstrap
                del x_bootstrap_results
                del y_bootstrap_results
                del td_bootstrap_results
            
        # ----------------------------------------
        # FINISHED REWEIGHTING, CLEANING UP VARIABLE SPACE
        del data
        del x_window_counts
        del y_window_counts
        del x_window_prob_density
        del y_window_prob_density
        del x_window_fe_counts
        del y_window_fe_counts

        # ----------------------------------------
        # FINISHING PLOTTING OF THE REWEIGHTED PROB. DENSITY OF EACH INDIVIDUAL WINDOW - XDATA
        plt.figure(1)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.ylabel('Probability Density',size=14)
        plt.xlabel(parameters['x_axis_label'],size=14)
        plt.xlim((xMin,xMax))
        plt.savefig(parameters['output_directory']+parameters['reweighted_x_axis_prob_density_plot_name'],dpi=600,transparent=True)
        
        # ----------------------------------------
        # FINISHING PLOTTING OF THE REWEIGHTED, UNSTITCHED FREE ENERGY - XDATA
        plt.figure(2)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.ylabel(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
        plt.xlabel(parameters['x_axis_label'],size=14)
        plt.xlim((xMin,xMax))
        plt.savefig(parameters['output_directory']+parameters['reweighted_x_axis_unstitched_fe_plot_name'],dpi=600,transparent=True)
        
        # ----------------------------------------
        # FINISHING PLOTTING OF THE REWEIGHTED PROB. DENSITY OF EACH INDIVIDUAL WINDOW - YDATA
        plt.figure(3)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.ylabel('Probability Density',size=14)
        plt.xlabel(parameters['y_axis_label'],size=14)
        plt.xlim((yMin,yMax))
        plt.savefig(parameters['output_directory']+parameters['reweighted_y_axis_prob_density_plot_name'],dpi=600,transparent=True)
        
        # ----------------------------------------
        # FINISHING PLOTTING OF THE REWEIGHTED, UNSTITCHED FREE ENERGY - YDATA
        plt.figure(4)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.ylabel(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
        plt.xlabel(parameters['y_axis_label'],size=14)
        plt.xlim((yMin,yMax))
        plt.savefig(parameters['output_directory']+parameters['reweighted_y_axis_unstitched_fe_plot_name'],dpi=600,transparent=True)
        
        # ----------------------------------------
        # PLOTTING REWEIGHTED X-DATA FE SURFACE
        print np.sum(x_total_fe_counts)
        x_total_fe_counts /= delta_x*nValues_total  # no volume correction
        #x_total_fe_counts /= four_pi*delta_x*nValues_total
        print np.sum(x_total_fe_counts)

        if parameters['x_axis_line_functions'] != None:
                # ----------------------------------------
                # STATE DESCRIPTION OF X-DATA FE SURFACE; CURRENTLY ASSUMING ONLY TWO STATES
                line_x = [parameters['x_axis_line_functions'][0],parameters['x_axis_line_functions'][0]]    # two points in this line
                line_y = [0.0,parameters['x_axis_line_functions'][1]]
                
                # this assumes a straight line cutoff for a two state description... a whole lot of assumptions that are not generalizable...
                state_one_prob_density = np.sum([x_total_fe_counts[i] for i in range(xBins) if x_half_bins[i] < parameters['x_axis_line_functions'][0]])    # state one is closed conf.
                state_two_prob_density = np.sum([x_total_fe_counts[i] for i in range(xBins) if x_half_bins[i] >= parameters['x_axis_line_functions'][0]])   # state two is open conf.

                state_free_energies = np.array([-kT*np.log(state_one_prob_density), -kT*np.log(state_two_prob_density)])
                state_free_energies -= np.min(state_free_energies)
                print 'X-axis state space FE: State One: ', state_free_energies[0], 'kcal mol^-1; State two: ', state_free_energies[1], 'kcal mol^-1'
                # ----------------------------------------
        
        x_total_fe_counts = -kT*np.log(x_total_fe_counts) # no volume correction
        x_total_fe_counts -= np.ndarray.min(x_total_fe_counts)
        np.savetxt(parameters['output_directory'] + parameters['reweighted_x_axis_stitched_fe_data_file_name'], np.c_[range(xBins),x_half_bins,x_total_fe_counts], fmt='%.10f')
        
        plt.figure(5)
        if parameters['bootstrap_bool']:
                plt.errorbar(x_half_bins[:],x_total_fe_counts[:],yerr=x_std_error,ecolor='r',elinewidth=0.5,zorder=3)
        else:
                plt.plot(x_half_bins[:],x_total_fe_counts[:],zorder=3)
        if parameters['x_axis_line_functions'] != None:
                plt.plot(line_x,line_y,'r--',alpha=0.5,zorder=3)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.ylabel(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
        plt.xlabel(parameters['x_axis_label'],size=14)
        plt.xlim((xMin,xMax))
        ###NOTE
        plt.ylim((-0.05,10))
        ###
        plt.savefig(parameters['output_directory']+parameters['reweighted_x_axis_stitched_fe_plot_name'],dpi=600,transparent=True)

        # ----------------------------------------
        # PLOTTING REWEIGHTED Y-DATA FE SURFACE
        y_total_fe_counts /= delta_y*nValues_total  # no volume correction
        #y_total_fe_counts /= four_pi*delta_y*nValues_total

        if parameters['y_axis_line_functions'] != None:
                # ----------------------------------------
                # STATE DESCRIPTION OF X-DATA FE SURFACE; CURRENTLY ASSUMING ONLY TWO STATES
                line_x = [parameters['y_axis_line_functions'][0],parameters['y_axis_line_functions'][0]]    # two points in this line
                line_y = [0.0,parameters['y_axis_line_functions'][1]]
                
                # this assumes a straight line cutoff for a two state description... a whole lot of assumptions that are not generalizable...
                state_one_prob_density = np.sum([y_total_fe_counts[i] for i in range(yBins) if y_half_bins[i] < parameters['y_axis_line_functions'][2]])
                state_two_prob_density = np.sum([y_total_fe_counts[i] for i in range(yBins) if y_half_bins[i] >= parameters['y_axis_line_functions'][2]])

                state_free_energies = [-kT*np.log(state_one_prob_density), -kT*np.log(state_two_prob_density)]
                state_free_energies -= np.min(state_free_energies)
                print 'Y-axis state space FE: State One: ', state_free_energies[0], 'kcal mol^-1; State two: ', state_free_energies[1], 'kcal mol^-1'
                # ----------------------------------------
        
        y_total_fe_counts = -kT*np.log(y_total_fe_counts)
        y_total_fe_counts -= np.ndarray.min(y_total_fe_counts)
        np.savetxt(parameters['output_directory'] + parameters['reweighted_y_axis_stitched_fe_data_file_name'], np.c_[range(yBins),y_half_bins,y_total_fe_counts], fmt='%.10f')
        
        plt.figure(6)
        if parameters['bootstrap_bool']:
                plt.errorbar(y_half_bins[:],y_total_fe_counts[:],yerr=y_std_error,ecolor='r',elinewidth=0.5,zorder=3)
        else:
                plt.plot(y_half_bins[:],y_total_fe_counts[:],zorder=3)
        if parameters['y_axis_line_functions'] != None:
                plt.plot(line_x,line_y,'r--',alpha=0.5,zorder=3)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.ylabel(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
        plt.xlabel(parameters['y_axis_label'],size=14)
        plt.xlim((yMin,yMax))
        ###NOTE
        plt.ylim((-0.05,10))
        ###
        plt.savefig(parameters['output_directory']+parameters['reweighted_y_axis_stitched_fe_plot_name'],dpi=600,transparent=True)

        # ----------------------------------------
        # PLOTTING REWEIGHTED 2D FE LANDSCAPE
        td_total_fe_counts /= delta_x*delta_y*nValues_total     # currently a stitched probability density; no volume correction
        #td_total_fe_counts /= four_pi*delta_x*delta_y*nValues_total     # currently a stitched probability density; with vol correction

        if parameters['td_line_function'] != None:
                # ----------------------------------------
                # STATE DESCRIPTION OF 2D FE SURFACE; CURRENTLY ASSUMING ONLY TWO STATES
                
                # x_half_bins should be larger than the y_half_bins array; a safe assumption for the purposes of this code
                line_x = x_half_bins
                line_y = eval('line_x'+parameters['td_line_function'])
                state_one_prob_density = 0.
                state_two_prob_density = 0.
                for i in range(xBins):
                        state_one_prob_density += np.sum(td_total_fe_counts[i]*(y_half_bins < line_y[i]).astype(float))
                        state_two_prob_density += np.sum(td_total_fe_counts[i]*(y_half_bins >= line_y[i]).astype(float))
                        
                state_free_energies = [-kT*np.log(state_one_prob_density), -kT*np.log(state_two_prob_density)]
                state_free_energies -= np.min(state_free_energies)
                print '2D state space FE: State One: ', state_free_energies[0], 'kcal mol^-1; State two: ', state_free_energies[1], 'kcal mol^-1'
                # ----------------------------------------
       
        td_total_fe_counts = -kT*np.log(td_total_fe_counts)
        td_total_fe_counts -= np.ndarray.min(td_total_fe_counts)
        np.savetxt(parameters['output_directory'] + parameters['reweighted_2d_heatmap_data_file_name'], td_total_fe_counts, fmt='%.10f')
        masked_fe_counts = ma.masked_where(np.isinf(td_total_fe_counts),td_total_fe_counts)

        fig, ax = plt.subplots(num=7)
        #plt.pcolormesh(x_edges,y_edges,masked_fe_counts.T,cmap=free_energy_cmap,zorder=3)
        #cb1 = plt.colorbar(extend='max')    #
        plt.pcolormesh(x_edges,y_edges,masked_fe_counts.T,cmap=free_energy_cmap,zorder=3,vmax=10)
        cb1 = plt.colorbar()    #extend='max'
        cb1.set_label(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
        if parameters['td_line_function'] != None:
                plt.plot(line_x,line_y,'r--',alpha=0.5,zorder=3)
        ax.set_aspect('equal')
        plt.xlim((xMin,xMax))
        plt.ylim((yMin,yMax))
        plt.ylabel(parameters['y_axis_label'],size=14)
        plt.xlabel(parameters['x_axis_label'],size=14)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--',zorder=0)
        plt.savefig(parameters['output_directory']+parameters['reweighted_2d_heatmap_plot_name'],dpi=600,transparent=True)

        plt.close()
        print 'Done plotting...'

        ## ----------------------------------------
        ## FIND LOCAL MINIMA WITHIN THE 2D FE SURFACE
        #local_minima_locations = detect_local_minima(td_total_fe_counts)        # unfortunately, turning this value into an array creates a memory error crash
        #local_minima_values = td_total_fe_counts[local_minima_locations]
        #
        #idx = local_minima_values.argsort()[:1]     #parameters['number_of_local_minimas']
        #
        #global_minima_location = [local_minima_locations[0][idx[0]],local_minima_locations[1][idx[0]]]

        #for i in range(len(local_minima_values)):
        #        distance = np.absolute(local_minima_locations[0][i] - global_minima_location[0]) + np.absolute(local_minima_locations[1][i] - global_minima_location[1])
        #        print i, distance, local_minima_locations[0][i], local_minima_locations[1][i], local_minima_values[i]

        # ----------------------------------------
        # OUTPUT SUMMARY FILE
        summary()

# ----------------------------------------
# CREATING PARAMETER DICTIONARY
parameters = {}
config_parser(config_file)

# ----------------------------------------
# CREATING OUTPUT DIRECTORY
if parameters['output_directory'][-1] != os.sep:
        parameters['output_directory'] += os.sep

if os.path.exists(parameters['output_directory']):
        print 'The output directory, ', parameters['output_directory'], ' already exists. Please delete this directory or select a different one for output before proceeding.'
        sys.exit()
else:
        os.mkdir(parameters['output_directory'])

# ----------------------------------------
# MAIN
if __name__ == '__main__':
	main()

