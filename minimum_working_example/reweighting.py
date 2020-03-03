# ----------------------------------------
# USAGE:
# assuming 1D US simulations are being reweighted into a 1d unbiased collective variable space

# ----------------------------------------
# PREAMBLE:

import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from emus import emus,avar,usutils

config_file = sys.argv[1]
IO_functions_file = sys.argv[2]

config_parser = importlib.import_module(IO_functions_file.split('.py')[0],package=None).config_parser
summary = importlib.import_module(IO_functions_file.split('.py')[0],package=None).summary
read_meta = importlib.import_module(IO_functions_file.split('.py')[0],package=None).read_meta
finish_plot = importlib.import_module(IO_functions_file.split('.py')[0],package=None).finish_plot

# ----------------------------------------
# FUNCTIONS:

def main():
    """
    """
    four_pi = 4.*np.pi
    kT = parameters['k_B']*parameters['temperature']
    beta = 1./(2.*kT)

    # ----------------------------------------
    # RUN EMUS ANALYSIS STEP
    print('Starting EMUS analysis')
    psis, cv_trajs, neighbors = usutils.data_from_meta(parameters['emus_meta_file'],parameters['num_biased_dimensions'],T=parameters['temperature'],k_B=parameters['k_B']) # psis is a list of arrays containing the relative weights of each window's cv values in all windows. cv_trajs is a list of arrays containing the raw cv values for each window. neighbors is an 2D array containing indices for neighboring windows to be used. 
    z, F = emus.calculate_zs(psis,neighbors=neighbors,n_iter=1000)  # z is an 1D array that contains the normalization constant for each window. F is an 2D array (num windows by num windows) that contains the eigenvalues of the iterature EMUS process.
    zerr, zcontribs, ztaus = avar.calc_partition_functions(psis,z,F,iat_method='acor')
    np.savetxt('emus_stitching_constants.dat',np.c_[list(range(int(parameters['nWindows']))),z,zerr], fmt='%15.10f')
    r0_k_data, file_list = read_meta(parameters['emus_meta_file'])

    # ----------------------------------------
    # FOR REWEIGHTING, ONLY NEED THE Z ARRAY
    del psis
    del cv_trajs
    del neighbors
    del F
    del zerr
    del zcontribs
    del ztaus

    # ----------------------------------------
    # LOAD IN UNBIASED AND BIASED CV DATA
    print('Starting to load in cv data (original and new)')
    nWindows_range = range(int(parameters['nWindows'])) # assumes windows are numbered with zero-indexing
    data = ['' for i in nWindows_range]
    for i in nWindows_range:
        print('loading window', i)
        temp_biased_data = np.loadtxt(file_list[i])[:,1]
        temp_unbiased_data = np.loadtxt(parameters['unbiased_data_files_naming']%(i))

        if temp_biased_data.shape != temp_unbiased_data.shape:
            print('unbiased data file', i, 'has different number of values than the biased cv data file from the respective window. This should not happen.', temp_unbiased_data.shape, temp_biased_data.shape)
            sys.exit()
        
        temp_data = np.c_[temp_biased_data,temp_unbiased_data]  # biased CV data is row's index 0; unbiased CV data is row's index 1
        data[i] = temp_data

    # ----------------------------------------
    # PREP HISTOGRAM ARRAYS
    xMin = float(parameters['xMin']) 
    xMax = float(parameters['xMax']) 
    xBins = int(parameters['xBins'])
    delta_x = (xMax - xMin)/xBins
    print('Bin width:', delta_x)
    x_half_bins = np.array([xMin + delta_x*(i+0.5) for i in range(xBins)])
    x_edges = np.array([xMin + delta_x*i for i in range(xBins+1)])

    # ----------------------------------------
    # REWEIGHTING BIASED CV FE SURFACE INTO AN UNBIASED CV
    nValues_total = 0
    x_total_fe_counts = np.zeros(xBins)
    for i in nWindows_range:
        nValues = len(data[i])
        nValues_total += nValues

        x_window_counts = np.zeros(xBins)
        x_window_fe_counts = np.zeros(xBins)

        with open('window%03d.frame_weights.dat'%(i),'w') as W:
            for j in range(nValues):
                # ----------------------------------------
                # HISTOGRAMMING DATA POINT
                x_index = int((data[i][j][1] - xMin)/delta_x)

                if x_index < 0 or x_index > xBins:
                    print('!!! 0 > x_index >= xBins ...', data[i][j][0], x_index, i, 'Histogram bounds are not wide enough in the x-dimension. Job failed.')
                    sys.exit()
                elif x_index == xBins:
                    x_index = -1

                # ----------------------------------------
                # ANALYZING DATA POINT IN CONSIDERATION OF CURRENT WINDOW
                w = np.exp((-beta*r0_k_data[i][1])*(data[i][j][0] - r0_k_data[i][0])**2)/z[i]     # exp((-k/2*k_B*T)(r-r0)^2)/z; no volume correction...

                x_window_counts[x_index] += 1
                x_window_fe_counts[x_index] += 1/w

                # ----------------------------------------
                # ANALYZING DATA POINT IN CONSIDERATION OF ALL WINDOWS
                w = 0
                for k in nWindows_range:
                        w+= np.exp((-beta*r0_k_data[k][1])*(data[i][j][0] - r0_k_data[k][0])**2)/z[k]       # exp((-k/2*k_B*T)(r-r0)^2)/z; no volume correction...

                w /= parameters['nWindows'] # <r^2 * exp((-k/2*k_B*T)(r-r0)^2)/z>; average reciprocal boltzmann weight of data point in all possible windows;
                weight = 1./w
                W.write('%15d %15f\n'%(j,weight))
                x_total_fe_counts[x_index] += weight

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

        print('Done with window', i)

    # ----------------------------------------
    # FINISHED REWEIGHTING, RUNNING BOOTSTRAP ANALYSIS TO GET ERROR BARS
    if parameters['bootstrap_bool']:
        print('Beginning bootstrap analysis to approximate error in reweighted CVs')
        original_data = np.empty((0,2))
        for i in nWindows_range:
            original_data = np.concatenate((original_data,np.array(data[i])))

        if original_data.shape[0] != nValues_total:
            print(original_data.shape[0], nValues_total, 'something went wrong during bootstrapping')
            sys.exit()

        x_bootstrap_results = []
        for i in range(parameters['nIterations']):
            print('Starting Step %d of %d steps in bootstrap analysis'%(i,parameters['nIterations']))
            # create bootstrap data
            sample_data = original_data[np.random.randint(nValues_total,size=nValues_total)]
            x_total_fe_bootstrap = np.zeros(xBins)

            # analyze new dataset to get reweighted FE of each bin
            for j in range(nValues_total):
                # ----------------------------------------
                # HISTOGRAMMING DATA POINT
                x_index = int((sample_data[j,1] - xMin)/delta_x)

                if x_index == xBins:
                    x_index = -1

                w = 0
                for k in nWindows_range:
                    w+= np.exp((-beta*r0_k_data[k][1])*(sample_data[j,0] - r0_k_data[k][0])**2)/z[k]       # exp((-k/2*k_B*T)(r-r0)^2)/z; no volume correction...

                w /= parameters['nWindows'] # <exp((-k/2*k_B*T)(r-r0)^2)/z>; average reciprocal boltzmann weight of data point in all possible windows;

                x_total_fe_bootstrap[x_index] += 1/w

            x_total_fe_bootstrap /= delta_x*nValues_total
            x_total_fe_bootstrap = -kT*np.log(x_total_fe_bootstrap)  # no volume correction
            x_total_fe_bootstrap -= np.ndarray.min(x_total_fe_bootstrap)
            x_bootstrap_results.append(x_total_fe_bootstrap)

        ### NOTE: CALCS THE STANDARD DEVIATION OF THE BOOTSTRAPPED DATA
        x_std_error = np.std(np.array(x_bootstrap_results),axis=0)

        np.savetxt('x_axis_error_analysis.dat', x_std_error, fmt='%.10f')

        del original_data
        del sample_data
        del x_total_fe_bootstrap
        del x_bootstrap_results

    # ----------------------------------------
    # FINISHED REWEIGHTING, CLEANING UP VARIABLE SPACE
    del data
    del x_window_counts
    del x_window_prob_density
    del x_window_fe_counts

    # ----------------------------------------
    # FINISHING PLOTTING OF THE REWEIGHTED PROB. DENSITY OF EACH INDIVIDUAL WINDOW - XDATA
    finish_plot(1,parameters['reweighted_x_axis_prob_density_plot_name'],parameters['x_axis_label'],'Probability Density',xlim=(xMin,xMax))

    # ----------------------------------------
    # FINISHING PLOTTING OF THE REWEIGHTED, UNSTITCHED FREE ENERGY - XDATA
    finish_plot(2,parameters['reweighted_x_axis_unstitched_fe_plot_name'],parameters['x_axis_label'],r'Relative Free Energy (kcal mol$^{-1}$)',xlim=(xMin,xMax))

    # ----------------------------------------
    # PLOTTING REWEIGHTED X-DATA FE SURFACE
    print(np.sum(x_total_fe_counts))
    x_total_fe_counts /= delta_x*nValues_total  # no volume correction
    print(np.sum(x_total_fe_counts))

    x_total_fe_counts = -kT*np.log(x_total_fe_counts) # no volume correction
    x_total_fe_counts -= np.ndarray.min(x_total_fe_counts)
    np.savetxt(parameters['reweighted_x_axis_stitched_fe_data_file_name'], np.c_[range(xBins),x_half_bins,x_total_fe_counts], fmt='%15.10f')

    plt.figure(3)
    if parameters['bootstrap_bool']:
        plt.errorbar(x_half_bins[:],x_total_fe_counts[:],yerr=x_std_error,ecolor='r',elinewidth=0.5,zorder=3)
    else:
        plt.plot(x_half_bins[:],x_total_fe_counts[:],zorder=3)
    finish_plot(3,parameters['reweighted_x_axis_stitched_fe_plot_name'], parameters['x_axis_label'], r'Relative Free Energy (kcal mol$^{-1}$)',xlim=(xMin,xMax),ylim=(-0.05,10)) # NOTE

    plt.close()
    print('Done plotting.')

    # ----------------------------------------
    # OUTPUT SUMMARY FILE
    summary('reweighting.summary',sys.argv,parameters)

# ----------------------------------------
# CREATING PARAMETER DICTIONARY
parameters = {}
config_parser(config_file,parameters)

# ----------------------------------------
# MAIN
if __name__ == '__main__':
    main()

