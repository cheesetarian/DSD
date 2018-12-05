# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:36:46 2017

@author: harryholt

Main.py

This script is the central script for my GMM program. It coordinates:
    Load.py
    Print.py
    Plot.py
    GMM.py
    PCA.py
    Bic.py
    Reconstruct.py

The intention is to have a program, comprised of a few modules, which can take
a data set, select a training dataset, a test data set and return:
    GMM model (Classes)
    Labels and probabilities

The should be flexibility in seclecting the Data set, the locations of the
input and output files, the ML algorithm we want to use and the option to loop
over the files/modules themselves
"""
# Import the various modules necessary
import time
import numpy as np

import Load, Print, PCA, GMM, Reconstruct, Bic #,  ClassProperties
import Plot

start_time = time.clock()

""" Define some initial conditions, locations and parameters """
n_comp = 4         # Number of classes in GMM object
n_dimen = 0.998     # Amount of variance retained in PCA
#n_dimen = 0.999     # Amount of variance retained in PCA
loop = False        # Do we want to loop the program
if loop:
    run_total = 8 #25  # Number of times we want to loop

address = "/home/dudavid/projects/dsd/dj/OceanClustering/"     # Location of Program
#filename_raw_data = "/home/dudavid/projects/dsd/dj/OceanClustering/data/allraw.mat"   # Location of raw data
dir_raw_data = "/home/dudavid/projects/dsd/dj/data/"
#address_fronts = "/Users/harryholt/Documents/BAS/data/fronts/"

# Define some booleans to determine what the program should do
run_bic = True #True #False         # Do we want to calculate the BIC scores ?
if run_bic:
    subsample_bic = "random" #"uniform"
    repeat_bic = 35 #50
    max_groups = 15 # Highest number of GMM classes tested
    grid_bic = 4        # size of cell in lat/lon degrees
    conc_bic = 1        # # of samples from each grid
    size_bic = 1500     # Ideal size of the BIC training set -- DOES NOT DO anything rn

subsample_uniform = False  # Indicates how the Training dataset is selected
subsample_random = True  # Indicates how the Training dataset is selected
subsample_inTime = False

dtype = 'norm' # 'lograw', 'raw', 'dm', 'normdm' (norm=1), 'norm' are options -- input to Load
# the input data are not treated locally by this code, but rather as
# preprocessing by collate_raw.ipynb!

grid, conc, fraction_train, inTime_start, inTime_finish = None, None, None, None, None
if subsample_uniform:
    grid = 1        # size of cell in lat/lon degrees
    conc = 1    # # of samples from each grid
if subsample_random:
    fraction_train = .30  # Size of Training dataset as a fraction of whole dataset
if subsample_inTime:
    inTime_start = 0.0      # WRONG AT THE MOMENT
    inTime_finish = 200.0   # WRONG AT THE MOMENT

fraction_nan_samples = 16.0 # If a vertical sample has more than this fraction, it's removed
fraction_nan_depths = 32.0  # If a depth level across all samples has " " " ", it's removed
""" end of initialisation conditions """


###############################################################################
###############################################################################

""" Program """
def main(run=None):
    print("Starting Main.main()")  
    
    # Now start the GMM process
    Load.main(address, dir_raw_data, run, subsample_uniform, subsample_random,\
               subsample_inTime, grid, conc, fraction_train, inTime_start,\
               inTime_finish, fraction_nan_samples, fraction_nan_depths, dtype)
               
    #Load.main(address, filename_raw_data, run, subsample_uniform, subsample_random,\
        # Loads data, selects Train, cleans, centres/standardises, prints
    
    PCA.create(address, run, n_dimen)     # Uses Train to create PCA, prints results, stores object
    GMM.create(address, run, n_comp)      # Uses Train to create GMM, prints results, stores object
   
    PCA.apply(address, run)               # Applies PCA to test dataset     
    GMM.apply(address, run, n_comp)       # Applies GMM to test dataset
    
    # Reconstruction
    Reconstruct.gmm_reconstruct(address, run, n_comp)  # Reconstructs the results in original space
    Reconstruct.full_reconstruct(address, run)
    Reconstruct.train_reconstruct(address, run)

    # new stuff DD 27/08/18, after seeing updates on DJ github
    #mainProperties(address, runIndex, n_comp)

    
    # Plotting -- first commented out DD
    #Plot.plotMapCircular(address, address_fronts, run, n_comp)
    
    #Plot.plotPosterior(address, address_fronts, run, n_comp, plotFronts=True)
    Plot.plotPostZonal(address, run, n_comp, dtype, plotFronts=False) ## zonal frequencies
    #Plot.plotPosterior(address, run, n_comp, dtype, plotFronts=False) ## works but data overlaps spatially...

    Plot.plotProfileClass(address, run, n_comp, dtype, 'uncentred')
    Plot.plotProfileClass(address, run, n_comp, dtype, 'depth')

    Plot.plotGaussiansIndividual(address, run, n_comp, dtype, 'reduced')#uncentred')#'depth')#reduced')
#    Plot.plotGaussiansIndividual(address, run, n_comp, 'depth') # ERROR NOT WOKRING PROPERLY
#    Plot.plotGaussiansIndividual(address, run, n_comp, 'uncentred') # ERROR NOT WOKRING PROPERLY
    
    #Plot.plotProfile(address, run, dtype, 'original') # these run just fine but are huge and unhelpful
    Plot.plotProfile(address, run, dtype, 'uncentred')
    
    Plot.plotWeights(address, run, dtype)
    
""" Opt to run different sections or variations of the program """


# function that only carries out the classification step
#def mainProperties(address, runIndex, n_comp):
#    # calculate class properties, create data frame for later use
#    ClassProperties.main(address, runIndex, n_comp)

def loopMain(run_total):
    for run in range(1,run_total+1):    # Ensures loop starts from run = 1
        print("RUN NUMBER ", str(run))
        main(run)
        
        lon, lat, labels_run = None, None, None
        #lon, lat, varTime, labels_run = None, None, None, None
        lon, lat, labels_run = Print.readLabels(address, run)
        #lon, lat, varTime, labels_run = Print.readLabels(address, run)
        if run == 1:
            labels_loop   = labels_run.reshape(labels_run.size,1)
        else:
            labels_run = labels_run.reshape(labels_run.size,1)
            labels_loop = np.hstack([labels_loop, labels_run])
        print("RUN NUMBER ", str(run)," finish time = ", time.clock() - start_time)
        # Labels shape should be (# profiles, # runs)
    axis = 1
    labels_mostFreq, indices = np.unique(labels_loop, return_inverse=True)
    labels_mostFreq = labels_mostFreq[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(labels_loop.shape),
                                None, np.max(indices) + 1), axis=axis)]    
    Print.printLabels(address, 'Total', lon, lat, labels_mostFreq)
    #Print.printLabels(address, 'Total', lon, lat, varTime, labels_mostFreq)
    del run
#    Plot.loop(address, run_total)   # Plot the results fo the looped program
    

if run_bic:
    #Bic.main(address, filename_raw_data,subsample_bic, repeat_bic, max_groups, \
    Bic.main(address, dir_raw_data, subsample_bic, fraction_train, repeat_bic, max_groups, \
             grid_bic, conc_bic, size_bic, n_dimen, fraction_nan_samples, fraction_nan_depths, dtype)
    
    Plot.plotBIC(address, repeat_bic, max_groups, dtype )
elif loop:
    loopMain(run_total)
else:
    main()

""" Process
    - Choose between BIC and GMM
    - Load, clean, select train
    - Centre train and then USE result to centre overall profiles
    - PCA train and then use result on test/overall array
    - Calculate GMM and store the object
    - Calculate the Labels for the training and test data
    - Print the results to a file for later plotting
    - Score the accuracy of the model using Scotts function
    - Plot the result
    """

print('Main runtime = ', time.clock() - start_time,' s')
    
