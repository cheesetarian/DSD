"""
 repurposed from code in Dan Jones' OceanClustering from github, Plot.py

Purpose:
    - Almost stand alone module which plots the results to the rest of the program
    - Loads the data from the stored files

#### NOTE ####
If wanting to do more bic sensitivity runs, do so in dj/OceanClustering/Main.py
 and as long as output files are saved in same spot, it's easy enough to
 generate the plot from here.
"""

import numpy as np
import time
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

# call plotBIC using pre-run input data from OceanClustering:
repeated = 35 #10
max_gr = 12  #60
dtype = "norm"
addr = "/home/dudavid/projects/dsd/dj/OceanClustering/" 
pver = '_v3MSs4' # plot version, added DD -- not automated with input data form

#plotBIC(addr, repeated, max_gr, dtype)  # called down below
# if the combo of repeat/max has been run this works


def readBIC(address, repeat_bic):
    # This section reads the information on components, minimum scores and means
    n_mean, n_stdev, n_min = None, None, None
    head_number = 1
    filename = address+"Data_store/Info/BIC_Info.csv"
    csvfile = np.genfromtxt(filename, delimiter=",",skip_header=head_number)
    n_mean  = csvfile[0]
    n_stdev = csvfile[1]
    n_min   = csvfile[2]

    # Read the bic_scores
    bic_r, bic_many, bic_mean, bic_stdev = None, None, None, None
    head_number = 1

    for r in range(repeat_bic):
        filename = address+"Data_store/Info/BIC_r"+str(int(r))+".csv"
        csvfile = np.genfromtxt(filename, delimiter=",",skip_header=head_number)
        if r == 0:
            bic_mean     = csvfile[:,1]
            bic_stdev     = csvfile[:,2]
        bic_r    = csvfile[:,0]
        if r == 0:
            bic_many        = bic_r.reshape(1, bic_r.size)
        else:
            bic_r = bic_r.reshape(1,bic_r.size)
            bic_many = np.vstack([bic_many, bic_r])
            
    print("SHAPE OF BIC MANY AFTER READING = ", bic_many.shape)
    return bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min

###############################################################################    
def plotBIC(address, repeat_bic, max_groups, dtype, trend=False): 
    # Load the data and define variables first
    bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min = None, None, None, None, None, None
    bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min = readBIC(address, repeat_bic)
    n_comp_array = None
    n_comp_array = np.arange(1, max_groups) #, 3) # trying this
    
    print("Calculating n and then averaging across runs, n = ", n_mean, "+-", n_stdev)
    print("Averaging BIC scores and then calculating, n = ", n_min)
    
    # Plot the results
    fig, ax1 = plt.subplots()
    ax1.errorbar(n_comp_array, bic_mean, yerr = bic_stdev, lw = 2, ecolor =\
            'black', label = 'Mean BIC Score', color='red')
    
    if trend:
        # Plot the trendline
        #initial_guess = [20000, 1, 20000, 0.001]
        initial_guess = [47030, 1.553, 23080, 0.0004652]
        #print("initial_guess!?",initial_guess)
        def expfunc(x, a, b, c, d):
            return (a * np.exp(-b * x)) + (c * np.exp(d * x))
        
        popt, pcov, x, y = None, None, None, None
        #popt, pcov = curve_fit(expfunc, n_comp_array, bic_mean, p0 = initial_guess, maxfev=10000)
        popt, pcov = curve_fit(expfunc, n_comp_array, bic_mean, maxfev=10000)
        print("Exponential Parameters = ", *popt)
        x = np.linspace(1, max_groups, 100)
        y = expfunc(x, *popt)
        ax1.plot(x, y, 'r-', label="Exponential Fit")
        
        y_min_index = np.where(y==y.min())[0]
        x_min = (x[y_min_index])[0]
        #ax1.axvline(x=x_min, linestyle=':', color='black', label = 'Exponential Fit min = '+str(np.round_(x_min, decimals=1)))

    # Plot the individual and minimum values
    #ax1.axvline(x=n_mean, linestyle='--', color='black', label = 'n_mean_min = '+str(n_mean))
    #ax1.axvline(x=n_min, linestyle='-.', color='black', label = 'n_bic_min = '+str(n_min))
    for r in range(repeat_bic):
        ax1.plot(n_comp_array, bic_many[r,:], alpha = 0.15, color = 'grey')
        
    ax1.set_ylabel("BIC value")
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_xlabel(r"N$_{GMM}$")
    ax1.grid(True)
    #ax1.set_title("BIC values for GMM with different number of components")
    ax1.set_ylim(min(bic_mean)*0.95, min(bic_mean)*1.08)
    #ax1.legend(loc='best')
    if trend:
        plt.savefig(address+"Plots/BIC_trend"+pver+"_"+dtype+"_"+str(repeat_bic)+".png",bbox_inches="tight",transparent=True)
    else:
        plt.savefig(address+"Plots/BIC"+pver+"_"+dtype+"_"+str(repeat_bic)+"x_"+str(max_groups)+"ng.png",bbox_inches="tight",transparent=True, dpi=350)
    #plt.show()
    ##### if wanting to run solo after things have run:
    #####  Plot.plotBIC('/home/dudavid/projects/dsd/dj/OceanClustering/',35,15, 'MSs3' ) # etc
    
###############################################################################
# Use the VBGMM to determine how many classes we should use in the GMM
def plotWeights(address, run, dtype):
    # Load depth
    depth = None
    depth = Print.readDepth(address, run)

    # Load Weights    
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = Print.readGMMclasses(address, run, depth, 'depth')
    
    n_comp = len(gmm_weights)
    class_array = np.arange(1,n_comp+1)
    
    # Plot weights against class number
    fig, ax1 = plt.subplots()
    ax1.scatter(class_array, np.sort(gmm_weights)[::-1], s = 20, marker = '+', color = 'blue', label = 'Class Weights')
    ax1.axhline(y=1/(n_comp+1), linestyle='-.', color='black', label = '1/(N+1) threshold')
    ax1.axhline(y=1/(n_comp+5), linestyle='--', color='black', label = '1/(N+5) threshold')
    ax1.axhline(y=0.05, linestyle=':', color='black', label = '5% threshold')
    #ax1.axhline(y=1/(n_comp+1), linestyle='-.', color='black', label = str(np.round_(1/(n_comp+1), decimals=3))+' threshold')
    #ax1.axhline(y=1/(n_comp+5), linestyle='--', color='black', label = str(np.round_(1/(n_comp+5), decimals=3))+' threshold')
    ax1.set_xlabel("Class")
    ax1.set_xlim(-1,n_comp)
    ax1.set_ylabel("Weight")
    ax1.grid(True)
    ax1.set_title("VBGMM Class Weights")
    ax1.legend(loc='best')
    plt.savefig(address+"Plots/Weights_VBGMM_"+str(n_comp)+pver+"."+dtype+".pdf", bbox_inches="tight",transparent=True)
    


plotBIC(addr, repeated, max_gr, dtype)  # called down below
print('all done')
