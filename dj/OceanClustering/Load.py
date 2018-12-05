# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:08:09 2017

@author: harryholt

Load.py

Purpose:
    - Load the data
    - Clean (remove Nan values from profiles and depths)
    - Centre and Standardise
    - Print data to store the results

"""
# Importing modules
import h5py
import numpy as np
import pickle
from sklearn import preprocessing
import time

import Print

start_time = time.clock()


def main(address, dir_raw_data, run, subsample_uniform, subsample_random,\
         subsample_inTime, grid, conc, fraction_train, inTime_start, inTime_finish,\
         fraction_nan_samples, fraction_nan_depths, dtype, run_Bic=False):
    print("Starting Load.main")
    """ Main function for module"""
    lon, lat, Tint = load(dir_raw_data, dtype)  # just keep cts as 'Tint' for now
    print('size of imput lat/lon, cts: ',np.shape(lon),np.shape(Tint))
    #lon, lat, Tint, Sint, varTime = load(dir_raw_data)
    
    # try taking log of counts at outset:
    if dtype == 'lograw' or dtype == 'logdm':
        #Tint[np.where(Tint==0)] = 0.5 # make log(0) linear to log(1):log(2)
        Tint = np.log(Tint)
        Tint[np.where(Tint < 0)] = -1*np.log(2) # set zero counts to -1 in log10 space


    Tint, depth = removeDepthFractionNan(Tint, fraction_nan_depths)
    #Tint, Sint, depth = removeDepthFractionNan(Tint, Sint, fraction_nan_depths)
    lon, lat, Tint = removeSampleFractionNan(lon, lat, Tint, fraction_nan_samples)
    #lon, lat, Tint, Sint, varTime = removeSampleFractionNan(lon, lat, Tint, Sint, varTime, fraction_nan_samples)
    #Tint, Sint = dealwithNan(Tint, Sint)
    #cts = dealwithNan(cts)
    
    ## At this point the data has been successfully cleaned.
    """ Now I need to subselect the training data """
    Tint_train, varTime_train = None, None
    #Tint_train, Sint_train, varTime_train = None, None, None
    if subsample_uniform: # Currently the only option working  --- not being used here!
        lon_train, lat_train, Tint_train, Sint_train, varTime_train = uniformTrain(lon, lat, Tint, Sint, varTime, depth, grid, conc)
    if subsample_random: # Now also written and working
        lon_train, lat_train, Tint_train = randomTrain(lon, lat, Tint, fraction_train) #passing depth doesnt do anything here
        #lon_train, lat_train, Tint_train, Sint_train, varTime_train = randomTrain(lon, lat, Tint, Sint, varTime, depth, fraction_train)
    if subsample_inTime:
        lon_train, lat_train, Tint_train, Sint_train, varTime_train = inTimeTrain(lon, lat, Tint, Sint, varTime, depth, inTime_start, inTime_finish)
    
    ## At this point we should have a training data set to go with the full data set
    """ Now we can centre and standardise the training data, and the whole data will follow """
    # NOTE: currently unsure how to include Sint in the centring
    # I also change the nomenaculature at this point in the code
    """ It is important that the training data set initialises the standardised object !! """
    stand, stand_store, varTrain_centre = centreAndStandardise(address, run, Tint_train)
    var_centre = stand.transform(Tint)  # Centre the full dataset based on the training data set
    
    """ Create the test dataset, by subtracting the set(var_train) from set(var)"""
    """
    lon_test = [x for x in lon if x not in set(lon_train)]
    lat_test = [x for x in lat if x not in set(lat_train)]
    Tint_test = [x for x in Tint if x not in set(Tint_train)]
    Sint_test = [x for x in Sint if x not in set(Sint_train)]
    varTime_test = [x for x in varTime if x not in set(varTime_train)]
    
    varTest_centre = stand.transform(Tint_test)
    """
    # INFORMATION
    # varTrain_centre stores the training, standardised
    # var_centre stores the full standardised data set
    # varTest_centre stores the test dataset
    
    """ Now we can print the results of this process to a file for later use """
#    print("Starting Print")
    print("varTrain_centre.shape = ", varTrain_centre.shape)
    if not run_Bic:
        Print.printLoadToFile(address, run, lon, lat, Tint, var_centre, depth )#, Sint, varTime, \
                              #depth)
        Print.printLoadToFile_Train(address, run, lon_train, lat_train, Tint_train, \
                                varTrain_centre, depth) #, Sint_train, varTime_train, depth)
                                #varTrain_centre, Sint_train, varTime_train, depth)
        #Print.printLoadToFile_Test(address, run, lon_test, lat_test, Tint_test, \
        #                        varTest_centre, Sint_test, varTime_test, depth)
        Print.printDepth(address, run, depth)
    
    if run_Bic:
        return lon_train, lat_train, Tint_train, varTrain_centre #, Sint_train, varTime_train
        #return lon_train, lat_train, Tint_train, varTrain_centre, Sint_train, varTime_train
    
###############################################################################
# Functions which Main uses
def load(dir_raw_data,inny):
    #print("Load.load")
    """ This function loads the raw data from [numpy files]   a .mat file """
    lon, lat, allcts = [], [], []
    #lon, lat, Tint, Sint, varTime = [], [], [], [], []
    # Import lon, lat, Temperature, Salinity and times explicitly
    #variables = ['lon', 'lat', 'Tint', 'Sint','dectime']
    variables = ['lon', 'lat', 'cts']
    #mat = h5py.File(filename_raw_data, variable_names = variables)

    tout = 'MSs3'

    lat = np.load(dir_raw_data+'alllatd'+tout+'.npy')
    lon = np.load(dir_raw_data+'alllond'+tout+'.npy')
    #lat = np.load(dir_raw_data+'shrnk.alllatd4.npy')
    #lon = np.load(dir_raw_data+'shrnk.alllond4.npy')
    if inny == 'raw' or inny == 'lograw': 
        #cts = np.load(dir_raw_data+'sallcts'+tout+'.80.npy').transpose() # smoothed, NOT normed
        cts = np.load(dir_raw_data+'vallcts'+tout+'.80.npy').transpose() # vol-weighted, smoothed, NOT normed
        #cts = np.load(dir_raw_data+'shrnk.allctsVS4.75.npy').transpose() # flip dimensions to match below
    if inny == 'norm': 
        #cts = np.load(dir_raw_data+'shrnk.normallctsVS4.75.npy').transpose() # flip dimensions to match below
        #cts = np.load(dir_raw_data+'normallctsVS4.75.npy').transpose() # flip dimensions to match below
        cts = np.load(dir_raw_data+'rwallcts'+tout+'.80.npy').transpose() # normed by RWC!
        #cts = np.load(dir_raw_data+'normallcts'+tout+'.80.npy').transpose() # flip dimensions to match below
    #if inny == 'dm' or inny == 'logdm': 
    #    cts = np.load(dir_raw_data+'dmnormcts.30.npy').transpose() # flip dimensions to match below
    #if inny == 'normdm': 
    #    cts = np.load(dir_raw_data+'dmnormctsV.30.npy').transpose() # flip dimensions to match below
    #cts = np.load(dir_raw_data+'normallcts.npy').transpose() # flip dimensions to match below
    #print('cts shape: ',np.shape(cts))
    #print('lat shape: ',np.shape(lat))
    byy = 1  #every _ minute (data point)
    bye = 2  #avg/sum together every _ bin
    nbn = 60 # to limit number of size bins used (skipped first 12, so 50=62=3.6mm
    #if inny != 'dm' and inny != 'logdm': nbn = 45
    #nbn = 45 # to limit number of size bins used (skipped first 12, so 50=62=3.6mm

    print("Turned OFF limits on latitude for input data!")
    #hem = abs(lat) < 80 # new, limit to different lat bands (signify in pver in plot)
    #lat = lat[hem]
    #lon = lon[hem]
    #cts = cts[hem,:]

    lat = lat[::byy]
    lon = lon[::byy]
    #  previously used when dealing with raw counts exclusively:
    #if np.max(cts) > 1:  # aka if normalized already by prep routine
    #    cts = cts.astype(int)[::byy,0:nbn]#0:40] # limit while debugging!!!
    #else:
    #    cts = cts[::byy,0:nbn]
    #cts = cts[::byy,0:nbn]
    cts = np.array([ np.mean(cts[:,x*bye:(x*bye+bye)],axis=1) for x in range(int(nbn/bye)) ]).transpose()
    print('LIMITING TO ',nbn,' BINS ')
    print(' and limiting to every ',byy,'point and shrinking every ',bye,'bins')
    #lon = mat["lon"]
    #lon = np.array(lon)[:,0]
    #lat = mat["lat"]
    #lat = np.array(lat)[:,0]
    #Tint = mat["Tint"]
    #Tint = np.array(Tint)
    #Sint = mat["Sint"]
    #Sint = np.array(Sint)
    #varTime = mat["dectime"]
    #varTime = np.array(varTime)
    
    #print("Shape of variable = ", cts.shape)
#    print("axis 0 = ", np.ma.size(Tint, axis=0)) # axis 0 should be ~290520
#    print("axis 1 = ", np.ma.size(Tint, axis=1)) # axis 1 should be ~400
    
    return lon, lat, cts
    #return lon, lat, Tint, Sint, varTime

def removeDepthFractionNan(VAR, fraction_of):
#def removeDepthFractionNan(VAR, VAR2, fraction_of):
    #print("Load.removeDepthFractionNan")
    """ This function removes all depths will a given number of Nan values """
    delete_depth = []
    depth_remain = 1 * np.arange(np.ma.size(VAR, axis=1)) # NO LONGER DEPTHS, ARE INSTEAD JUST NUMBERS OF BINS
    #depth_remain = 5 * np.arange(np.ma.size(VAR, axis=1)) # Create a pressure array to record the pressure of the depths kept in data
    
    for i in range(np.ma.size(VAR, axis=1)):
        var_mean_nan = []
        var_nan_sum = 0
        var_mean_nan = np.isnan(VAR[:,i])
        var_nan_sum = var_mean_nan.sum()
        if var_nan_sum >= np.ma.size(VAR, axis=0)/fraction_of:
            delete_depth.append(i)
            #print(i," DELETED")
            
    delete_depth = np.squeeze(np.asarray(delete_depth))
    #print(delete_depth.shape)
    VAR = np.delete(VAR, (delete_depth), axis=1 )
    #VAR2 = np.delete(VAR2, (delete_depth), axis=1 )
    depth_remain = np.delete(depth_remain, (delete_depth))
    #print("Number of depths deleted above the 1/"+str(fraction_of)+" criterion = ", delete_depth.size)
    #print("Depth pressure remain = ", depth_remain.shape)
    #print("VAR shape after half Sample removed = ", VAR.shape)
    #return VAR, VAR2, depth_remain  
    return VAR, depth_remain  

def removeSampleFractionNan(LON, LAT, VAR, fraction_of):
#def removeSampleFractionNan(LON, LAT, VAR, VAR2, varTime, fraction_of):
    #print("Load.removeSampleFractionNan")
    """ This function removes all profiles will a given number of Nan values """
    delete_sample = []
    sample_count = 0
    for i in range(np.ma.size(VAR, axis=0)):
        var_mean_nan = []
        var_mean_nan = np.isnan(VAR[i,:])
        var_nan_sum = var_mean_nan.sum()        
        if var_nan_sum >= np.ma.size(VAR, axis=1)/fraction_of:
            sample_count = sample_count + 1
            delete_sample.append(i)
            
    delete_sample = np.squeeze(np.asarray(delete_sample))
    #print(delete_sample.shape)
    VAR = np.delete(VAR, (delete_sample), axis=0 )
    #VAR2 = np.delete(VAR2, (delete_sample), axis=0 )
    #varTime = np.delete(varTime, (delete_sample))
    LON = np.delete(LON, (delete_sample))
    LAT = np.delete(LAT, (delete_sample))   
    #print("Number of samples deleted above the 1/"+str(fraction_of)+" criterion = ", sample_count)
    #print("VAR shape after half Col removed = ", VAR.shape)
    #print("LON shape = ", LON.shape)
    #print("LAT shape = ", LAT.shape)
    #return LON, LAT, VAR, VAR2, varTime
    return LON, LAT, VAR #, varTime

#def dealwithNan(VAR, VAR2):
def dealwithNan(VAR): #, VAR2):
    print("Load.dealwithNan")
    """ Simple Linear interpolator to deal with th eremaining Nan valus """
    #print("total number of values before nan interpolation = ", np.size(VAR))
    #print("number of nans before nan interpolation = ",np.isnan(VAR).sum())
    for i in range(np.ma.size(VAR, axis=0)):
        mask = []
        VAR_temp = []
        VAR_temp = VAR[i,:]
        mask = np.isnan(VAR_temp)
        VAR_temp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), VAR_temp[~mask])
        VAR[i,:] = VAR_temp
    """
    for i in range(np.ma.size(VAR2, axis=0)):
        mask2 = []
        VAR2_temp = []
        VAR2_temp = VAR2[i,:]
        mask2 = np.isnan(VAR2_temp)
        VAR2_temp[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2), VAR2_temp[~mask2])
        VAR2[i,:] = VAR2_temp
    """
    #print("number of nans after = ",np.isnan(VAR).sum())
    return VAR #, VAR2

###############################################################################

def uniformTrain(lon, lat, VAR, VAR2, varTime, depth, grid, concentration):
    #print("Load.uniformTrain")
    i_array = np.arange(-180, 180, grid ,dtype=np.int32)    # array of intergers from -180 to 180
                             # Change to 180
    array_lon, array_lat, array_time = [], [], []
    count = 0
    # Loop over the Longitudinal values
    for i in i_array:
        indices_lon = []
        indices_lon = np.squeeze( np.nonzero((lon>=i)&(lon<i+grid) ) )  # indices of the lon values that match the criteria i to i + grid
        
        lon_temp_i = lon[indices_lon] # lon values that match the criteria
        lat_temp_i = lat[indices_lon] # lat values corresponding to lon values that match the criteria
        time_temp_i = varTime[indices_lon] # time values corresponding to lon values that match the criteria
        
        if indices_lon.size == 0:
            #print("Indices_lon is null for range ", i, " + ", grid)
            continue
        j_array = []
        j_array = np.arange(min(np.floor(lat[indices_lon])), max(np.ceil(lat[indices_lon])), grid ,dtype=np.int32) # array of latitude values for each longitude
        #print("j_array = ", j_array)
        
        # Loop over the Latitudes for a given longitude
        for j in j_array:
            indices_lat = []
            indices_lat = np.squeeze( np.nonzero((lat_temp_i >= j) & (lat_temp_i < j + grid)) ) # Indices of the lat values that match the criteria j to j+grid
            
            lon_temp_j = lon_temp_i[indices_lat] # lon values that match both lon and lat criteria
            lat_temp_j = lat_temp_i[indices_lat] # lat values that match both lon and lat criteria
            time_temp_j = time_temp_i[indices_lat] # time values that match both lon and lat criteria
            #print("Indices: ", indices_lat.size , indices_lat)
            
            if indices_lat.size == 0:
                #print("Indices_lat is null for range ",i , j, " + ",grid)
                continue
            if indices_lat.size == 1:
                #print("Converting # into [#] for indices_lat")
                indices_lat = np.asarray([indices_lat])
                lon_temp_j = np.asarray([lon_temp_j])
                lat_temp_j = np.asarray([lat_temp_j])
                time_temp_j = np.asarray([time_temp_j])
                
            select = None   # Reset the select variable for every iteration
            select = np.random.randint(indices_lat.size, size = concentration)
            #print(i, j, " select = ", select, "indices_lat[select] = ", indices_lat[select])
            
            # Select the training sample
            lon_select = lon_temp_j[select] # single lon value for selected sample
            lat_select = lat_temp_j[select] # single lat value for selected sample
            time_select = time_temp_j[select] # single time value for selected sample
            
            for col_depth in range(0,len(depth)):
                variable_col, variable2_col = [], []
                variable_col = VAR[:,col_depth]
                variable2_col = VAR2[:,col_depth]
                variable_train_lon = variable_col[indices_lon]
                variable2_train_lon = variable2_col[indices_lon]
                variable_train_lat = variable_train_lon[indices_lat]
                variable2_train_lat = variable2_train_lon[indices_lat]
                variable_train = variable_train_lat[select] # single depth sample selected from criteria
                variable2_train = variable2_train_lat[select]
                
                if col_depth == 0:
                    var_train_sample = variable_train.reshape(variable_train.size,1)
                    var2_train_sample = variable2_train.reshape(variable2_train.size,1)
                else:
                    var_train = variable_train.reshape(variable_train.size,1)
                    var_train_sample = np.hstack([var_train_sample, var_train])
                    
                    var2_train = variable2_train.reshape(variable2_train.size,1)
                    var2_train_sample = np.hstack([var2_train_sample, var2_train])
                    
                # var_train_sample has whole sample selected from criteria

            array_lon = np.append(array_lon,lon_select)
            array_lat = np.append(array_lat,lat_select)
            array_time = np.append(array_time, time_select)
            
            if i == i_array[0] and j == j_array[0]:
                var_train_array = var_train_sample
                var2_train_array = var2_train_sample
            else:
                var_train_array = np.vstack([var_train_array,var_train_sample])
                var2_train_array = np.vstack([var2_train_array,var2_train_sample])
            
            count = count + concentration
    
    array_lon = np.asarray(array_lon)
    array_lon = np.squeeze(array_lon.reshape(1,array_lon.size))
    array_lat = np.asarray(array_lat)
    array_lat = np.squeeze(array_lat.reshape(1,array_lat.size))
    array_time = np.asarray(array_time)
    array_time = np.squeeze(array_time.reshape(1,array_time.size))    

    print("var_train_array.shape = ", var_train_array.shape)
    
    return array_lon, array_lat, var_train_array, var2_train_array, array_time

###############################################################################
#def randomTrain(lon, lat, Tint, Sint, varTime, depth, fraction_train):
def randomTrain(lon, lat, Tint, fraction_train):
    #lon_rand, lat_rand, Tint_rand, Sint_rand, varTime_rand = None, None, None, None, None, None
    lon_rand, lat_rand, Tint_rand = None, None, None
    array_size = np.ma.size(Tint, axis = 0)   # Expecting around 280,000
    print(array_size,fraction_train)#number_rand)
    number_rand = round(fraction_train * array_size)
    
    indices_rand = None
    indices_rand = np.random.randint(0, high=array_size, size = number_rand)
    
    lon_rand = lon[indices_rand]
    lat_rand = lat[indices_rand]
    Tint_rand = Tint[indices_rand, :]
    #Sint_rand = Sint[indices_rand, :]
    #varTime_rand = varTime[indices_rand]
    
    print("Tint_rand.shape = ", Tint_rand.shape)
    
    #return lon_rand, lat_rand, Tint_rand, Sint_rand, varTime_rand
    return lon_rand, lat_rand, Tint_rand #, Sint_rand, varTime_rand
###############################################################################

"""

def inTimeTrain(lon, lat, Tint, Sint, varTime, depth, inTime_start, inTime_finish):
    lon_train, lat_train, Tint_train, Sint_train, varTime_train = None, None, None, None, None
    
    indices_time = None
    indices_time = np.logical_and(varTime > inTime_start, varTime < inTime_finish)
    
    lon_train = lon[indices_time]
    lat_train = lat[indices_time]
    Tint_train = Tint[indices_time, :]
    Sint_train = Sint[indices_time, :]
    varTime_train = varTime[indices_time]
    
    print("Tint_train.shape = ", Tint_train.shape)
    
    return lon_train, lat_train, Tint_train, Sint_train, varTime_train
"""
###############################################################################
def centreAndStandardise(address, run, VAR):
    print("Load.centreAndStandardise")
    """ Function to creat a standardised object using the training data set """
    stand_store = address+"Objects/Scale_object.pkl"
    stand = preprocessing.StandardScaler()
    stand.fit(VAR)  # VAR = Tint_train so that stand is initiaised with the traing dataset
    var_stand = stand.transform(VAR)
    
    with open(stand_store, 'wb') as output:
        ScaleObject = stand
        pickle.dump(ScaleObject, output, pickle.HIGHEST_PROTOCOL)
    del ScaleObject
    
    return stand, stand_store, var_stand
    

print('Load runtime = ', time.clock() - start_time,' s')
