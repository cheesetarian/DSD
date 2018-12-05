# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:59:59 2017

@author: harryholt

GMM.py

Purpose:
    - Generate the GMM object in reduced space using X_train
    - Retain information on the classes in reduced and reconstructed space
    - Assign labels and posterior probabilities to the full training dataset X
    
"""
import pickle
from sklearn import mixture
import numpy as np
import time

import Print

start_time = time.clock()

def create(address, run, n_comp):
    print("GMM.create")
    """ Takes the training dataset and creates the GMM object """
    # Load col_reduced
    col_reduced = None
    col_reduced = Print.readColreduced(address, run)
    col_reduced_array = np.arange(col_reduced)
    
    # Load Training data in reduced pca space
    lon_train, lat_train, X_train_array, varTime_train = None, None, None, None
    lon_train, lat_train, X_train_array = Print.readPCAFromFile_Train(address, run, col_reduced)
    #print(np.shape(X_train_array))
    #lon_train, lat_train, X_train_array, varTime_train = Print.readPCAFromFile_Train(address, run, col_reduced)
    
    # Calculate GMM Object
    gmm, gmm_weights, gmm_means, gmm_covariances = None, None, None, None
    gmm, gmm_weights, gmm_means, gmm_covariances = GaussianMixtureModel(address, run, n_comp, X_train_array)
    
    """ Print the information on the classes to a file """
    class_number_array = np.arange(0,n_comp).reshape(-1,1)
    Print.printGMMclasses(address, run, class_number_array, gmm_weights, \
                          gmm_means, gmm_covariances, col_reduced_array, 'reduced')
    
###############################################################################
def apply(address, run, n_comp):
    print("GMM.apply")
    # Load col_reduced value
    col_reduced = None
    col_reduced = Print.readColreduced(address, run)
    
    # Load full data array - X
    lon, lat, X_array, varTime = None, None, None, None
    lon, lat, X_array = Print.readPCAFromFile(address, run, col_reduced)
    #lon, lat, X_array, varTime = Print.readPCAFromFile(address, run, col_reduced)
    
    # Load GMM object
    gmm = None
    with open(address+'Objects/GMM_Object.pkl', 'rb') as input:
        gmm = pickle.load(input)
    
    # Calculate the labels and probabilities of the profiles
    labels, post_prob = None, None
    labels = gmm.predict(X_array)  # Expected shaope (n_profiles,)
    post_prob = gmm.predict_proba(X_array) # Expected shape (n_profiles, classes)
    class_number_array = np.arange(0,n_comp).reshape(-1,1)
    
    # Print Labels and probabilities to file
    Print.printLabels(address, run, lon, lat, labels)
    Print.printPosteriorProb(address, run, lon, lat, post_prob, class_number_array)
    #Print.printLabels(address, run, lon, lat, varTime, labels)
    #Print.printPosteriorProb(address, run, lon, lat, varTime, post_prob, class_number_array)
    
    
def GaussianMixtureModel(address, run, n_comp, X_train):
    print("GMM.GaussianMixtureModel")
    gmm = None
    gmm = mixture.GaussianMixture(n_components = n_comp, covariance_type = 'diag')
#    gmm = mixture.BayesianGaussianMixture(n_components = n_comp, covariance_type = 'diag')
    gmm.fit(X_train)
    
    #### Attempt to store the GMM object
    gmm_store = address+"Objects/GMM_Object.pkl"
    with open(gmm_store, 'wb') as output:
        gmmObject = gmm
        pickle.dump(gmmObject, output, pickle.HIGHEST_PROTOCOL)
    del gmmObject
    
    weights, means, covariances = None, None, None
    weights = np.squeeze(gmm.weights_)  # shape (n_components)
    # NOTE: weights is the same for each col_red
    means = np.squeeze(gmm.means_)  # shape (n_components, n_features)
    covariances = abs(np.squeeze(gmm.covariances_))  # shape (n_components, n_features) 

    print('Bic score: ', gmm.bic(X_train))
    
    return gmm, weights, means, covariances

print('GMM runtime = ', time.clock() - start_time,' s')
