# -*- coding: utf-8 -*-
"""
Since other files from git are designed to be run via j-notebooks or such and
do not work locally as py files simply pulling code from git notebook to make 
local py file that works

Update 2019-08-05: Commenting through some lines just for sake of my own memory

"""



import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
import scipy.ndimage.filters as filt
from sklearn import svm
from sklearn.linear_model import Perceptron
from tqdm import tqdm

import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *

## Files for vocalizations and noise
load_noise = False #toggle whether noises is generated or pulled in from a pre-generated file
noiselen = 100000 #if loading in a pre-generated file, only take this many samples
# noise = np.random.randn(noiselen) #old way of generating noise
noise = None #toggle for whether testing with noise or not

vocal_files = ['basic.wav', 'altered.wav'] #set names of files to be played/trained and tested on
noise_file = 'Matlab_SoundTextureSynth/Output_Folder/Bubbling_water_10111010100.wav' #file to load for nosie
num_vocals = len(vocal_files) #for use later, get number of unique stimulus files loaded

## Parameters for vocalization and noise preprocessing
signal_to_noise_ratio = 50 #unclear if scales on moment by moment amplitude or by power (i.e. intergrated energy across frequencies)
gfb = g.GammatoneFilterbank(order=1, density = 1.0, startband = -21, endband = 21, normfreq = 2200) #sets up parameters for our gammatone filter model of the cochlea.
                            #Need to look at documentation to figure out exactly how these parameters work , but normfreq at least seems to be central frequency from
                            #which the rest of the fitler a distributed (accoruding to startband and endband)
plot_gammatone_transformed = True #toggle to plot output of gammatone filtered stimulus
plot_temporal_filters = False #toggle to plot temporal filters (i.e. temporal component of STRF)
plot_temporal_transformed = True #toggle to plot signal after being gammatone filtered and temporally filtered 
down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  I believe mostly reduces resolution of frequency
down_sample_pre = 10 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude)
down_sample_post = 10 #Factor by which to reduce Fs after applying filters

## Parameters for training data
num_samples = num_vocals * 1 #choose how many times you see each stimulus
gaps = True #toggle whether there can be gaps between presentation of each stimulus
min_gap = 25 #sets min range of gap in units of samples (?)
max_gap = 100 #set max range of gap in units of samples (?)
apply_noise = False #toggle for applying noise

## Parameters for testing data
test_noise = False #unclear, I guess toggle for adding unique noise in test case that is different from training case?
plot_test = True #plotting toggle for ?
plot_features = True #plotting toggle for filters found by SFA

classifier_baseline = Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
classifier_SFA = Perceptron(max_iter = 10000, tol = 0.001)
classifier_features = 2 #how many features from SFA  SFA-Perceptron gets to use
baseline_features = 'all' #how many features the Perceptron by itself gets to use

## Load in files

vocalizations = get_data(vocal_files)
print('Vocalizations Loaded...')

##Load in and adjust noise power accordingly to sigal to noise ratio

if(load_noise):
    noise, _ = sf.read(noise_file)

print('Noises loaded...')
print('Ready for preprocessing.')

if noise is not None:
    noise = scale_noise(vocalizations,noise,signal_to_noise_ratio)
    noise = noise[:noiselen]
print('Noise Scaled...')
print('Ready For Gammatone Transform')

## Apply Gammatone Transform to signal and noise

vocals_transformed = gamma_transform_list(vocalizations, gfb)
print('Vocalizations Transformed...')

if noise is not None:
    noise_transformed = gamma_transform(noise, gfb)
    print('Noise Transformed...')
    
## Down sample for computation tractablility
#reeval gammatone transform accordingly
    
if down_sample:
    for i,vocal in enumerate(vocals_transformed):
        vocals_transformed[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)

if(plot_gammatone_transformed):
    for i,vocal in enumerate(vocals_transformed):
        plot_input(vocal, vocal_files[i])
    if noise is not None:
        plot_input(noise_transformed, 'Noise')
    
print('Ready For Temporal Filters')

## Apply temporal filters

tFilter = temporalFilter()
tFilter2 = np.repeat(tFilter,3)/3 #slightly unlear what is going on here
tFilters = [tFilter, tFilter2]

if(plot_temporal_filters):
    plt.plot(tFilter)
    plt.plot(tFilter2)
    plt.show()

vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
print('Vocals Temporally Filtered...')

if noise is not None:
    noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
    print('Noise Temporally Filtered')

#again re-evaluate if down sampled
    
if down_sample:
    for i,vocal in enumerate(vocals_temporal_transformed):
        vocals_temporal_transformed[i] = vocal[:,::down_sample_post] #I guess this does a separate down sample after the temporal filters have been applied?
    if noise is not None:
        noise_temporal_transformed = noise_temporal_transformed[:,::down_sample_post]
        
if(plot_temporal_transformed):
    for i,vocal in enumerate(vocals_temporal_transformed):
        plot_input(vocal, vocal_files[i])
    
    if noise is not None:
        plot_input(noise_temporal_transformed, 'Noise')

print('Ready For SFA')

## Create Training Dataset

samples = np.random.randint(num_vocals, size = num_samples)

training_data = None
initialized = False
for i in tqdm(samples):
    if(not(initialized)):
        training_data = vocals_temporal_transformed[i]
        initialized = True
    else:
        training_data = np.concatenate((training_data, vocals_temporal_transformed[i]),1)
        
    if(gaps):
        training_data = np.concatenate((training_data, np.zeros((training_data.shape[0], np.random.randint(min_gap,max_gap)))),1)     
print('Data arranged...')
if(apply_noise):
    while(noise_temporal_transformed[0].size < training_data[0].size):
        noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
    training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
    print('Applied Noise...')
else:
    print('No Noise Applied...')

print('Ready For SFA')

## Train SFA On Data, two layers in this example


(layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1', transform = True)
print('SFA Training Complete')

data = np.vstack((layer1[:,5:], layer1[:,:-5]))

(mean2, variance2, data_SS2, weights2) = getSF(data, 'Layer2')

## Test Results

samples = np.arange(num_vocals)

testing_data = None
initialized = False
for i in tqdm(samples):
    if(not(initialized)):
        testing_data = vocals_temporal_transformed[i]
        initialized = True
    else:
        testing_data = np.concatenate((testing_data, vocals_temporal_transformed[i]),1) 
print('Data arranged...')

if(test_noise):
    testing_data = testing_data + noise_temporal_transformed[:,0:testing_data[0].size]
    print('Applied Noise...')
else:
    print('No Noise Applied...')

if(plot_test):
    plot_input(testing_data, 'Testing Data')
print('Testing Data Ready')

## Apply SFA to Test Data, also toggles for using second layer

test = testSF(testing_data, 'Layer 1', mean, variance, data_SS, weights)
# print('SFA Applied To Test Set')
# test = np.vstack((test[:,5:], test[:,:-5]))
# test = testSF(test, 'Layer 2', mean2, variance2, data_SS2, weights2)

## Plot SFA features

labels = getlabels(vocals_temporal_transformed)
if(plot_features):
    for i in range(4):
        plt.plot(test[i])
        plt.plot(labels)
        plt.show() 
    print('SFA Features Plotted')
else:
    print('Skipping Feature Plotting')
    
## Compare SFA With Baseline For Linear Classification
    
print('SFA Based Classifier with ', classifier_features, ' features')
classifier_SFA.fit(test[:classifier_features].T,labels)
print(classifier_SFA.score(test[:classifier_features].T,labels), '\n')

print('Baseline Classifier with ', baseline_features, ' features')
classifier_baseline.fit(testing_data.T,labels)
print(classifier_baseline.score(testing_data.T,labels))

##Plot it

SFAClassifiedPlot(test,classifier_SFA,labels[:-5])
