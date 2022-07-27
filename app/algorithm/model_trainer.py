#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.utils as utils
from algorithm.model.classifier import Classifier
from algorithm.utils import get_model_config




# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # we are not doing train/valid split 
    train_data = data
    # print('train_data shape:',  train_data.shape) #; sys.exit()
    
    # balance the target classes  
    train_data = get_resampled_data(train_data)
    # print(train_data.head()); sys.exit()    

    # preprocess data
    print("Pre-processing data...")
    train_data, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)       
    train_X, train_y = train_data['X'].astype(np.float), train_data['y'].astype(np.float)
    # print(train_X.shape, train_y.shape) ; sys.exit()  
    
    # Create and train model     
    print('Fitting model ...')  
    model = train_model(train_X, train_y, hyper_params)    
    
    return preprocess_pipe, model


def train_model(train_X, train_y, hyper_params):    
    # get model hyper-parameters parameters 
    #data_based_params = get_data_based_model_params(train_X)
    model_params = {**hyper_params }
    
    # Create and train model   
    model = Classifier(  **model_params )  
    model.fit(train_X, train_y)    
    
    return model


def preprocess_data(train_data, valid_data, data_schema):
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
      
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
    # print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
    return train_data, valid_data, preprocess_pipe 


def get_resampled_data(data):    
    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # samples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time). 
    max_resample = model_cfg["max_resample_of_minority_classes"]
    classes, class_count = np.unique(data["class"], return_counts=True)
    max_obs_count = max(class_count)
    
    resampled_data = []
    for class_, count in zip(classes, class_count):
        if count == 0: continue 
        # find total num_samples to use for this class
        size = max_obs_count if max_obs_count / count < max_resample else count * max_resample
        # if observed class is 50 samples, and we need 125 samples for this class, 
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples
        
        full_samples = size // count        
        idx = data["class"] == class_
        for _ in range(full_samples):
            resampled_data.append(data.loc[idx])
            
        # find the remaining samples to draw randomly
        remaining =  size - count * full_samples   
        
        resampled_data.append(data.loc[idx].sample(n=remaining))
        
    resampled_data = pd.concat(resampled_data, axis=0, ignore_index=True)    
    
    # shuffle the arrays
    resampled_data = resampled_data.sample(frac=1.0, replace=False)
    return resampled_data