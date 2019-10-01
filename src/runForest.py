#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:49:17 2019

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import numpy as np
import pandas as pd
import pickle
import os, glob, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse

p = argparse.ArgumentParser()
p.add_argument('-mode', help='State if you want to train a model with feature data ("train"), or load a trained model to predict category labels of unseen data ("predict")',choices=['train', 'predict'], default=None, required=True)
p.add_argument('-features', type=str, help='Path to feature-array (numerical values required) for training or prediction',required=True)
p.add_argument('-rescaling_factors', type=str, help='Provide path to scaling array consisting of one rescaling-factor per feature-column. Alternatively provide a single value by which to rescale all features. If this flag is not provided each feature-column will be rescaled so that all values fall between 0 and 1.',default= 0)
p.add_argument('-labels', type=str, help='[mode "train"] Path to class labels, required for training. If required for mode "predict", this array will be used to determine the values for -n_labels and -outlabels', default= 0)
p.add_argument('-feature_indices', type=str, help='Array of feature indices to select (in case not all features should be used for training or prediction)', default= 0)
p.add_argument('-train_instance_indices', type=str, help='[mode "train"] Array of indices for selecting which instances should be used for training', default= 0)
p.add_argument('-test_size', type=float, help='Fraction of data set aside for testing accuracy', default= 0.2)
p.add_argument('-test_features', type=str, help='In case you have test features in a separate file, provide path here (disables -test_size flag)', default= 0)
p.add_argument('-test_labels', type=str, help='Provide path to labels for the test features provided under -test_features flag', default= 0)
p.add_argument('-trained_model', type=str, help='[mode "predict"] Path to model weights of previously trained network', default= 0)
p.add_argument('-outpath', type=str, help='Provide output path where all output files will be stored', default= 0)
p.add_argument('-seed', type=int, help='Set a seed integer, used for initializing random forest and separating data into training and test set. Set to "random" (default) to draw a random integer between 0 and 1,000,000.', default = 'random')
p.add_argument('-print_labels', type=int, help='[mode "predict"] Print separate output file with labels (off=0/on=1). default = 1', default= 1)
p.add_argument('-n_trees', type=int, help='Set the number of decision trees to be included in the RandomForest model (default=10)', default= 10)
p.add_argument('-select_n_best', type=int, help='This flag will train the model only using the n best features (use the produced output file "final_feature_indices_training_order.txt" to keep track which columns were selected).', default= 0)
p.add_argument('-feature_names', type=str, help='Provide path to array containing feature names, in case you want runForest to report the names of the most informative features', default= 0)
args = p.parse_args()


#args.mode = 'predict'
#args.features = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/split_feature_arrays/test_features_0.10.npy'
#args.labels = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/split_feature_arrays/training_labels_0.90.npy'
##args.test_size = 0.0
#args.seed = 1234
#args.outpath = '/Users/tobias/Desktop/test_predict'
#args.n_trees = 10
#args.rescaling_factors = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/feature_files/training_features/rescaling_array.npy'
#args.select_n_best = 0
#args.feature_names = 0#'/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/feature_files/training_features/feature_labels.npy'
#args.feature_indices = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/target_indices/P1.000C0.100_U_VSTC.txt' #manual_feature_selection_indices.txt
#args.train_instance_indices = 0#'/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/target_indices/instance_indices_P1.000C0.100_U_VSTC.txt'
##args.test_features = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/split_feature_arrays/test_features_0.10.npy'
##args.test_labels = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/split_feature_arrays/test_labels_0.10.npy'
#args.trained_model = '/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/model_out_seed_1234/trained_models/RF/trained_model_RF_10_1396_10_1234.pkl'
#args.print_labels = 1

# select mode
mode = args.mode

# read feature array
feature_array_path = args.features
try:
    features = np.loadtxt(feature_array_path)
except:
    features = np.load(feature_array_path)

# read class label array    
label_array_path = args.labels
if label_array_path:
    try:
        labels = np.load(label_array_path)
    except:
        labels = np.loadtxt(label_array_path,dtype=str)

# rescaling factors (if provided)
rescaling_factor_path = args.rescaling_factors
try:
    rescaling_factors = int(rescaling_factor_path)
    if rescaling_factors == 0:
        rescale = False
    else:
        rescale = True
except:
    rescaling_factors = np.loadtxt(rescaling_factor_path)
    rescale = True

# feature names (if provided)
if args.feature_names:
    try:
        feature_names = np.loadtxt(args.feature_names,dtype=str)
    except:
        feature_names = np.load(args.feature_names)
else:
    feature_names = np.arange(features.shape[1]).astype(str)    
remaining_feature_indices = np.arange(features.shape[1])

# select the user defined features, if provided
if args.feature_indices:
    feature_indices = np.loadtxt(args.feature_indices,dtype=int)
    features = features[:,feature_indices]
    feature_names = feature_names[feature_indices]
    remaining_feature_indices = feature_indices
    rescaling_factors = rescaling_factors[feature_indices]

# select the user defined instances, if provided
if args.train_instance_indices:
    train_instance_indices = np.loadtxt(args.train_instance_indices,dtype=int)
    features = features[train_instance_indices,:]
    labels = labels[train_instance_indices]

if rescale:
    # if rescalign factors are provided, rescale the array accordingly
    scaled_features = features/rescaling_factors
else:
    # rescale the input features usign MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)

# read other model settings
select_n_best = int(args.select_n_best)
n_trees = args.n_trees

# specify the outpath and create dir in case it doens't exists already
outpath = args.outpath
if not os.path.exists(outpath):
    os.makedirs(outpath)

# set the random seed for all numpy operations
if args.seed == 'random':
    random_seed = np.random.randint(0,1000000)
else:
    random_seed = int(args.seed)
np.random.seed(random_seed)


if mode =='train':
    # define training and test set for evaluation
    if not args.test_features:
        test_size = args.test_size
        if test_size == 0:
            scaled_features_training = scaled_features
            labels_training = labels
            scaled_features_test = np.array([])
            labels_test = np.array([])
        else:
            scaled_features_training, scaled_features_test, labels_training, labels_test = train_test_split(scaled_features, labels, test_size=test_size, random_state=random_seed,shuffle=True)
    else:
        scaled_features_training = scaled_features
        labels_training = labels
        if not args.test_features and args.test_labels:
            exit('Please provide -test_labels flag when using -test_features. Alternatively choose -test_size and remove -test_features flag, for runForest to split the input array into training and test set internally.')
        try:
            features_test = np.load(args.test_features)
            labels_test = np.load(args.test_labels)
        except:
            features_test = np.loadtxt(args.test_features)
            labels_test = np.loadtxt(args.test_labels)
        selected_features_test = features_test[:,remaining_feature_indices]
        scaled_features_test = selected_features_test/rescaling_factors

    # select the best labels from training and test array
    if select_n_best:
        # if train, train RF and export feature-weights
        model_random_forest = RandomForestClassifier(random_state=random_seed,n_estimators=n_trees)
        model_random_forest.fit(scaled_features_training, labels_training)        
        # get the feature importances and select the features with highest scores
        feature_importances = model_random_forest.feature_importances_
        stacked_labels_and_weights = np.stack([feature_names,feature_importances]).T
        # check if there are enough features present to selct n best
        if len(feature_importances) >= select_n_best:
            pass
        else:
            print('Warning: Searching for %i best features, but only %i features were found. Proceeding selecting all features (check usage of -feature_indices flag and input array dimensions).'%(select_n_best,len(feature_importances)))
        # sort the 2D array by feature_importances
        sortedArr = stacked_labels_and_weights[stacked_labels_and_weights[:,1].argsort()]
        best_features = sortedArr[-select_n_best:,:][:,0][::-1]
        best_features_indices = [np.where(feature_names==i)[0][0] for i in best_features]
        remaining_feature_indices = remaining_feature_indices[best_features_indices]
        # make subselection of the corresponding features
        selected_scaled_features_training = scaled_features_training[:,best_features_indices]
        selectes_scaled_features_test = scaled_features_test[:,best_features_indices]
        if args.feature_names:
            np.savetxt(os.path.join(outpath,'selected_best_%i_features.txt'%select_n_best),best_features,fmt='%s')
    else:
        selected_scaled_features_training = scaled_features_training
        selectes_scaled_features_test = scaled_features_test

    # train the model
    print('Training model, using %i features'%(selected_scaled_features_training.shape[1]))
    model_random_forest = RandomForestClassifier(random_state=random_seed,n_estimators=n_trees)
    model_random_forest.fit(selected_scaled_features_training, labels_training)
    
    if len(scaled_features_test) > 0:
        # check accuracy on test set
        guessed_labels = model_random_forest.predict(selectes_scaled_features_test)
        random_forest_accuracy = accuracy_score(labels_test, guessed_labels)
        print('Accuracy of training on test set:', random_forest_accuracy)
    # save the trained model for later predictions
    filename = os.path.join(outpath,'trained_model_RF_%i_%i_%i_%i.pkl'%(n_trees,selected_scaled_features_training.shape[0],selected_scaled_features_training.shape[1],random_seed))
    pickle.dump(model_random_forest, open(filename, 'wb'))
    # save the final feature indices, so that same indices can be applied for training
    np.savetxt(os.path.join(outpath,'final_feature_indices_training_order.txt'),remaining_feature_indices,fmt='%i')
        
    
if mode =='predict':
    if not args.trained_model:
        exit('For predicting labels, please load a trained model (*.pkl file), using the -trained_model flag.')
    try:
        filename = args.trained_model
        loaded_model = pickle.load(open(filename, 'rb'))
    except:
        exit('No trained model found. Please provide full path to trained model (*.pkl file) produced by runForest -mode train')
    # predict the category labels of the unseen data
    try:
        guessed_labels = loaded_model.predict(scaled_features)
    except ValueError:
        exit('Predicting category labels failed. Make sure you laoded the correct model and to provide an input array of the correct dimensions (must have same features as training data). In case you used feature selection during training, make sure to provide the indices of the features used for training (use -feature_indices and give path to file "final_feature_indices_training_order.txt", as produced by runForest mode train)')
    except:
        guessed_labels = loaded_model.predict(scaled_features)
    # print guessed labels to file
    if args.feature_indices:
        model_name_string = '_'+os.path.basename(args.feature_indices).replace('.txt','')
    else:
        model_name_string = ''
    if args.print_labels:
        try:
            guessed_labels = guessed_labels.astype(int)
            np.savetxt(os.path.join(outpath,'labels%s.txt'%model_name_string),guessed_labels,fmt='%i')
        except:
            np.savetxt(os.path.join(outpath,'labels%s.txt'%model_name_string),guessed_labels,fmt='%s')
    # calculate probabilities
    label_probabilities = loaded_model.predict_proba(scaled_features)
    np.savetxt(os.path.join(outpath,'label_probabilities%s.txt'%model_name_string),label_probabilities,delimiter='\t',fmt='%.4f',)
    print('Finished estimating class labels for input data. Written to %s' %outpath)
    
    
    

