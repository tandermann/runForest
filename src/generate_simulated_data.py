#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:25:58 2019

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import os
import numpy as np
from sklearn.datasets import make_classification
seed=1234
np.random.seed(seed)

n_features = 10

script_path = os.path.dirname(os.path.realpath(__file__))

X,y = make_classification(n_samples=10000, # The number of samples (data points) to simulate.
                          n_features=n_features, # The total number of features to simulate.
                          n_informative=5, # The number of informative features.
                          n_redundant=0, # The number of redundant features. These features are generated as random linear combinations of the informative features.
                          n_repeated=0, # The number of duplicated features, drawn randomly from the informative and the redundant features.
                          n_classes=4, # The number of classes (or labels) of the classification problem.
                          n_clusters_per_class=1, # The number of clusters per class.
                          class_sep=1.5, # The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                          flip_y=0, # The fraction of samples whose class are randomly exchanged. Larger values introduce noise in the labels and make the classification task harder.
                          weights=[0.25,0.25,0.25,0.25], # The proportions of samples assigned to each class. If None, then classes are balanced. Note that if len(weights) == n_classes - 1, then the last class weight is automatically inferred. More than n_samples samples may be returned if the sum of weights exceeds 1.
                          random_state=seed
                          )

feature_names = ['feature%i'%i for i in range(n_features)]

training_data, prediction_data = np.split(X,2)
training_labels, prediction_labels = np.split(y,2)

randomely_selected_features = np.random.choice(np.arange(n_features),int(np.round(n_features/3)),replace=False)

np.savetxt('%s/../example_files/training_data.txt'%script_path,training_data,fmt='%.5f')
np.savetxt('%s/../example_files/unknown_data.txt'%script_path,prediction_data,fmt='%.5f')
np.savetxt('%s/../example_files/training_labels.txt'%script_path,training_labels,fmt='%i')
np.savetxt('%s/../example_files/unknown_labels.txt'%script_path,prediction_labels,fmt='%i')
np.savetxt('%s/../example_files/feature_names.txt'%script_path,feature_names,fmt='%s')
np.savetxt('%s/../example_files/manual_feature_selection_indices.txt'%script_path,randomely_selected_features,fmt='%i')
