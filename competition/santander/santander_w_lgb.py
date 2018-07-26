# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 00:54:49 2018
https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
@author: jacob
"""

# library
import pandas as pd
import numpy as np 
from fe_santaander import get_data, add_statistics, get_selected_features 
#from tuneLGB import RandomTuneLGB, CartesianTuneLGB

# Get the data
data, test = get_data()

# Get target and ids 
y = data[['ID','target']].copy()
del data['target'], data['ID']
sub = test[['ID']].copy()
del test['ID']

# Add features
data, test = add_statistics(data, test)
bak = data.copy()

# feature selection
#features = get_selected_features() + ['nb_nans', 'the_median', 'the_mean', 'the_sum', 'the_std', 'the_kur']
#features = pd.unique(features).tolist() 
#data = data[features].copy()
#test = test[features].copy()

# ...
data = pd.concat([y,data],axis=1)
del data['ID']
data.head()


# -------------------------------
# step1 : find optimal max_depth
# -------------------------------
from tuneLGB import RandomTuneLGB, CartesianTuneLGB

# grid_params 
grid_params = {
  'max_depth' : [-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
}

# tuneLGB
grid, pred = CartesianTuneLGB(
    data = data,
    test = test,
    valid_prop = 0.4,
    target = "target", 
    grid_params = grid_params, 
    objective = "regression"    
)
print(">> best max_depth :", grid.best_params_["max_depth"])
grid.cv_results_['params']
grid.cv_results_['mean_test_score']


# -------------------------------
# submit
# -------------------------------
sub['target'] = np.expm1(pred)
sub.to_csv("./output/sub3.csv",index=False)


# -------------------------------
# step2 : find optimal combination 
# -------------------------------
# grid_params 
grid_params = {
  'max_depth'        : [4],
  'learning_rate'    : [0.05, 0.01],
  'colsample_bytree' : [0.6453, 0.5, 0.7, 0.9],
  'subsample'        : [0.6143, 0.5, 0.7, 0.9],
  'reg_alpha'        : [np.power(10, -2.2887), 0.001, 0.01, 0],
  'reg_lambda'       : [np.power(10, 1.7570), 15, 10, 5, 0],
  'num_leaves'       : [58, 30, 90],
  'min_split_gain'   : [np.power(10, -2.5988)],
  'min_child_weight' : [np.power(10, -0.1477)]
}

# tuneLGB
grid, pred = RandomTuneLGB(
    data = data,
    test = test,
    valid_prop = 0.4,
    target = "target", 
    grid_params = grid_params, 
    objective = "regression"    
)


# -------------------------------
# submit
# -------------------------------
sub['target'] = np.expm1(pred)
sub.to_csv("./output/sub4.csv",index=False)






