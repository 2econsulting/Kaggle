# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:53:11 2018

@author: hsw
"""
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.random_forest import H2ORandomForestEstimator

h2o.init()

raw_train = pd.read_csv("./raw_data/train.csv")

# check var
raw_train_var = raw_train.var()

# zero variance index
none_0_idx = np.where(raw_train_var != 0)[0]

# zero variance index remove
raw_train_rm = raw_train.iloc[:,none_0_idx]


raw_train_hex = h2o.H2OFrame(raw_train_rm)

# default random forest
ml_rf = H2ORandomForestEstimator(
    seed=1234)

X = raw_train_hex.col_names[2:]     
y = raw_train_hex.col_names[1]    

ml_rf.train(X, y, raw_train_hex)

rf_varimp = ml_rf.varimp(use_pandas=True)

# select column which has upper percentage than 1/4736(the number of column)
rf_varimp_col = rf_varimp['variable'][:sum(rf_varimp['percentage'] >= 0.0002)]


raw_train_varimp = raw_train_rm[rf_varimp_col]
raw_train_varimp = pd.concat([raw_train_rm[['target']], raw_train_varimp], axis = 1)
