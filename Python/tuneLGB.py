# author: Heo Sung Wook

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

def tuneLGB(params, grid_params, objective, cv, trainData, trainLabel, isReg):
    # check where Regression or not
    if isReg == True:
        mdl = lgb.LGBMRegressor(
                                 boosting_type= 'gbdt', 
                                 objective = objective,
                                 n_jobs = 4,
                                 silent = False,
                                 random_state = 1234,
                                 n_estimators = params['n_estimators'],
                                 subsample_for_bin = params['subsample_for_bin'],
                                 subsample = params['subsample'], 
                                 subsample_freq = params['subsample_freq'],
                                 min_split_gain = params['min_split_gain'], 
                                 min_child_weight = params['min_child_weight'], 
                                 min_child_samples = params['min_child_samples'], 
                                 scale_pos_weight = params['scale_pos_weight']
                               )
    else:
        mdl = lgb.LGBMClassifier(
                                  boosting_type= 'gbdt', 
                                  objective = objective,
                                  n_jobs = 4, 
                                  silent = False,
                                  random_state = 1234,
                                  n_estimators = params['n_estimators'],
                                  subsample_for_bin = params['subsample_for_bin'],
                                  subsample = params['subsample'], 
                                  subsample_freq = params['subsample_freq'],
                                  min_split_gain = params['min_split_gain'], 
                                  min_child_weight = params['min_child_weight'], 
                                  min_child_samples = params['min_child_samples'], 
                                  scale_pos_weight = params['scale_pos_weight']
                                )
        
    # Create the grid
    grid = GridSearchCV(mdl, grid_params, verbose=2, cv=cv, n_jobs=4)
    
    # Run the grid
    grid.fit(trainData, trainLabel)
    
    params['min_child_samples'] = grid.best_params_['min_child_samples']
    params['max_depth'] = grid.best_params_['max_depth']
    params['learning_rate'] = grid.best_params_['learning_rate'] 
    params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    params['subsample'] = grid.best_params_['subsample']
    params['reg_alpha'] = grid.best_params_['reg_alpha']
    params['reg_lambda'] = grid.best_params_['reg_lambda']
    
    return params


""" 
****** example ******
params = {}
params['n_estimators'] = 500
params['subsample_for_bin'] = 200
params['subsample'] = 1
params['subsample_freq'] = 1
params['min_split_gain'] = 0.5
params['min_child_weight'] = 1
params['min_child_samples'] = 1
params['scale_pos_weight'] = 1

gridParams = {
               'max_depth'        : [3, 5, 7, 9, 11],
               'learning_rate'    : [0.05, 0.01, 0.5, 0.1],
               'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
               'subsample'        : [0.8, 0.9, 1.0],
               'reg_alpha'        : [0, 0.1, 0.05, 0.001],
               'reg_lambda'       : [0, 0.1, 0.05, 0.001],
               'min_child_samples': [1, 2, 3, 4, 5]
             }

drop_column = ["Cabin", "Name", "Ticket"]

best_params = tuneLGB(
                       params = params, 
                       grid_params = gridParams, 
                       objective = "binary", 
                       cv = 3, 
                       trainData = train.drop(drop_column, axis = 1), 
                       trainLabel = train['Survived'], 
                       isReg = False
                      )
"""