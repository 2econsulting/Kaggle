# author: Heo Sung Wook

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def tuneXGB(params, grid_params, objective, cv, trainData, trainLabel, isReg):
    # check where Regression or not
    if isReg == True:
        mdl = xgb.XGBRegressor(boosting_type= "gbdt", 
                               objective = objective,
                               n_jobs = 4,
                               silent = False,
                               random_state = 1234,
                               scale_pos_weight = params["scale_pos_weight"])
    else:
        mdl = xgb.XGBClassifier(boosting_type= "gbdt", 
                                objective = objective,
                                n_jobs = 4, 
                                silent = False,
                                random_state = 1234,
                                scale_pos_weight = params["scale_pos_weight"])
        
    # Create the grid
    grid = GridSearchCV(mdl, grid_params, verbose=2, cv=cv, n_jobs=4)
    
    # Run the grid
    grid.fit(trainData, trainLabel)
    
    params['min_child_weight'] = grid.best_params_['min_child_weight']
    params['max_depth'] = grid.best_params_['max_depth']
    params['gamma'] = grid.best_params_['gamma']
    params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    params['subsample'] = grid.best_params_['subsample']
    params['learning_rate'] = grid.best_params_['learning_rate']
    params['n_estimators'] = grid.best_params_['n_estimators']
    
    return params


""" 
##### example #####
## after run "prepXGB.py" example code 

params = {}
params['scale_pos_weight'] = 1

gridParams = {'max_depth'        : [3, 5, 7, 9, 11],
              'min_child_weight' : [1, 3, 5],
              'gamma'            : [0.05, 0.01, 0.5, 0.1],
              'colsample_bytree' : [0.8, 0.9, 1.0],
              'subsample'        : [0.8, 0.9, 1.0],
              'learning_rate'    : [0.001, 0.01, 0.1],
              'n_estimators'     : [500, 700, 900]}

best_params = tuneXGB(params = params, 
                      grid_params = gridParams, 
                      objective = "binary:logistic",  ## for binary classification
                      cv = 3, 
                      trainData = trainDummy.drop("Survived", axis = 1), 
                      trainLabel = trainDummy['Survived'], 
                      isReg = False)

best_params

"""
