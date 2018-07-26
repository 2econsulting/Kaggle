# author: Heo Sung Wook

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def tuneXGB(data, target, grid_params, valid_prop, isReg, objective, params={}, eval_metric="AUC", stopping_rounds=10, cv=3):
    trainData, validData = train_test_split(data, 
                                            test_size=valid_prop, 
                                            stratify=data[target], 
                                            random_state=1)
    
    trainLabel = trainData[target]
    trainData.drop(target, axis = 1, inplace = True)
    
    validLabel = validData[target]
    validData.drop(target, axis = 1, inplace = True)
    
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
    grid.fit(trainData, trainLabel, 
             early_stopping_rounds=stopping_rounds,
             eval_metric=eval_metric, 
             eval_set=[(validData, validLabel)],
             verbose=False)
    
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

gridParams = {'max_depth'        : [7, 9],
              'min_child_weight' : [1, 2, 3],
              'gamma'            : [0.01, 0.1],
              'colsample_bytree' : [0.6, 0.7, 0.8],
              'subsample'        : [0.9, 0.92, 0.94],
              'learning_rate'    : [0.005, 0.01],
              'n_estimators'     : [300, 400, 500]}



best_params = tuneXGB(params = params, 
                      grid_params = gridParams, 
                      objective = "binary:logistic",  ## for binary classification
                      eval_metric = "logloss",
                      stopping_rounds = 50,
                      valid_prop = 0.4,
                      cv = 3, 
                      data = trainDummy, 
                      target = 'Survived', 
                      isReg = False)

best_params

"""

