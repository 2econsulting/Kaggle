# author: Heo Sung Wook

import catboost as cbt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def tuneCBT(data, target, grid_params, valid_prop, isReg, params={}, eval_metric="AUC", stopping_tolerance=0.01, stopping_rounds=10, cv=3):
    trainData, validData = train_test_split(data, 
                                            test_size=valid_prop, 
                                            stratify=data[target], 
                                            random_state=1)
    
    trainLabel = trainData[target]
    trainData.drop(target, axis = 1, inplace = True)
    
    validLabel = validData[target]
    validData.drop(target, axis = 1, inplace = True)
    
    # for specifying index of categorical columns
    categorical_features_indices = np.where((trainData.dtypes == "category") | (trainData.dtypes == "object"))[0]
    
    # check where Regression or not
    if isReg == True:
        mdl = cbt.CatBoostRegressor(use_best_model=True,
                                    eval_metric=eval_metric,
                                    od_type="Iter",
                                    od_pval=stopping_tolerance,
                                    od_wait=stopping_rounds,
                                    random_seed=1234)
    else:
        mdl = cbt.CatBoostClassifier(use_best_model=True,
                                     eval_metric=eval_metric,
                                     od_type="Iter",
                                     od_pval=stopping_tolerance,
                                     od_wait=stopping_rounds,
                                     random_seed=1234)

    # Create the grid
    grid = GridSearchCV(mdl, grid_params, verbose=2, cv=cv, n_jobs=4)
    
    # Run the grid
    grid.fit(X=trainData, 
             y=trainLabel, 
             cat_features=categorical_features_indices,
             eval_set=(validData, validLabel),
             verbose=False,
             plot=False)
    
    params['depth'] = grid.best_params_['depth']
    params['rsm'] = grid.best_params_['rsm']
    params['l2_leaf_reg'] = grid.best_params_['l2_leaf_reg']
    params['learning_rate'] = grid.best_params_['learning_rate']
    
    return params


""" 
##### example #####
drop_column = ['Ticket', 'Cabin', 'Name']

train.drop(drop_column, axis = 1, inplace = True)
test.drop(drop_column, axis = 1, inplace = True)

gridParams = {'depth'            : [4, 5, 6],
              'rsm'              : [0.8, 0.9, 1],                    # col sample by level
              'l2_leaf_reg'      : [0, 0.1, 0.01, 0.001, 0.0001],    # lambda in Ridge
              'learning_rate'    : [0.001, 0.01, 0.1]}

bestParams = tuneCatBoost(data = train, 
                          target = "Survived", 
                          grid_params = gridParams, 
                          valid_prop = 0.4,
                          eval_metric = "Accuracy",
                          stopping_tolerance = 0.01,
                          stopping_rounds = 10,
                          cv = 3,
                          isReg = False)

"""