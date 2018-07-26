# author: Heo Sung Wook

import catboost as cbt
import numpy as np
from sklearn.model_selection import train_test_split

def cbt_cv_predict(data, new_data, params, iterations, cv, valid_prop, target, isReg, eval_metric="AUC", stopping_tolerance=0.01, stopping_rounds=10):
    # check where Regression or not
    if isReg == True:
        mdl = cbt.CatBoostRegressor(use_best_model=True,
                                    iterations=iterations,
                                    eval_metric=eval_metric,
                                    od_type="Iter",
                                    od_pval=stopping_tolerance,
                                    od_wait=stopping_rounds,
                                    random_seed=1234)
    else:
        mdl = cbt.CatBoostClassifier(use_best_model=True,
                                     iterations=iterations,
                                     eval_metric=eval_metric,
                                     od_type="Iter",
                                     od_pval=stopping_tolerance,
                                     od_wait=stopping_rounds,
                                     random_seed=1234)
    
    predsTest = 0
    for i in range(0, cv): 
        # Prepare the data set for fold
        trainData, validData = train_test_split(data, 
                                                test_size=valid_prop, 
                                                stratify=data[target], 
                                                random_state = (i+1))        
        
        
        trainLabel = trainData[target]
        trainData.drop(target, axis = 1, inplace = True)
        
        validLabel = validData[target]
        validData.drop(target, axis = 1, inplace = True)
        
        # for specifying index of categorical columns
        categorical_features_indices = np.where((trainData.dtypes == "category") | (trainData.dtypes == "object"))[0]
        
        
        ml_cbt = mdl.fit(X=trainData, 
                         y=trainLabel, 
                         cat_features=categorical_features_indices,
                         eval_set=(validData, validLabel),
                         verbose=False,
                         plot=False)
        
        # Predict
        predsTest += ml_cbt.predict(new_data)/cv

    return ml_cbt, predsTest


"""
##### example #####
## after run "tuneCBT.py" example code 


cbt_model, pred = cbt_cv_predict(data = train, 
                                 new_data = test,
                                 params = bestParams,  
                                 iterations = 1000, 
                                 cv = 12,
                                 isReg = False,
                                 valid_prop = 0.4, 
                                 target = "Survived",
                                 stopping_rounds = 50)
"""