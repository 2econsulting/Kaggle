# author: Heo Sung Wook

import xgboost as xgb
import prepXGB # call preprocessing function

from sklearn.model_selection import train_test_split

prepXGB = prepXGB.prepXGB

def xgb_cv_predict(data, new_data, params, num_boost_round, num_iter, valid_prop, target, early_stopping_rounds=10):
    predsTest = 0
    for i in range(0, num_iter): 
        # Prepare the data set for fold
        trainData, validData = train_test_split(data, 
                                                test_size=valid_prop, 
                                                stratify=data[target], 
                                                random_state = (i+1))
        
        # preprocessing to run XGBoost
        trainXGB, _, _= prepXGB(data = trainData, 
                                target = target)
        
        validXGB, _, _= prepXGB(data = validData, 
                                target = target)
        
        # different from Light GBM, test data set also need to be transformed with XGB DMatrix
        testXGB = prepXGB(data = new_data,
                          target = target,
                          train = False)
        
        # Train     
        ml_xgb = xgb.train(params = params, 
                           dtrain = trainXGB, 
                           num_boost_round = num_boost_round, 
                           evals = [(validXGB, 'valid')],
                           early_stopping_rounds = early_stopping_rounds,
                           verbose_eval = 4)
    
        # Predict
        predsTest += ml_xgb.predict(testXGB, ntree_limit=ml_xgb.best_ntree_limit)/num_iter

    return ml_xgb, predsTest


"""
##### example #####
## after run "prepXGB.py" and "tuneXGB.py" example code 

best_params["eval_metric"] = "logloss"   ## for binary classification
best_params['seed'] = 1234               ## set seed

xgb_model, pred = xgb_cv_predict(data = train, 
                                 new_data = test,
                                 params = best_params,  
                                 num_boost_round = 10000, 
                                 num_iter = 12,
                                 valid_prop = 0.4, 
                                 target = "Survived",
                                 early_stopping_rounds = 50)
"""

