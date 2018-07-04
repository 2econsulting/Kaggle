# author: Heo Sung Wook

import xgboost as xgb
import prepXGB # call preprocessing function

from sklearn.model_selection import train_test_split

prepXGB = prepXGB.prepXGB

def xgb_cv_predict(data, new_data, params, num_boost_round, num_iter, valid_size, target, categorical_column_name):
    predsTest = 0
    for i in range(0, num_iter): 
        # Prepare the data set for fold
        trainData, validData = train_test_split(data, 
                                                test_size=valid_size, 
                                                stratify=data[target], 
                                                random_state = (i+1))
        
        # preprocessing for run XGBoost module
        trainXGB, _, _= prepXGB(data = trainData, 
                                target = target,
                                categorical_column_name = categorical_column_name)
        
        validXGB, _, _= prepXGB(data = validData, 
                                target = target,
                                categorical_column_name = categorical_column_name)
        
        # Train     
        ml_xgb = xgb.train(params = params, 
                           dtrain = trainXGB, 
                           num_boost_round = num_boost_round, 
                           evals = [(validXGB, 'valid')],
                           early_stopping_rounds = 50,
                           verbose_eval = 4)
    
        # Predict
        predsTest  += ml_xgb.predict(new_data, ntree_limit=ml_xgb.best_ntree_limit)/num_iter

    return ml_xgb, predsTest


"""
****** example ******
****** after run "tuneXGB.py" example code 
categorical_column_name = ["Embarked", "Sex", "CL", "CN", "Surname", "Title"]
best_params["eval_metric"] = "logloss"

lgb_model, pred = xgb_cv_predict(
                                  data = train, 
                                  new_data = test,
                                  params = best_params,  
                                  num_boost_round = 100, 
                                  num_iter = 12,
                                  valid_size = 0.4, 
                                  target = "Survived",
                                  categorical_column_name = categorical_column_name
                                )
"""