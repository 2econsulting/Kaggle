# author: Heo Sung Wook

import lightgbm as lgb
import prepLGB # call preprocessing function

from sklearn.model_selection import train_test_split

prepLGB = prepLGB.prepLGB

def lgb_cv_predict(data, new_data, params, num_boost_round, num_iter, valid_prop, target, early_stopping_rounds = 10):
    predsTest = 0
    for i in range(0, num_iter): 
        # Prepare the data set for fold
        trainData, validData = train_test_split(data, 
                                                test_size=valid_prop, 
                                                stratify=data[target], 
                                                random_state = (i+1))
        
        # preprocessing for run Light GBM module
        trainDataL, _= prepLGB(data = trainData, 
                               target = target)
        
        validDataL, _= prepLGB(data = validData, 
                               target = target)
        
        # Train     
        ml_lgb = lgb.train(params = params, 
                           train_set = trainDataL, 
                           num_boost_round = num_boost_round, 
                           valid_sets = [validDataL],
                           early_stopping_rounds = early_stopping_rounds,
                           verbose_eval = 4)
    
        # Predict
        predsTest  += ml_lgb.predict(new_data, num_iteration=ml_lgb.best_iteration)/num_iter

    return ml_lgb, predsTest


"""
##### example #####
## after run "prepLGB.py" and "tuneLGB.py" example code 

best_params['metric'] = 'binary' ## for binary classification
best_params['seed'] = 1234       ## set seed

lgb_model, pred = lgb_cv_predict(data = train, 
                                 new_data = test,
                                 params = best_params,  
                                 num_boost_round = 10000, 
                                 num_iter = 12,
                                 valid_prop = 0.4, 
                                 target = "Survived",
                                 early_stopping_rounds = 50)
"""