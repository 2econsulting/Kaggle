# author: Heo Sung Wook

import lightgbm as lgb
import prepLGB # call preprocessing function

from sklearn.model_selection import train_test_split

prepLGB = prepLGB.prepLGB

def lgb_cv_predict(data, new_data, params, num_boost_round, num_iter, valid_size, target):
    predsTest = 0
    for i in range(0, num_iter): 
        # Prepare the data set for fold
        trainData, validData = train_test_split(data, 
                                                test_size=valid_size, 
                                                stratify=data[target], 
                                                random_state = (i+1))
        
        # preprocessing for run Light GBM module
        trainDataL, _= prepLGB(data = trainData, 
                               target = target)
        
        validDataL, _= prepLGB(data = validData, 
                               target = target)
        
        # Train     
        lgbm = lgb.train(params = params, 
                         train_set = trainDataL, 
                         num_boost_round = num_boost_round, 
                         valid_sets = [validDataL],
                         early_stopping_rounds = 50,
                         verbose_eval = 4)
    
        # Predict
        predsTest  += lgbm.predict(new_data, num_iteration=lgbm.best_iteration)/num_iter

    return lgbm, predsTest


"""
****** example ******
****** after run "tuneLGB.py" example code 
drop_column = ["Cabin", "Name", "Ticket"]
params['metric'] = 'binary'

lgb_model, pred = lgb_cv_predict(
                                  data = train.drop(drop_column, axis = 1), 
                                  new_data = test.drop(drop_column, axis = 1),
                                  params = params,  
                                  num_boost_round = 10000, 
                                  num_iter = 12,
                                  valid_size = 0.4, 
                                  target = "Survived"
                                )
"""