# author: Heo Sung Wook

import pandas as pd
import xgboost as xgb

def prepXGB(data, target="", train = True):
    if train == True:
         # Transformation with test data set
        labels = data[target]
        
        data_with_dummy = pd.get_dummies(data)
        
        # use DMatrix for xgbosot
        XGB_data = xgb.DMatrix(data_with_dummy.drop(target, axis = 1), label=labels)
        return XGB_data, labels, data_with_dummy
    
    else:
        # Transformation with test data set
        if target in data.columns:
            data.drop(target, axis = 1, inplace = True)
            
        data_with_dummy = pd.get_dummies(data)
        XGB_data = xgb.DMatrix(data_with_dummy)
        return XGB_data


""" 
##### example #####
drop_column = ['Ticket', 'Cabin', 'Name']

train.drop(drop_column, axis = 1, inplace = True)
test.drop(drop_column, axis = 1, inplace = True)

trainXGB, trainLabel, trainDummy = prepXGB(data = train,
                                           target = "Survived")

testXGB = prepXGB(data = test,
                  target = "Survived",
                  train = False)
"""