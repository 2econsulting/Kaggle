# author: Heo Sung Wook

import pandas as pd
import xgboost as xgb

def prepXGB(data, target, categorical_column_name):
    labels = data[target]
    
    data_with_dummy = pd.get_dummies(data)
    
    # use DMatrix for xgbosot
    XGB_data = xgb.DMatrix(data_with_dummy, label=labels)
    
    return XGB_data, labels, data_with_dummy



""" 
****** example ******
categorical_column_name = ["Embarked", "Sex", "CL", "CN", "Surname", "Title"]
trainXGB, trainLabel, trainDummy = prepXGB(data = train, 
                                           target = "Survived", 
                                           categorical_column_name = categorical_column_name)
"""