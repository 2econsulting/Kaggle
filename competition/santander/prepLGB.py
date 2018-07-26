# author: Heo Sung Wook

import lightgbm as lgb

def prepLGB(data, target):
    labels = data[target]
    
    # Create Light GBM data set        
    lData = lgb.Dataset(data = data.drop(target, axis = 1), 
                        label = labels,
                        feature_name = list(data.columns.drop(target)),
                        categorical_feature = 'auto')
    
    return lData, labels

""" 
##### example #####
drop_column = ["Cabin", "Name", "Ticket"]

train.drop(drop_column, axis = 1, inplace = True)
test.drop(drop_column, axis = 1, inplace = True)

trainDataL, labels = prepLGB(data = train, 
                             target = "Survived")

"""