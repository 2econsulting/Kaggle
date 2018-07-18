# author: Heo Sung Wook

import numpy as np
from sklearn.metrics import mean_squared_error 
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def RandomTuneLGB(data, target, grid_params, objective, test, valid_prop):
 
    n_jobs = 8
    
    # split dataset
    train, valid= train_test_split(data,test_size=valid_prop,random_state=1)
    train_y = train[target].copy()
    train_x = train.drop(target, axis = 1)    
    valid_y = valid[target].copy()
    valid_x = valid.drop(target, axis = 1) 
    
    mdl = lgb.LGBMRegressor(
            boosting_type= 'gbdt', 
            objective = objective,
            n_jobs = n_jobs,
            silent = False,
            random_state = 1,
            n_estimators = 1000
            )
    
    # Create the grid
    grid = RandomizedSearchCV(mdl, grid_params, n_iter=100, verbose=3, cv=10, n_jobs=n_jobs)
    
    # self-defined eval metric
    # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    # Root Mean Squared Logarithmic Error (RMSLE)
    def rmsle(y_true, y_pred):
        return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

    # Run the grid
    grid.fit(
        train_x, train_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric="l2",
        early_stopping_rounds=10
        ) 

    # ----------
    # cv_predict
    # ----------
    data_y = data[target].copy()
    data_x = data.drop(target, axis = 1)   
    ddata = lgb.Dataset(data=data_x, label=np.log1p(data_y), free_raw_data=False)

    # Construct dataset so that we can use slice()
    ddata.construct()         
    
    # Create folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    
    # Init predictions
    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])
    
    dict_ = {
            'objective':'regression',
            'boosting_type':'gbdt',
            'metric':'l2',
            'seed':3,
            'verbose':-1            
            }
    
    params = grid.best_params_
    params.update(dict_)
    print(params)

    # Run KFold
    for trn_idx, val_idx in folds.split(data):

        # Train lightgbm
        clf = lgb.train(
            params=params,
            train_set=ddata.subset(trn_idx),
            valid_sets=ddata.subset(val_idx),
            num_boost_round=10000, 
            early_stopping_rounds=100,
            verbose_eval=50
        )            
        
        # Predict Out Of Fold and Test targets
        oof_preds[val_idx] = clf.predict(ddata.data.iloc[val_idx])
        sub_preds += clf.predict(test) / folds.n_splits
        print(mean_squared_error(np.log1p(data_y.iloc[val_idx]),oof_preds[val_idx]) ** .5)
        
    # Display Full OOF score (square root of a sum is not the sum of square roots)
    print('>> RMSLE : %9.6f' % (mean_squared_error(np.log1p(data_y), oof_preds) ** .5))     

    return grid, sub_preds



def CartesianTuneLGB(data, target, grid_params, objective, test, valid_prop):    
    
    n_jobs = 8
    
    # split dataset
    train, valid = train_test_split(data,test_size=valid_prop,random_state=1)
    train_y = train[target].copy()
    train_x = train.drop(target, axis = 1)    
    valid_y = valid[target].copy()
    valid_x = valid.drop(target, axis = 1) 
    
    mdl = lgb.LGBMRegressor(
            boosting_type= 'gbdt', 
            objective = objective,
            n_jobs = n_jobs,
            silent = False,
            random_state = 1,
            n_estimators = 1000
            )
    
    # Create the grid
    grid = GridSearchCV(mdl, grid_params, verbose=3, cv=10, n_jobs=n_jobs)
    
    # self-defined eval metric
    # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    # Root Mean Squared Logarithmic Error (RMSLE)
    def rmsle(y_true, y_pred):
        return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

    # Run the grid
    grid.fit(
        train_x, train_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric="l2",
        early_stopping_rounds=10
        ) 

    # ----------
    # cv_predict
    # ----------
    data_y = data[target].copy()
    data_x = data.drop(target, axis = 1)   
    ddata = lgb.Dataset(data=data_x, label=np.log1p(data_y), free_raw_data=False)

    # Construct dataset so that we can use slice()
    ddata.construct()         
    
    # Create folds
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    
    # Init predictions
    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])
    
    dict_ = {
            'objective':'regression',
            'boosting_type':'gbdt',
            'metric':'l2',
            'seed':3,
            'verbose':-1            
            }
    
    params = grid.best_params_
    params.update(dict_)
    print(params)

    # Run KFold
    for trn_idx, val_idx in folds.split(data):

        # Train lightgbm
        clf = lgb.train(
            params=params,
            train_set=ddata.subset(trn_idx),
            valid_sets=ddata.subset(val_idx),
            num_boost_round=10000, 
            early_stopping_rounds=100,
            verbose_eval=50
        )            
        
        # Predict Out Of Fold and Test targets
        oof_preds[val_idx] = clf.predict(ddata.data.iloc[val_idx])
        sub_preds += clf.predict(test) / folds.n_splits
        print(mean_squared_error(np.log1p(data_y.iloc[val_idx]),oof_preds[val_idx]) ** .5)
        
    # Display Full OOF score (square root of a sum is not the sum of square roots)
    print('>> RMSLE : %9.6f' % (mean_squared_error(np.log1p(data_y), oof_preds) ** .5))     

    return grid, sub_preds