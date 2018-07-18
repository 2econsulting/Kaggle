# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 00:54:49 2018

@author: jacob
"""

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np 
np.random.seed(1)

def get_selected_features():
    return [
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]

def get_selected_features2():
    tmp = pd.read_csv("./tmp9090.csv").x.tolist()
    return tmp

def get_data():
    print('Reading data')
    data = pd.read_csv('./input/train.csv', nrows=None)
    test = pd.read_csv('./input/test.csv', nrows=None)
    print('Train shape ', data.shape, ' Test shape ', test.shape)
    return data, test

def add_statistics(data, test):
    # This is part of the trick I think, plus lightgbm has a special process for NaNs
    data.replace(0, np.nan, inplace=True)
    test.replace(0, np.nan, inplace=True)
    
    for df in [data, test]:
        df['nb_nans'] = df.isnull().sum(axis=1)
        # All of the stats will be computed without the 0s 
        df['the_median'] = df.median(axis=1)
        df['the_mean'] = df.mean(axis=1)
        df['the_sum'] = df.sum(axis=1)
        df['the_std'] = df.std(axis=1)
        df['the_kur'] = df.kurtosis(axis=1)
        
    return data, test

