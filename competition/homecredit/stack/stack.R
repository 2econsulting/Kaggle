# title : home credit 
# author : jacob 

# library 
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(data.table)
library(e1071)
library(caret)
library(Metrics)
require(Matrix)
require(lightgbm)  
library(xgboost)
library(catboost)
library(rBayesianOptimization)

# path 
path_code   = "~/GitHub/2econsulting/Kaggle/competition/homecredit/base"
path_output = "~/GitHub/2econsulting/Kaggle_data/homecredit/output" 
path_input  = "~/GitHub/2econsulting/Kaggle_data/homecredit/input"

# train options 
y = "TARGET"
sample_rate = 1
kfolds = 5
early_stopping_rounds = 100
iterations = 10000
num_threads = 8
learning_rate = 0.02

# bayesian search options
init_points = 100      
n_iter = 100  

# bayesian search 
table_nm = "kageyama"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))
table_nm = "bojan"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))
table_nm = "ivan"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))
table_nm = "will"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))
table_nm = "pooh"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))
table_nm = "olivier"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))
table_nm = "aguiar"
source(file.path(path_code,"homecredit_w_LGB_bayes.R"))

# random search options 
max_model = 200

# two step random search 
table_nm = "kageyama"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "bojan"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "ivan"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "will"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "pooh"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "olivier"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "aguiar"
source(file.path(path_code,"homecredit_w_LGB.R"))


