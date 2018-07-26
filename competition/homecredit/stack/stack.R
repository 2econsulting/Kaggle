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
library(lightgbm)

# path 
path_code   = "~/GitHub/2econsulting/Kaggle/competition/homecredit/base"
path_output = "~/GitHub/2econsulting/Kaggle_data/homecredit/output" 
path_input  = "~/GitHub/2econsulting/Kaggle_data/homecredit/input"

# train setting 
# y = "TARGET"
# max_model = 500
# sample_rate = 1
# kfolds = 5
# early_stopping_rounds = 50
# iterations = 10000
# num_threads = 8
# learning_rate = 0.05
y = "TARGET"
max_model = 3
sample_rate = 0.001
kfolds = 3
early_stopping_rounds = 5
iterations = 100
num_threads = 8
learning_rate = 0.05

# base learners
table_nm = "will"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "pooh"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "olivier"
source(file.path(path_code,"homecredit_w_LGB.R"))
table_nm = "aguiar"
source(file.path(path_code,"homecredit_w_LGB.R"))


