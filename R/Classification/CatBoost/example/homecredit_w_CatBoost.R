# title : homecredit_w_CatBoost
# author : jacob

# library 
setwd("~/GitHub/2econsulting/Kaggle/R/Classification/CatBoost/example/")
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(caret)
library(rAutoFE)
library(data.table)
library(e1071)
library(Metrics)
library(catboost)
source("../tuneCatBoost.R")
source("../cvpredictCatBoost.R")

# path 
path_input = "~/Kaggle/homecredit/input/"
path_output = "~/Kaggle/homecredit/output/" 
path_ztable = "~/Kaggle/homecredit/ztable/" 

# output file
file_ztable = "ztableCAT_w_will.csv"
file_sub = "sub_w_cat.csv"

# set y 
y = "TARGET"

# read data
data = fread(file.path(path_input,'will/will_train.csv'))
test = fread(file.path(path_input,'will/will_test.csv'))
sub = fread(file.path(path_input,'will/sample_submission.csv'))

# sampling
# data <- head(data, round(nrow(data)*0.01))
# test <- head(test, round(nrow(test)*0.01))
# sub <- head(sub, round(nrow(sub)*0.01))

# ..
data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL
names <- which(sapply(data, class) != "numeric")
data[, (names) := lapply(.SD, as.numeric), .SDcols = names]

# missing value
data[is.na(data)] <- 0
test[is.na(test)] <- 0

# ------------------------
#  optimal Depth Range
# ------------------------
params <- expand.grid(
  depth = c(2, 3, 4, 5, 6, 7, 8, 9),
  learning_rate = 0.03, 
  iterations = 1000,
  border_count = 128,
  rsm = 1,
  l2_leaf_reg = 3
)
optimalDepthRange <- tuneCatBoost(data, y=y, max_model=nrow(params), cv=5, grid=params)
optimalDepthRange$results

# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  depth = head(optimalDepthRange$results$depth,3),
  learning_rate = c(0.3, 0.1, 0.05, 0.01),
  l2_leaf_reg = c(3 ,1, 2 ,6),
  rsm = c(1, 0.9, 0.8, 0.7, 0.6),
  border_count = c(32, 64, 128),
  iterations = 1000
)
optimalParams <- tuneCatBoost(data, y=y, grid=params, cv=5, max_model=100)
optimalParams$results

# ------------------------
# cvpredict catboost 
# ------------------------
params <- as.list(optimalParams$bestTune)
output = cvpredictCatBoost(data, test, k=10, y=y, params=params)
output$cvpredict_score
output$crossvalidation_score

# ztable and submit
fwrite(data.frame(ztable=output$ztable), paste0(path_ztable,file_ztable))
sub[,y] <- ifelse(output$pred>1,1,output$pred)
fwrite(sub, paste0(path_output,file_sub))

