# title : homecredit_w_XGB 
# authro : jacob 

# library 
setwd("~/GitHub/2econsulting/Kaggle/R/Classification/XGB/example/")
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(rAutoFE)
library(data.table)
library(e1071)
library(caret)
library(Metrics)
library(xgboost)
source('../tuneXGB.R')
source('../cvpredictXGB.R')

# read data
data = fread('./input/homecredit_data.csv')
test = fread('./input/homecredit_test.csv')
sample = fread('./input/sample_submission.csv')

# ..
data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL

# ..
data[is.na(data)]<- 0
test[is.na(test)]<- 0


# ------------------------
#  optimal Depth Range
# ------------------------
params <- expand.grid(
  max_depth = c(2,3,4)
)
optimalDepthRange <- tuneXGB(data, y="TARGET", params=params, cv=5, max_model=nrow(params))
optimalDepthRange$scores


# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  max_depth = head(optimalDepthRange$scores$max_depth, 3),
  eta = seq(0.01, 1, 0.3),
  gamma = seq(0, 1, 0.2), 
  subsample = seq(0.6, 1, 0.9),
  colsample_bytree = seq(0.5, 1, 0.1), 
  min_child_weight = seq(1, 40, 1),
  max_delta_step = seq(1, 10, 1)
)
optimalParams <- tuneXGB(data, y="TARGET", params=params, cv=5, max_model=3)
optimalParams$scores


# ------------------------
# cvpredict catboost 
# ------------------------
source('../cvpredictXGB.R')
params = as.list(head(optimalParams$scores[names(params)],1))
output <- cvpredictXGB(data, test, k=5, y="TARGET", params=params)
output$crossvalidation_score
output$cvpredict_score


