# title : homecredit_w_lightgbm
# author : jacob

# library 
setwd("~/GitHub/2econsulting/Kaggle/R/Classification/LGB/example")
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(rAutoFE)
library(data.table)
library(e1071)
library(caret)
library(Metrics)
library(lightgbm)
source("../tuneLGB.R")
source("../cvpredictLGB.R")

# read data
data = fread('./input/homecredit_data.csv')
test = fread('./input/homecredit_test.csv')
sample = fread('./input/sample_submission.csv')

# ..
data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL
names <- which(sapply(data, class) != "numeric")
data[, (names) := lapply(.SD, as.numeric), .SDcols = names]

# ..
data[is.na(data)]<- 0
test[is.na(test)]<- 0


# ------------------------
#  optimal Depth Range
# ------------------------
params <- expand.grid(
  max_depth = c(2,3,4)
)
optimalDepthRange <- tuneLGB(data, y="TARGET", params=params, cv=5, max_model=nrow(params))
optimalDepthRange$scores


# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  max_depth = head(optimalDepthRange$scores$max_depth, 3),
  learning_rate = seq(0.01, 1, 0.3),
  subsample = seq(0.6, 1, 0.9),
  colsample_bytree = seq(0.5, 1, 0.1), 
  min_child_weight = seq(1, 40, 1),
  max_delta_step = seq(1, 10, 1)
)
optimalParams <- tuneLGB(data, y="TARGET", params=params, cv=5, max_model=10)
optimalParams$scores


# ------------------------
# cvpredict catboost 
# ------------------------
params = as.list(head(optimalParams$scores[names(params)],1))
output <- cvpredictLGB(data, test, k=10, y="TARGET", params=params)
output$crossvalidation_score
output$cvpredict_score


