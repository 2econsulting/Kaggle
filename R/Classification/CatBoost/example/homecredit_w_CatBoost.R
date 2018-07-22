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

# read data
data = fread('./input/homecredit_data.csv')
test = fread('./input/homecredit_test.csv')
sample = fread('./input/sample_submission.csv')

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
source("../tuneCatBoost.R")
params <- expand.grid(
  depth = c(3,4,5,6,7),
  learning_rate = 0.05, # 0.05
  iterations = 10,
  border_count = 32,
  rsm = 1,
  l2_leaf_reg = 3
)
optimalDepthRange <- tuneCatBoost(data, y="TARGET", max_model=nrow(params), cv=3, grid=params)
optimalDepthRange$results


# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  depth = head(optimalDepthRange$results$depth,3),
  learning_rate = c(0.1, 0.03),
  l2_leaf_reg = c(0, 3 ,1, 2),
  rsm = c(1,0.9,0.8),
  border_count = 32,
  iterations = 10
)
optimalParams <- tuneCatBoost(data, y="TARGET", max_model=5, cv=3, grid=params)
optimalParams$results


# ------------------------
# cvpredict catboost 
# ------------------------
source("../cvpredictCatBoost.R")
params <- as.list(optimalParams$bestTune)
output = cvpredictCatBoost(data, test, k=3, y="TARGET", params=params)
output$cvpredict_score
output$crossvalidation_score


