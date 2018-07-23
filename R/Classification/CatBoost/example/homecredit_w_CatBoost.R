# title : homecredit_w_CatBoost
# author : jacob
# parms : https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-train-docpage/

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
data = fread('~/Kaggle/homecredit/input/homecredit_data.csv')
test = fread('~/Kaggle/homecredit/input/homecredit_test.csv')
sample = fread('~/Kaggle/homecredit/input/sample_submission.csv')

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
  depth = c(2, 3, 4, 5, 6, 7, 8, 9),
  learning_rate = 0.03, 
  iterations = 1000,
  border_count = 128,
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
  learning_rate = c(0.3, 0.1, 0.05, 0.01),
  l2_leaf_reg = c(3 ,1, 2 ,6),
  rsm = c(1, 0.9, 0.8, 0.7, 0.6),
  border_count = c(32, 64, 128),
  iterations = 1000
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

# ztable and submit
fwrite(data.frame(ztable=output$ztable), '~/Kaggle/homecredit/ztable/ztableCAT_w_will.csv')
sample$TARGET <- output$pred
sample$TARGET[which(sample$TARGET>1)] <- 1
fwrite(sample, paste0("~/Kaggle/homecredit/output/sub_w_cat",sample(100,1),".csv"))



