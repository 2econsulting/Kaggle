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
data = fread('~/Kaggle/homecredit/input/homecredit_data.csv')
test = fread('~/Kaggle/homecredit/input/homecredit_test.csv')
sample = fread('~/Kaggle/homecredit/input/sample_submission.csv')

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
  max_depth =  c(2, 3, 4, 5, 6, 7, 8, 9)
)
optimalDepthRange <- tuneXGB(data, y="TARGET", params=params, cv=5, max_model=nrow(params))
optimalDepthRange$scores
xgb.train()

# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  max_depth = head(optimalDepthRange$scores$max_depth, 3),
  eta = c(0.3, 0.1, 0.05, 0.01),
  subsample = c(1, 0.9, 0.8, 0.7, 0.6),
  colsample_bytree = c(1, 0.9, 0.8, 0.7, 0.6),
  min_child_weight = c(20, 1, 2, 3, 5, 10, 15, 40),
  lambda  = c(1, 2, 3)
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

# ztable and submit
fwrite(data.frame(ztable=output$ztable), '~/Kaggle/homecredit/ztable/ztableXGB_w_will.csv')
sample$TARGET <- output$pred
sample$TARGET[which(sample$TARGET>1)] <- 1
fwrite(sample, paste0("~/Kaggle/homecredit/output/sub_w_xgb",sample(100,1),".csv"))

