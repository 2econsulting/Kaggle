# title : homecredit_w_lightgbm
# author : jacob
# params : https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

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
data = fread('~/Kaggle/homecredit/input/will/will_train.csv')
test = fread('~/Kaggle/homecredit/input/will/will_test.csv')
sample = fread('~/Kaggle/homecredit/input/will/sample_submission.csv')

# ..
data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL
names <- which(sapply(data, class) != "numeric")
data[, (names) := lapply(.SD, as.numeric), .SDcols = names]

# ..
print(sum(is.na(data)))
data[is.na(data)] <- 0
test[is.na(test)] <- 0


# ------------------------
#  optimal Depth Range
# ------------------------
params <- expand.grid(
  max_depth = c(-1, 2, 3, 4, 5, 6, 7, 8, 9)
)
optimalDepthRange <- tuneLGB(data, y="TARGET", params=params, cv=3, max_model=nrow(params))
optimalDepthRange$scores


# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  max_depth = head(optimalDepthRange$scores$max_depth, 3),
  learning_rate = c(0.1, 0.05, 0.01),
  subsample = c(1, 0.9, 0.8, 0.7, 0.6),
  colsample_bytree = c(1, 0.9, 0.8, 0.7, 0.6), 
  min_child_samples = c(20, 1, 2, 3, 5, 10, 15, 40)
)
optimalParams <- tuneLGB(data, y="TARGET", params=params, cv=5, max_model=3)
optimalParams$scores


# ------------------------
# cvpredict catboost 
# ------------------------
params = as.list(head(optimalParams$scores[names(params)],1))
output <- cvpredictLGB(data, test, k=5, y="TARGET", params=params)
output$crossvalidation_score
output$cvpredict_score

# ztable and submit
fwrite(data.frame(ztable=output$ztable), '~/Kaggle/homecredit/ztable/ztableLGB_w_will.csv')
sample$TARGET <- output$pred
sample$TARGET[which(sample$TARGET>1)] <- 1
fwrite(sample, paste0("~/Kaggle/homecredit/output/sub_w_lgb",sample(100,1),".csv"))

