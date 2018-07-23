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

# path 
path_input = "~/Kaggle/homecredit/input/"
path_output = "~/Kaggle/homecredit/output/" 
path_ztable = "~/Kaggle/homecredit/ztable/" 

# output file
file_ztable = "ztableLGB_w_will.csv"
file_sub = "sub_w_lgb.csv"

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
optimalDepthRange <- tuneLGB(data, y=y, params=params, cv=5, max_model=nrow(params))
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
optimalParams <- tuneLGB(data, y=y, params=params, cv=5, max_model=100)
optimalParams$scores

# ------------------------
# cvpredict catboost 
# ------------------------
params = as.list(head(optimalParams$scores[names(params)],1))
output <- cvpredictLGB(data, test, k=10, y=y, params=params)
output$crossvalidation_score
output$cvpredict_score

# ztable and submit
fwrite(data.frame(ztable=output$ztable), paste0(path_ztable,file_ztable))
sub[,y] <- ifelse(output$pred>1,1,output$pred)
fwrite(sub, paste0(path_output,file_sub))

