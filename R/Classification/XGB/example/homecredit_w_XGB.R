# title : homecredit_w_XGB 
# authro : jacob 

# library 
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(parallel)
library(data.table)
library(e1071)
library(caret)
library(Metrics)
library(xgboost)

# tuning code
path_code = "~/GitHub/2econsulting/Kaggle/R/Classification/XGB"
source(file.path(path_code,"tuneXGB.R"))
source(file.path(path_code,"cvpredictXGB.R"))

# set input files 
path_input = "~/Kaggle/homecredit/input"
file_data = 'will/will_train.csv'
file_test = 'will/will_test.csv'
file_submit  = 'will/sample_submission.csv'

# set output files
path_output = "~/Kaggle/homecredit/output" 
file_ztable = "ztableXGB_w_will.csv"
file_pred = "pred_w_xgb.csv"

# .. 
y = "TARGET"
ml = "xgb"
max_model = 2
sample_rate = 0.001
kfolds = 2

# read data
data = fread(file.path(path_input, file_data))
test = fread(file.path(path_input, file_test))
submit = fread(file.path(path_input, file_submit))

# sampling
set.seed(1)
data <- data[sample(nrow(data)),]
sample_num =round(nrow(data)*sample_rate)

# ..
data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL

# ..
# data[is.na(data)] <- -9999
# test[is.na(test)] <- -9999

# ------------------------
#  optimal Depth Range
# ------------------------
params <- expand.grid(
  max_depth =  c(2, 3, 4, 5, 6, 7, 8, 9),
  learning_rate = 0.1
)
optimalDepthRange <- tuneXGB(head(data, sample_num), y=y, params=params, k=kfolds, max_model=nrow(params))

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
optimalParams <- tuneXGB(head(data, sample_num), y=y, params=params, k=kfolds, max_model=max_model)

# ------------------------
# cvpredict catboost 
# ------------------------
params = as.list(head(optimalParams$scores[names(params)],1))
output <- cvpredictLGB(data, test, k=kfolds*2, y=y, params=params)
output$crossvalidation_score
output$cvpredict_score

# ztable and submit
fwrite(data.frame(ztable=output$ztable), file.path(path_output, file_ztable))
submit[,y] <- ifelse(output$pred>1, 1, output$pred)
fwrite(submit, file.path(path_output, file_pred))

# save params
BestParams = paste0(ml,output$cvpredict_score,"_params.Rda")
saveRDS(optimalParams$scores, file.path(path_output, BestParams))


