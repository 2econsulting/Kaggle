# title : santander
# author : jacob

# library 
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(rAutoFE)
library(data.table)
library(caret)
library(e1071)
library(Metrics)
library(catboost)
source("./make_submit.R")
source("./TheFeatures.R")
source("./add_statistics.R")
source("./ml/CatBoost/trainCatBoost.R")
source("./ml/CatBoost/catboost_cv_predict.R")
source("./ml/CatBoost/catboost_cv_performance.R")
source("./ml/CatBoost/tuneCatBoost.R")  
source("./ml/CatBoost/prepCatBoost.R")

# read data
data <- fread("./input/train.csv")
data[, ID:=NULL]
test <- fread("./input/test.csv")
sample <- fread("./input/sample_submission.csv")

# feature engineering 
source("./fe_santander.R")

# set x and y 
y = "target"
x = colnames(data)[colnames(data)!=y]

# catboost
# set.seed(1234)
# ml_tunecat = tuneCatBoost(as.data.frame(data), y=y, max_model=100, cv=5, gridtype="big-size")

# save and load
# catboost.save_model(ml_tunecat$finalModel, model_path = "./model/ml_tunecat")
ml <- catboost.load_model(model_path = "./model/ml_tunecat")

# predict & submit
test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
pred <- catboost.predict(ml, test_pool)
pred <- exp(pred)-1
make_submit(pred, name = "tmp")

# catboost_cv_performance
params <- list(
  depth = 6,
  learning_rate = 0.01,
  iterations = 500,
  l2_leaf_reg = 3,
  rsm = 1,
  border_count = 34,
  use_best_model = T,
  loss_function = 'RMSE',
  eval_metric = "RMSE"
)

cvEval <- catboost_cv_performance(
  data = as.data.frame(data),
  newdata = as.data.frame(test),
  k = 5,
  y = y,
  params = params,
  train_dir = "./model/catboost"
)
cvEval$eval
mean(cvEval$eval$eval)

# submit
pred = rowMeans(cvEval$pred)
make_submit(pred = pred, name = "sample10_cvpred")










