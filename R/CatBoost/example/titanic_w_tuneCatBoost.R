# title : titanic 
# authro : jacob 

# library 
library(rAutoFE)
library(h2o)
library(caret)
library(catboost)
library(lightgbm)
library(xgboost)
library(Matrix)
library(rjson)
source('fe_titanic.R')
source('make_submit.R')
source("tuneCatBoost.R")
source("catboost_cv_predict.R")

# prepare dataset 
train <- read.csv("./input/train.csv")
test <- read.csv("./input/test.csv")
test$Survived <- 9999
data <- rbind(train,test)

# fe_titanic
data <- fe_titanic(data)
train <- data[data$Survived!=9999,]
test <- data[data$Survived==9999,]
test$Survived <- NULL

# tuneCatBoost
cat_w_tune <- tuneCatBoost(train, y="Survived", max_model=100, cv=3, gridtype="small-size")
saveRDS(cat_w_tune,"./cat_w_tune.Rda")

# predict
test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
p1 <- catboost.predict(cat_w_tune$finalModel, test_pool, prediction_type="Probability")
make_submit(p1, name="R_TEST_CATBOOST_TUNE_P1")
pred <- catboost.predict(cat_w_tune$finalModel, test_pool, prediction_type="Class")
make_submit(pred, name="R_TEST_CATBOOST_TUNE_PRED") # 0.82296

# catboost_cv_predict
params <- list(
  depth = cat_w_tune$bestTune$depth,
  learning_rate = cat_w_tune$bestTune$learning_rate,
  iterations = cat_w_tune$bestTune$iterations,
  l2_leaf_reg = cat_w_tune$bestTune$l2_leaf_reg,
  rsm = cat_w_tune$bestTune$rsm,
  border_count = cat_w_tune$bestTune$border_count,
  use_best_model = T,
  loss_function = 'Logloss',
  eval_metric = "Accuracy"
)
output <- catboost_cv_predict(data=train, test=test, k=12, y="Survived", params=params, train_dir = "./catboost")

# predict
pred <- ifelse(rowMeans(output$p1)>0.5,1,0)
make_submit(pred, name="R_TEST_CATBOOST_TUNE_CVPRED")
p1 <- ifelse(rowMeans(output$p1)>0.5,1,0)
make_submit(p1, name="R_TEST_CATBOOST_TUNE_CVP1")
