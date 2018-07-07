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
source("catboost.R")

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

# cat_w_default
# cat_w_default <- train_catboost(train, ratio = c(0.6,0.2), y="Survived")
# test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
# pred <- catboost.predict(cat_w_default, test_pool, prediction_type="Class")
# make_submit(pred, name="cat_w_default")

# cat_w_tune
cat_w_tune <- tune_catboost(train, y="Survived", max_model=100, cv=3, gridtype="big-size")
saveRDS(cat_w_tune,"./bigsize.Rda")

# predict & submit
test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
pred <- catboost.predict(cat_w_tune$finalModel, test_pool, prediction_type="Class")
make_submit(pred, name="R_TEST_CATBOOST_TUNE_PRED_78")

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

# cvpredict & submit
p1 <- ifelse(rowMeans(output$p1)>0.5,1,0)
make_submit(p1, name="R_TEST_CATBOOST_TUNE_CVP1_78")
pred <- ifelse(rowMeans(output$pred)>0.5,1,0)
make_submit(pred, name="R_TEST_CATBOOST_TUNE_CVPRED_78")


