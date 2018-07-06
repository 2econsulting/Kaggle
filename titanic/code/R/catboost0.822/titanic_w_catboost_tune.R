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
cat_w_tune <- tune_catboost(train, y="Survived", max_model=100, cv=3, gridtype="small-size")
saveRDS(cat_w_tune,"./cat_w_tune.Rda")

# predict
# test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
# p1 <- catboost.predict(cat_w_tune$finalModel, test_pool, prediction_type="Probability")
# make_submit(p1, name="R_TEST_CATBOOST_TUNE_P1")
# pred <- catboost.predict(cat_w_tune$finalModel, test_pool, prediction_type="Class")
# make_submit(pred, name="R_TEST_CATBOOST_TUNE_PRED")

# catboost_cv_predict
params <- list(
  depth = 3,
  learning_rate = 0.01,
  iterations = 500,
  l2_leaf_reg = 3,
  rsm = 1,
  border_count = 32,
  use_best_model = T,
  loss_function = 'Logloss',
  eval_metric = "Accuracy"
)
output <- catboost_cv_predict(data=train, test=test, k=12, y="Survived", params=params, train_dir = "./catboost")

# predict
pred <- ifelse(rowMeans(output$p1)>0.5,1,0)
make_submit(pred, name="R_TEST_CATBOOST_TUNE_CVPRED")
p1 <- rowMeans(output$p1)
make_submit(p1, name="R_TEST_CATBOOST_TUNE_CVP1")



