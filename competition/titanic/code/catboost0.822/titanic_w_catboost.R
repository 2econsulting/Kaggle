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
# cat_w_tune <- tune_catboost(train, y="Survived", max_model=100, cv=4, gridtype="small-size")
# test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
# pred <- catboost.predict(cat_w_tune, test_pool, prediction_type="Class")
# make_submit(pred, name="cat_w_tune")

# catboost_cv_predict
params <- list(loss_function = 'Logloss', random_seed = 1234, eval_metric = "Accuracy", use_best_model = T)
output <- catboost_cv_predict(data=train, test=test, k=12, y="Survived", params=params, train_dir = "./catboost")

# predict 
pred <- rowMeans(output$p1)
make_submit(pred, name="R_TEST_CATBOOST_DEFAULT_CVPRED")
# make_submit(ifelse(pred>0.5,1,0), name="R_TEST_CATBOOST_DEFAULT_CVPRED")
# pred2 <- rowMeans(output$pred)
# make_submit(ifelse(pred2>0.5,1,0), name="R_TEST_CATBOOST_DEFAULT_CVPRED2")









