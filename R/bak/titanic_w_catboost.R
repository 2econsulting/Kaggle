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
source('fe_titanic.R')
source('functions.R')
source("ml/train_catboost.R")
source("ml/tune_catboost.R")
source("ml/xgboost.R")  
source("ml/lightgbm.R")

# prepare dataset 
train  <- read.csv("./input/train.csv")
test   <- read.csv("./input/test.csv")

# feature engineering  
test$Survived <- 9999
data <- rbind(train, test)
data <- fe_titanic(data)

# remove feature with high-levels
colnames(data)
data <- data[,!colnames(data) %in% c("Cabin","Surname")]

# split dataset after fe
train <- data[data$Survived!=9999, ] 
test  <- data[data$Survived==9999, ]
test$Survived <- NULL

# base learners
base_cat <- train_catboost(train, ratio = c(0.6, 0.2), y="Survived")
tune_cat <- tune_catboost(train, y="Survived", max_model=100, cv=4)

# predict
test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
pred_baseCat <- catboost.predict(base_cat, test_pool, prediction_type="Class")
pred_tuneCat <- catboost.predict(tune_cat$finalModel, test_pool, prediction_type="Class")

# submit 
make_submit(pred_baseCat, name="baseCat")
make_submit(pred_tuneCat, name="tuneCat")





