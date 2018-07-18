# title : santander
# author : jacob

# library 
rm(list=ls())
gc(reset=TRUE)
library(rAutoFE)
library(data.table)
library(caret)
library(e1071)
library(catboost)
source("./make_submit.R")
source("./TheFeatures.R")
source("./add_statistics.R")
source("./ml/CatBoost/trainCatBoost.R")
source("./ml/CatBoost/catboost_cv_predict.R")
source("./ml/CatBoost/tuneCatBoost.R")  
source("./ml/CatBoost/prepCatBoost.R")

# read data
data <- fread("./input/train.csv")
data[, ID:=NULL]
test <- fread("./input/test.csv")
sample <- fread("./input/sample_submission.csv")

# inspect data
str(data)
dim(data)

# samplig 
set.seed(1234)
#data <- data[sample(1:nrow(data),1000),]

# TheFeatures
data <- data[, c("target",TheFeatures), with=FALSE]
test <- test[, c(TheFeatures), with=FALSE]

# add_statistics
data <- add_statistics(data)
test <- add_statistics(test)

# target log
data$target <- log1p(data$target)

# set x and y 
y = "target"
x = colnames(data)[colnames(data)!=y]

# catboost
data_pool <- prepCatBoost(as.data.frame(data), y)
ml_traincat <- trainCatBoost(data=as.data.frame(data), ratio=c(0.6,0.2), y=y)
test_pool <- catboost.load_pool(data = test, cat_features = which(sapply(test, is.factor)))
pred <- catboost.predict(ml_traincat, test_pool)

# exp
pred <- exp(pred)-1

# submit
make_submit(pred, name = "sample06")


