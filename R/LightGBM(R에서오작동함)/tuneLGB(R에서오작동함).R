# title : titanic 
# authro : jacob 

# library 
gc(reset=TRUE)
library(rAutoFE)
library(caret)
library(lightgbm)
library(xgboost)
library(Matrix)
source('fe_titanic.R')
source('make_submit.R')

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

y="Survived"
target_idx   <- which(colnames(train)==y)
cat_features <- which(sapply(train[,-target_idx], is.factor))
train[sapply(train, is.factor)] <- lapply(train[sapply(train, is.factor)],as.numeric)
train[sapply(train, is.integer)] <- lapply(train[sapply(train, is.integer)],as.numeric)

splits  <- splitFrame(dt = train, ratio = c(0.6,0.2), seed = 1234)
train   <- splits[[1]]
valid   <- rbind(splits[[2]],splits[[3]]) 
cat(">> number of valid:", nrow(valid))

train_y <- train$Survived
train$Survived <- NULL
valid_y <- valid$Survived
valid$Survived <- NULL

dtrain <- lgb.Dataset(
  train %>% as.matrix, 
  label = train_y, 
  colnames = colnames(train),
  categorical_feature = names(cat_features)
)

dvalid <- lgb.Dataset(
  valid %>% as.matrix, 
  label = valid_y, 
  colnames = colnames(valid),
  categorical_feature = names(cat_features)
)

valids <- list(test = dvalid)

grid_search <- expand.grid(
  Depth = 2:5, 
  L1 = c(0), 
  L2 = c(0), 
  learning_rate = 0.001,
  min_data_in_leaf = c(1),
  num_leaves = as.integer(c(31))
)

i=1
ml = lgb.train(
  data = dtrain,
  eval = "auc",
  boosting = "gbdt",
  objective = "binary",
  nrounds = 10,
  valids = valids,
  verbose = 2,
  early_stopping_rounds = 3,
  list(
    lambda_l1 = grid_search[i, "L1"],
    lambda_l2 = grid_search[i, "L2"],
    max_depth = grid_search[i, "Depth"],
    learning_rate = grid_search[i, "learning_rate"],
    min_data_in_leaf = grid_search[i, "min_data_in_leaf"],
    num_leaves = grid_search[i, "num_leaves"]
  )
)


