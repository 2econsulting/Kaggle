# title : titanic 
# authro : jacob 

# library 
library(rAutoFE)
library(xgboost)
library(Matrix)
source('fe_titanic.R')
source('make_submit.R')
source('xgb_cv_predict.R')

# prepare dataset 
train0 <- read.csv("./input/train.csv")
test <- read.csv("./input/test.csv")
test$Survived <- 9999
data <- rbind(train0,test)
str(data)

# fe_titanic
data <- fe_titanic(data)
train0 <- data[data$Survived!=9999,]
test <- data[data$Survived==9999,]
test$Survived <- NULL
head(train0)

# split
splits  <- splitFrame(dt = train0, ratio = c(0.6,0.2), seed = 1234)
train   <- splits[[1]]
valid   <- rbind(splits[[2]],splits[[3]]) 
cat(">> number of valid:", nrow(valid))

# xgb.DMatrix
sparse_matrix_train <- sparse.model.matrix(Survived~.-1, data = train)
dtrain <- xgb.DMatrix(data = sparse_matrix_train, label = train$Survived) 
sparse_matrix_valid <- sparse.model.matrix(Survived~.-1, data = valid)
dvalid <- xgb.DMatrix(data = sparse_matrix_valid, label = valid$Survived)
sparse_matrix_test <- sparse.model.matrix(~.-1, data = test)
dtest <- xgb.DMatrix(data = sparse_matrix_test)
watchlist <- list(eval = dvalid)

# xgb 
bst <- xgb.train(
  data = dtrain, 
  watchlist = watchlist,
  # max.depth = 4,
  eta = 0.01,
  early_stopping_rounds = 3,
  nround = 1000,
  verbose = 1,
  eval_metric = "logloss",
  objective = "binary:logistic"
)

# predict and submit 
p1 <- predict(bst, newdata=dtest)
make_submit(p1,"R_TEST_XGB_DEFAULT_P1")
pred <- ifelse(predict(bst, newdata=dtest)>0.5,1,0)
make_submit(pred,"R_TEST_XGB_DEFAULT_PRED")

# xgb cv predict 
params <- list(eta = 0.01)
output <- xgb_cv_predict(data=train0, test=test, k=12, y="Survived", params=params)

# submit
p1 <- rowMeans(output$p1)
p1 <- ifelse(p1>0.5,1,0)
sum(p1)
make_submit(p1,"R_TEST_XGB_DEFAULT_CVPRED_V1")
pred <- rowMeans(output$pred)
pred <- ifelse(pred>0.5,1,0)
sum(pred)
make_submit(pred,"R_TEST_XGB_DEFAULT_CVPRED_V2")



