# title : titanic 
# authro : jacob 

# library 
library(rAutoFE)
library(xgboost)
library(Matrix)
source('fe_titanic.R')
source('make_submit.R')
source('tuneXGB.R')
source('prepXGB.R')
source('xgb_cv_predict.R')

# prepare dataset 
train <- read.csv("./input/train.csv")
test  <- read.csv("./input/test.csv")
test$Survived <- 9999
data <- rbind(train,test)

# fe_titanic
data <- fe_titanic(data)
train <- data[data$Survived!=9999,]
test <- data[data$Survived==9999,]
test$Survived <- NULL

# tuneXGB
params <- list(
  max_depth = c(6, 2, 3, 4, 5, 7, 9, 11, 13),
  eta = c(0.3, 0.1, 0.5, 0.01, 0.03),
  alpha = c(0, 0.01),
  lambda = c(0, 0.01),
  subsample = c(1, 0.9),
  colsample_bytree = c(1, 0.9),
  min_child_weight = c(1, 2, 3),
  gamma = c(0, 1) 
)
ml_xgb <- tuneXGB(data=train, y="Survived", params=params)
saveRDS(ml_xgb,"./ml_xgb.Rda")

# predict & submit 
sparse_matrix_test <- sparse.model.matrix(~.-1, data = test)
dtest <- xgb.DMatrix(data = sparse_matrix_test)
p1 <- predict(ml_xgb$bstModel, newdata=dtest)
make_submit(p1,"R_TEST_XGB_TUNE_P1")
pred <- ifelse(predict(ml_xgb$bstModel, newdata=dtest)>0.5,1,0)
make_submit(pred,"R_TEST_XGB_TUNE_PRED")

# cvpredict & submit 
params <- ml_xgb$bstGrid
output <- xgb_cv_predict(data=train, test=test, k=12, y="Survived", params=params)
p1 <- ifelse(rowMeans(output$p1)>0.5,1,0)
make_submit(p1,"R_TEST_XGB_TUNE_CVP1")
pred <- ifelse(rowMeans(output$pred)>0.5,1,0)
make_submit(pred,"R_TEST_XGB_TUNE_CVPRED")

