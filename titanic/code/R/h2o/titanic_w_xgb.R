# title : titanic 
# authro : jacob 

# library 
library(rAutoFE)
library(h2o)

# source
setwd("~/kitematic/titanic/code/h2o/")
source('fe_titanic.R')
source('make_submit.R')
source('H2Oxgb_cv_predict.R')

# prepare dataset 
train  <- read.csv("../../input/train.csv")
test   <- read.csv("../../input/test.csv")

# feature engineering  
test$Survived <- 9999
data <- rbind(train, test)
data <- fe_titanic(data)

# remove feature with high-levels
data <- data[,!colnames(data) %in% c("Cabin","Surname","letter")]
str(data)

# split dataset after fe
train <- data[data$Survived!=9999, ] 
test  <- data[data$Survived==9999, ]
test$Survived <- NULL

# build model 
h2o.init()
h2o.removeAll()
data_hex <- as.h2o(train)
data_hex$Survived <- h2o.asfactor(data_hex$Survived)
test_hex <- as.h2o(test)
splits <- h2o.splitFrame(data_hex, ratios = 0.6, seed = 1234)
train_hex <- splits[[1]]
valid_hex <- splits[[2]]
colnames(data_hex)
y = "Survived"
x = colnames(data_hex)[colnames(data_hex)!=y]

ml_xgb <- h2o.xgboost(
  x = x,
  y = y,
  model_id = "H2Oxgb",
  training_frame = train_hex,
  validation_frame = valid_hex,
  stopping_rounds = 3,
  stopping_metric = "misclassification", # misclassification, loglogss, AUC
  stopping_tolerance = 0.001,
  seed = 1234,
  categorical_encoding = "AUTO", # SortByResponse, Enum, EnumLimited
  max_depth = 6,
  min_rows = 1,
  learn_rate = 0.3,
  sample_rate = 1,
  col_sample_rate = 1,
  reg_lambda = 0,
  reg_alpha = 0, 
  ntrees = 1000
)

# predict and submit
maxF1 <- h2o.F1(h2o.performance(ml_xgb, newdata = valid_hex))
maxF1_thred <- maxF1[which.max(maxF1$f1),]$threshold
h2o.accuracy(h2o.performance(ml_xgb, newdata = valid_hex), maxF1_thred)[[1]]
pred <- h2o.predict(ml_xgb, newdata=test_hex)
pred <- as.data.frame(pred)
pred <- as.numeric(as.character(pred$predict))
sum(pred)

# cv_predict
cv_pred <- H2Oxgb_cv_predict(data=data_hex, test=test_hex, k=12, y="Survived")
output <- ifelse(rowMeans(cv_pred$pred)>0.5,1,0)
sum(output)
make_submit(pred, name = "R_TEST_H2OXGB_DEFAULT_CVPRED")































