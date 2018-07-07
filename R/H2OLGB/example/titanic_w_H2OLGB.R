# title : titanic 
# authro : jacob 

# library 
library(rAutoFE)
library(h2o)

# source
setwd("~/kitematic/Kaggle/titanic/code/R/h2o/")
source('fe_titanic.R')
source('make_submit.R')
source('tuneH2OLGB.R')
source('h2o_lgb_cv_predict.R')

# prepare dataset 
train  <- read.csv("../../../input/train.csv")
test   <- read.csv("../../../input/test.csv")

# feature engineering  
test$Survived <- 9999
data <- rbind(train, test)
data <- fe_titanic(data)

# remove feature with high-levels
data <- data[,!colnames(data) %in% c("Cabin","Surname","letter")]

# split dataset 
train <- data[data$Survived!=9999, ] 
test  <- data[data$Survived==9999, ]
train$Survived <- as.factor(train$Survived)
test$Survived  <- NULL

# h2o 
h2o.init()
h2o.removeAll()

# step[1] : find best categorical_encoding
lgb_4_catencode <- tuneH2OLGB(
  data = train, 
  y = "Survived", 
  max_models = 100, 
  max_runtime_secs = 60*60*24,
  grid_id = "lgb_4_catencode",
  params=list(
    categorical_encoding = c("Enum","OneHotInternal","OneHotExplicit","Binary","SortByResponse")
  )
)
categorical_encoding = lgb_4_catencode@summary_table$categorical_encoding[1]
cat(">> best categorical_encoding :", categorical_encoding, "\n")

# step[3] : find range max_depth
lgb_4_maxdepth <- tuneH2OLGB(
  data = train, 
  y = "Survived", 
  max_models = 100, 
  max_runtime_secs = 60*60*24,
  grid_id = "lgb_4_maxdepth",
  params=list(
    categorical_encoding = categorical_encoding,
    max_depth = c(6, 2, 3, 4, 5, 7, 9, 11, 13)
  )
)
TOP1_maxdepth = lgb_4_maxdepth@summary_table$max_depth[1]
TOP2_maxdepth = lgb_4_maxdepth@summary_table$max_depth[2]
TOP3_maxdepth = lgb_4_maxdepth@summary_table$max_depth[3]
cat(">> TOP1_maxdepth :", TOP1_maxdepth, "\n")
cat(">> TOP2_maxdepth :", TOP2_maxdepth, "\n")
cat(">> TOP3_maxdepth :", TOP3_maxdepth, "\n")

# step[3] : search for other params 
lgb <- tuneH2OLGB(
  data = train, 
  y = "Survived", 
  max_models = 100, 
  max_runtime_secs = 60*60*24,
  grid_id = "lgb",
  params = list(
    max_depth = c(TOP1_maxdepth, TOP2_maxdepth, TOP3_maxdepth),
    categorical_encoding = categorical_encoding,
    eta = c(0.3, 0.1, 0.01),
    min_rows = c(1, 2, 3, 5),
    sample_rate = c(1, 0.9),
    col_sample_rate = c(1, 0.9),
    gamma = c(0, 1, 2, 3),
    reg_lambda = c(0, 0.01),
    reg_alpha = c(0, 0.01)
  )
)
bst <- h2o.getModel(lgb@summary_table$model_ids[1])

# predict and submit
test_hex <- as.h2o(test)
pred <- h2o.predict(bst, newdata=test_hex)
pred <- as.data.frame(pred)
pred <- as.numeric(as.character(pred$predict))
make_submit(pred,"R_TEST_H2OLGB_TUNE_PRED.csv")

# h2o_lgb_cv_predict
output = h2o_lgb_cv_predict(
  data = train, 
  test = test,
  k = 12,
  y = "Survived",
  params=bst@allparameters
)
pred <- ifelse(rowMeans(output$pred)>0.5,1,0)
make_submit(pred, name = "R_TEST_H2OLGB_TUNE_CVPRED")
p1 <- ifelse(rowMeans(output$p1)>0.5,1,0)
make_submit(p1, name = "R_TEST_H2OLGB_TUNE_CVP1") # 0.81339


