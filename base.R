# title : santander
# author : jacob

# library 
rm(list=ls())
gc(reset=TRUE)
library(rAutoFE)
library(data.table)
library(caret)
library(h2o)
library(e1071)
source("./make_submit.R")
source("./TheFeatures.R")
source("./add_statistics.R")

# h2o 
# h2o.shutdown(prompt = FALSE)
h2o.init(max_mem_size = "30G")

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
data <- data[sample(1:nrow(data),500),]
test <- test[sample(1:nrow(test),500),]

# TheFeatures
data <- data[, c("target",TheFeatures), with=FALSE]
test <- test[, c(TheFeatures), with=FALSE]

# add_statistics
data <- add_statistics(data)
test <- add_statistics(test)














# remove near_zero_var
# nzv_cols <- nearZeroVar(data)
# data <- data[, -nzv_cols, with=FALSE]
# cat(">> near_zero_var:", length(nzv_cols))
# dim(data)



# set x and y 
y = "target"
x = colnames(data)[colnames(data)!=y]

# transformation 
data$target <- log(data$target)

# convert h2oFrame
data_hex <- as.h2o(data)

# split dataset
splits <- h2o.splitFrame(data_hex, ratios=0.6, seed=1234)
train_hex <- splits[[1]]
valid_hex <- splits[[2]]

# run pca
# h2o.rm("H2OPCA")
# pca <- h2o.prcomp(
#   model_id = "H2OPCA",
#   training_frame = data_hex, 
#   ignore_const_cols = TRUE, 
#   pca_method = "GLRM", # GramSVD,
#   use_all_factor_levels = TRUE,
#   max_iterations = 100, # 1000
#   seed = 1234,
#   k = 30, 
#   transform = "STANDARDIZE" # Demean
# )
# summary(pca)

# run ml
aml <- h2o.automl(
  training_frame = train_hex,
  leaderboard_frame = valid_hex,
  nfolds = 5,
  x = x,
  y = y,
  max_models = 100,
  max_runtime_secs = 60*60*24,
  stopping_metric = "RMSE",
  stopping_rounds = 3,
  stopping_tolerance = 0.001,
  exclude_algos = "DeepLearning",
  seed = 1234
)
aml
aml@leaderboard
h2o.saveModel(aml@leader, path="./model", force=TRUE)

# predict 
test <- test[, x, with=FALSE]
test_hex <- as.h2o(test)
pred <- predict(aml@leader,test_hex)
pred <- as.data.frame(pred)

# exp
pred <- exp(pred)

# submit
make_submit(pred$predict, name = "sample03")










