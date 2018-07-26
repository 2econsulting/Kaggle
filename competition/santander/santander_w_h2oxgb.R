# title : santander_w_h2oxgb
# author : jaob 

# library 
gc()
h2o.shutdown(prompt = F)
library(h2o)
library(data.table)

# h2o
h2o.init(max_mem_size = "25g")

# wd
setwd("/home/jacob/Kaggle/santander/")

# data
train <- fread("./input/train.csv")
test <- fread("./input/test.csv")
sub <- fread("./input/sample_submission.csv")

# hex
train <- as.h2o(train, destination_frame="train.hex", header=T)
test <- as.h2o(test, destination_frame="test.hex", header=T)
sub <- as.h2o(sub, destination_frame="sub.hex", header=T)

# set x and y 
response <- "target"
predictors <- setdiff(names(train), c(response, "ID"))

# split dataset 
splits <- h2o.splitFrame(
  data = train, 
  ratios = 0.9,   
  destination_frames = c("train.hex", "valid.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]


hyper_params <- list(
  ntrees = seq(10, 1000, 1),
  learn_rate = seq(0.0001, 0.2, 0.0001),
  max_depth = seq(1, 20, 1),
  sample_rate = seq(0.5, 1.0, 0.0001),
  col_sample_rate = seq(0.2, 1.0, 0.0001)
)

search_criteria <- list(
  strategy = "RandomDiscrete",
  max_models = 20, 
  seed = 1
)

# Train the grid
gbm_grid <- h2o.grid(
  algorithm = "xgboost",
  grid_id = "grid_binomial",
  x = predictors,
  y = response,
  training_frame = train,
  validation_frame=valid,
  seed = 1,
  nfolds = 5,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  hyper_params = hyper_params,
  search_criteria = search_criteria
)


ensemble <- h2o.stackedEnsemble(
  x = predictors,
  y = response,
  training_frame = train,
  validation_frame = valid,
  base_models = gbm_grid@model_ids
)

perf <- h2o.performance(ensemble, newdata = valid)
h2o.rmsle(perf)


