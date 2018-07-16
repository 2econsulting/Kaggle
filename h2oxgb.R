##Author: Arindam Dutta

####METHOD : Random-Discrete GRID SEARCH ALGORITHM for HYPERPARAMETER OPTMISATION

### NOTE: 1. I have tried to build a stacked ensemble xgboost models usinh h2o.
######    2. without FEATURE ENGG.  Didn't drop 256 constant columns as well. XGBOOST itself will take care of that part



library(h2o)

library(data.table)

h2o.init(max_mem_size = "15g")

train<-h2o.importFile("../input/train.csv",destination_frame="train_1.hex",header=T)

test<-h2o.importFile("../input/test.csv",destination_frame="test.hex",header=T)
train<-train[1:1000,]   ######## just  ran for 1000 obs due to time out  
sub<- h2o.importFile("../input/sample_submission.csv",destination_frame="sub.hex",header=T)
print("check!!")
gc()


response <- "target"

## use all other columns (except for the ID) as predictors
predictors <- setdiff(names(train), c(response, "ID"))

splits <- h2o.splitFrame(
  data = train, 
  ratios = 0.9,   ## only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]

hyper_params <- list(ntrees = seq(10, 1000, 1),
                     learn_rate = seq(0.0001, 0.2, 0.0001),
                     max_depth = seq(1, 20, 1),
                     sample_rate = seq(0.5, 1.0, 0.0001),
                     col_sample_rate = seq(0.2, 1.0, 0.0001))
search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 20, 
                        seed = 1)

# Train the grid
gbm_grid <- h2o.grid(algorithm = "xgboost",
                     grid_id = "xgboost_grid_binomial",
                     x = predictors,
                     y = response,
                     training_frame = train,
                     validation_frame=valid,
                     seed = 1,
                     nfolds = 5,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)




ensemble <- h2o.stackedEnsemble(x = predictors,
                                y = response,
                                training_frame = train,
                                validation_frame = valid,
                                base_models = gbm_grid@model_ids)

perf <- h2o.performance(ensemble, newdata = valid)
h2o.rmsle(perf)


######### FOOTNOTES: HOW TO IMPROVE MODELS:

## 1. Try Cartesian Grid Search Algorithm which will over entire space of hyperparameter combinations.
###  2. Try different algorithms -gbm, deep learning etc.
###    3.  Increse no of models ( This may casue a memory alloaction problem) --try IN GPU.


########  STAY TUNED for more.  HAPPY KAGGLING 