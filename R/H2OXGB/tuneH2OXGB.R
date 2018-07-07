# title : tuneH2OXGB
# author : jacob 

tuneH2OXGB <- function(data, y, max_models, params){
  
  # convert to H2OFrame
  if(class(data) != "H2OFrame"){
    data_hex <- as.h2o(data)
  }else{
    data_hex <- data
    rm(data)
  }
  
  # split dataset
  splits <- h2o.splitFrame(data_hex, ratios=c(0.6), seed=1234)
  train_hex <- splits[[1]]
  valid_hex <- splits[[2]]
  
  # search_criteria
  search_criteria <- list(
    strategy = "RandomDiscrete",
    max_runtime_secs = 60*60*24,
    max_models = max_models,
    seed = 1234
  )
  
  grid <- h2o.grid(
    algorithm = "xgboost",
    grid_id = "H2OXGB_Random",
    x = colnames(train_hex)[colnames(train_hex)!=y], 
    y = y, 
    seed = 1234,
    training_frame = h2o.rbind(train_hex, valid_hex),
    nfolds = 3,
    score_each_iteration = TRUE,
    stopping_metric = "logloss",
    ntrees = 10000,
    stopping_rounds = 3,
    stopping_tolerance = 0.001,
    hyper_params = params,
    search_criteria = search_criteria
  )
  grid_sorted <- h2o.getGrid(grid_id="H2OXGB_Random", sort_by="logloss", decreasing=FALSE)
  return(grid_sorted)
}




