# title : tuneH2OLGB
# author : jacob 

tuneH2OLGB <- function(data, y, max_models, max_runtime_secs, params, grid_id){
  
  # convert to H2OFrame
  if(class(data) != "H2OFrame"){
    data_hex <- as.h2o(data)
    rm(data)
  }else{
    data_hex <- data
    rm(data)
  }

  # search_criteria
  search_criteria <- list(
    strategy = "RandomDiscrete",
    max_runtime_secs = max_runtime_secs,
    max_models = max_models,
    seed = 1234
  )
  
  grid <- h2o.grid(
    algorithm = "xgboost",
    grid_id = grid_id,
    x = colnames(data_hex)[colnames(data_hex)!=y], 
    y = y, 
    seed = 1234,
    training_frame = data_hex,
    nfolds = 3,
    score_each_iteration = TRUE,
    stopping_metric = "logloss",
    ntrees = 1000,
    stopping_rounds = 3,
    stopping_tolerance = 0.001,
    tree_method = "hist",
    grow_policy = "lossguide",
    hyper_params = params,
    search_criteria = search_criteria
  )
  grid_sorted <- h2o.getGrid(grid_id=grid_id, sort_by="logloss", decreasing=FALSE)
  return(grid_sorted)
}




