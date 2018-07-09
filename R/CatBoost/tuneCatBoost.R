# title : tuneCatBoost
# author : jacob

# tuneCatBoost
tuneCatBoost <- function(data, y, max_models, cv, gridtype="small-size"){
  
  data_x <- data[, colnames(data)[colnames(data)!=y]]
  data_y <- data[, colnames(data)[colnames(data)==y]]
  
  fit_control <- caret::trainControl(
    method = "repeatedcv", 
    number = cv, 
    search = "random",
    classProbs = TRUE,
    seeds = 1234
  )
  
  if(gridtype=="small-size"){
    grid <- expand.grid(
      depth = c(3, 4, 5, 7, 9, 11, 13),
      learning_rate = c(0.01, 0.03),
      l2_leaf_reg = 3,
      rsm = 1,
      border_count = 32,
      iterations = 500
    )
  } else{
    grid <- expand.grid(
      depth = c(2, 3, 4, 5, 7, 9, 11, 13),
      learning_rate = c(0.1, 0.5, 0.01, 0.03),
      l2_leaf_reg = c(0, 0.1, 3),
      rsm = c(1, 0.95, 0.9, 0.8),
      border_count = 32,
      iterations = 500
    )
  }
  
  cat <- catboost.caret
  cat$type <- "Classification"
  
  model <- caret::train(
    x = data_x, 
    y = as.factor(make.names(data_y)),
    method = cat,
    metric = "Accuracy",
    maximize = TRUE,
    preProc = NULL,
    tuneGrid = grid, 
    tuneLength = max_models, # max_models 
    trControl = fit_control
  )
  return(model)
}

