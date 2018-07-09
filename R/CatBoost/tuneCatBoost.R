# title : tuneCatBoost
# author : jacob

# tuneCatBoost
tuneCatBoost <- function(data, y, max_models, cv, gridtype="small-size"){
  
  data_x <- data[, colnames(data)[colnames(data)!=y]]
  data_y <- data[, colnames(data)[colnames(data)==y]]
  
  if(gridtype=="small-size"){
    grid <- expand.grid(
      # depth = c(3, 4, 5, 7, 9, 11, 13),
      # learning_rate = c(0.01, 0.03),
      depth = c(3,4,5,6),
      learning_rate = c(0.1,0.3),
      l2_leaf_reg = 3,
      rsm = c(0.9,1),
      border_count = 32,
      iterations = 100
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
  
  setSeeds <- function(cv,num_grid) {
    cv = cv+1
    seeds <- vector(mode = "list", length = cv)
    set.seed(1234)
    for(i in 1:(cv-1)) seeds[[i]]<- sample.int(n=1000, num_grid)
    seeds[[cv]] <- sample.int(1000, 1)
    return(seeds)
  }
  
  set.seed(1234)
  fit_control <- caret::trainControl(
    method = "repeatedcv", 
    number = cv, 
    classProbs = TRUE,
    seeds = setSeeds(cv=cv, num_grid=nrow(grid))
  )
  
  set.seed(1234)
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

