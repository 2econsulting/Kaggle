# title : tuneCatBoost
# author : jacob


# tuneCatBoost
tuneCatBoost <- function(data, y, max_model, cv, grid){
  
  require(Matrix)
  require(catboost) 
  
  data <- as.data.frame(data)
  
  # convert char to factor
  if(sum(sapply(data, function(x) is.character(x)))>0){
    char_idx <- which(sapply(data, function(x) is.character(x)))
    data[char_idx] <- lapply(data[char_idx], as.factor)
  }
  
  data_x <- data[, colnames(data)[colnames(data)!=y]]
  data_y <- data[, colnames(data)[colnames(data)==y]]
  
  set.seed(1)
  grid <- grid[sample(nrow(grid),max_model),] 
  cat <- catboost.caret
  
  # warning fixer
  cat$fit  <- fit
  cat$prob <- prob
  cat$type <- "Classification"
  
  setSeeds <- function(cv,num_grid) {
    cv = cv+1
    seeds <- vector(mode = "list", length = cv)
    set.seed(1)
    for(i in 1:(cv-1)) seeds[[i]]<- sample.int(n=1000, num_grid)
    seeds[[cv]] <- sample.int(1000, 1)
    return(seeds)
  }
  
  set.seed(1)
  fit_control <- caret::trainControl(
    method = "cv", 
    number = cv,
    search = "random", # grid, random
    seeds = setSeeds(cv=cv, num_grid=nrow(grid))
  )
  
  set.seed(1)
  model <- caret::train(
    x = data_x, 
    y = as.factor(make.names(data_y)),
    method = cat,
    metric = "Accuracy",
    maximize = TRUE,
    preProc = NULL,
    tuneGrid = grid, 
    trControl = fit_control
  )
  return(model)
}


fit <- function (x, y, wts, param, lev, last, weights, classProbs, ...){
  
  param <- c(param, list(...), list(random_seed=1))
  if (is.null(param$loss_function)) {
    param$loss_function = "RMSE"
    if (is.factor(y)) {
      param$loss_function = "Logloss"
      if (length(lev) > 2) {
        param$loss_function = "MultiClass"
      }
      y = as.double(y) - 1
    }
  }
  
  pool <- catboost.load_pool(x, y, weight = wts)
  model <- catboost.train(pool, NULL, param)
  model$lev <- lev
  return(model)
}


prob <- function (modelFit, newdata, preProc = NULL, submodels = NULL){
  
  pool <- catboost.load_pool(newdata)
  prediction <- catboost.predict(modelFit, pool, prediction_type = "Probability")
  if (is.matrix(prediction)) {
    colnames(prediction) <- modelFit$lev
    prediction <- as.data.frame(prediction)
  }
  
  param <- catboost.get_model_params(modelFit)
  if (param$loss_function$'type' == "Logloss") {
    prediction <- cbind(1 - prediction, prediction)
    colnames(prediction) <- modelFit$lev
  }
  
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) + 1)
    tmp[[1]] <- prediction
    
    for (j in seq(along = submodels$iterations)) {
      tmp_pred <- catboost.predict(
        modelFit, 
        pool, 
        prediction_type = "Probability",
        ntree_end = submodels$iterations[j]
      )
      if (is.matrix(tmp_pred)) {
        colnames(tmp_pred) <- modelFit$lev
        tmp_pred <- as.data.frame(tmp_pred)
      }
      param <- catboost.get_model_params(modelFit)
      if (param$loss_function$'type' == "Logloss") {
        tmp_pred <- cbind(1 - tmp_pred, tmp_pred)
        colnames(tmp_pred) <- modelFit$lev
      }
      tmp[[j + 1]] <- tmp_pred
    }
    prediction <- tmp
  }
  return(prediction)
}




