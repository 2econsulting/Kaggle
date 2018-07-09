# title : tuneCatBoost
# author : jacob

# tuneCatBoost
tuneCatBoost <- function(data, y, max_models, cv, gridtype="small-size"){
  
  data_x <- data[, colnames(data)[colnames(data)!=y]]
  data_y <- data[, colnames(data)[colnames(data)==y]]
  
  if(gridtype=="small-size"){
    grid <- expand.grid(
      depth = c(3, 4, 5, 7, 9, 11, 13),
      learning_rate = c(0.01, 0.03),
      l2_leaf_reg = 3,
      rsm = c(0.9,1),
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
  
  # cat$fit
  cat$fit <- function (x, y, wts, param, lev, last, weights, classProbs, ...) 
  {
    param <- c(param, list(...), list(random_seed=1234))
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
  
  # cat$prob
  cat$prob <- 
  function (modelFit, newdata, preProc = NULL, submodels = NULL) 
  {
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
      tmp <- vector(mode = "list", length = nrow(submodels) + 
                      1)
      tmp[[1]] <- prediction
      for (j in seq(along = submodels$iterations)) {
        tmp_pred <- catboost.predict(modelFit, pool, prediction_type = "Probability", 
                                     ntree_end = submodels$iterations[j])
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
  
  # cat$type
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

