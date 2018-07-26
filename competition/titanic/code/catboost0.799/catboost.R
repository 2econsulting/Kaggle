# title : catboost
# author : jacob


# train_catboost
train_catboost <- function(data, ratio = c(0.6, 0.2), y){
  
  splits  <- splitFrame(dt = data, ratio = ratio, seed = 1234)
  train   <- splits[[1]]
  valid   <- rbind(splits[[2]],splits[[3]]) 
  cat(">> number of valid:", nrow(valid))
  
  target_idx   <- which(colnames(data)==y)
  cat_features <- which(sapply(train[,-target_idx], is.factor))
  
  train_pool <- catboost.load_pool(data = train[,-target_idx], label = train[,target_idx], cat_features = cat_features)
  valid_pool <- catboost.load_pool(data = valid[,-target_idx], label = valid[,target_idx], cat_features = cat_features)

  fit_params <- list(
    loss_function = 'Logloss',
    logging_level = "Verbose",
    random_seed = 1234,
    eval_metric = "Accuracy",
    train_dir = "./catboost",
    use_best_model = T
  )
  
  ml_cat <- catboost.train(learn_pool = train_pool, test_pool = valid_pool, params = fit_params)
  
  return(ml_cat)
}


# tune_catboost
tune_catboost <- function(data, y, max_model, cv, gridtype="small-size"){
  
  data_x <- data[, colnames(data)[colnames(data)!=y]]
  data_y <- data[, colnames(data)[colnames(data)==y]]
  
  fit_control <- caret::trainControl(
    method = "repeatedcv", 
    number = cv, 
    search = "random",
    classProbs = TRUE
  )
  
  if(gridtype=="small-size"){
    grid <- expand.grid(
      depth = c(3, 4, 5, 7, 9, 11, 13),
      learning_rate = 0.03,
      l2_leaf_reg = c(0, 3),
      rsm = 1,
      border_count = c(32, 64),
      iterations = c(50, 100, 200)
    )
  } else{
    grid <- expand.grid(
      depth = c(3, 4, 5, 7, 9, 11, 13),
      learning_rate = c(0.1, 0.5, 0.01, 0.03),
      l2_leaf_reg = c(0, 0.1, 0.001, 3),
      rsm = c(1, 0.95, 0.9, 0.8),
      border_count = c(32, 64),
      iterations = c(50, 100, 200, 300)
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
    tuneLength = max_model, # max_model 
    trControl = fit_control
  )
  return(model$finalModel)
}


# catboost_cv_predict
catboost_cv_predict <- function(data, test, k, y, params, train_dir){
  pred <- list()
  eval <- list()
  for(i in 1:k){
    
    cat(">> fitting model:", i ,"\n")
    
    splits  <- splitFrame(dt = data, ratio = c(0.6, 0.2), seed = i)
    train   <- splits[[1]]
    valid   <- rbind(splits[[2]],splits[[3]]) 
    
    target_idx   <- which(colnames(data)==y)
    cat_features <- which(sapply(train[,-target_idx], is.factor))
    
    train_pool <- catboost.load_pool(data = train[,-target_idx], label = train[,target_idx], cat_features = cat_features)
    valid_pool <- catboost.load_pool(data = valid[,-target_idx], label = valid[,target_idx], cat_features = cat_features)
    
    params$train_dir <- train_dir
    
    ml_cat <- catboost.train(
      learn_pool = train_pool, 
      test_pool = valid_pool,
      params = params
    )
    
    # eval_metric
    catboost_log <- rjson::fromJSON(file=file.path(train_dir,"catboost_training.json"))
    eval_metric <- max(sapply(sapply(catboost_log$iterations,"[","test"),"[",1))
    eval[[i]] <- eval_metric
    
    # predict 
    test_pool <- catboost.load_pool(data = test, cat_features = cat_features)
    pred[[i]] <- data.frame(pred=catboost.predict(ml_cat, test_pool, prediction_type="Probability"))
  }
  evalDF <- data.frame(k=1:k, eval=do.call(rbind, eval))
  predDF <- setNames(data.frame(do.call(cbind, pred)), paste0(rep("pred",k),1:k))
  return(list(pred=predDF, eval=evalDF))
}
