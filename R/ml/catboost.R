# title : catboost
# author : jacob


# train_catboost
train_catboost <- function(data, ratio = c(0.6, 0.2), y){
  
  splits  <- splitFrame(dt = data, ratio = ratio, seed = 1234)
  train   <- splits[[1]]
  valid   <- rbind(splits[[2]],splits[[3]]) 
  cat(">> number of valid:", nrow(valid))
  
  target_idx   <- which(colnams(data)==y)
  cat_features <- which(sapply(train[,-target_idx], is.factor))
  
  train_pool <- catboost.load_pool(data = train[,-target_idx], label = train[,target_idx], cat_features = cat_features)
  valid_pool <- catboost.load_pool(data = valid[,-target_idx], label = valid[,target_idx], cat_features = cat_features)

  fit_params <- list(
    loss_function = 'Logloss',
    logging_level = "Verbose",
    random_seed = 1234,
    iterations = 500,
    custom_loss = "Accuracy",
    eval_metric = "Accuracy",
    train_dir = "ml/catboost",
    one_hot_max_size = 30,
    use_best_model = T
  )
  
  ml_cat <- catboost.train(learn_pool = train_pool, test_pool = valid_pool, params = fit_params)
  
  return(ml_cat)
}


# tune_catboost
tune_catboost <- function(data, y, max_model, cv){
  
  data_x <- data[, colnames(data)[colnames(data)!=y]]
  data_y <- data[, colnames(data)[colnames(data)==y]]
  
  fit_control <- caret::trainControl(
    method = "repeatedcv", 
    number = cv, 
    search = "random",
    classProbs = TRUE
  )
  
  grid <- expand.grid(
    depth = c(3, 4, 5, 7, 9, 11, 13),
    learning_rate = c(0.1, 0.5, 0.01),
    l2_leaf_reg = c(0.1, 0.001, 0.0001),
    rsm = c(0.95, 0.9, 0.8),
    border_count = c(32, 64),
    iterations = c(100, 150, 200, 250)
  )
  
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
  return(model)
}
