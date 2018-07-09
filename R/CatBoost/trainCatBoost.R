# title : trainCatBoost
# author : jacob

trainCatBoost <- function(data, ratio = c(0.6, 0.2), y){
  
  splits  <- splitFrame(dt = data, ratio = ratio, seed = 1234)
  train   <- splits[[1]]
  valid   <- rbind(splits[[2]],splits[[3]]) 
  cat(">> number of valid:", nrow(valid))
  
  target_idx   <- which(colnames(data)==y)
  cat_features <- which(sapply(train[,-target_idx], is.factor))
  
  train_pool <- catboost.load_pool(data = train[,-target_idx], label = train[,target_idx], cat_features = cat_features)
  valid_pool <- catboost.load_pool(data = valid[,-target_idx], label = valid[,target_idx], cat_features = cat_features)

  params <- list(
    loss_function = 'Logloss',
    logging_level = "Verbose",
    random_seed = 1234,
    eval_metric = "Accuracy",
    train_dir = "./catboost",
    use_best_model = T
  )
  
  ml_cat <- catboost.train(learn_pool = train_pool, test_pool = valid_pool, params = params)
  
  return(ml_cat)
}
