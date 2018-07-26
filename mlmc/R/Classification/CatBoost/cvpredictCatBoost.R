# title : cvpredictCatBoost
# author : jacob

cvpredictCatBoost <- function(data, test, k, y, params){
  
  if(k<2) stop(">> k is very small \n")
  require(caret)
  require(Metrics)
  
  data <- as.data.frame(data)
  test <- as.data.frame(test)
  
  # convert char to factor
  if(sum(sapply(data, function(x) is.character(x)))>0){
    char_idx <- which(sapply(data, function(x) is.character(x)))
    data[char_idx] <- lapply(data[char_idx], as.factor)
  }
  
  # convert char to factor
  if(sum(sapply(test, function(x) is.character(x)))>0){
    char_idx <- which(sapply(test, function(x) is.character(x)))
    test[char_idx] <- lapply(test[char_idx], as.factor)
  }
  
  # ...
  target_idx <- which(colnames(data)==y)
  cat_features <- which(sapply(data[,-target_idx], is.factor))
  params$use_best_model <- TRUE
  params$random_seed <- 1
  params$loss_function <- 'Logloss'
  params$eval_metric <- 'AUC'  
  params$od_type = "Iter"
  params$od_wait = 50
  params$thread_count = detectCores(logical=F)
  
  # make x and y 
  data_y <- data[,y]
  data_x <- data[,which(colnames(data)!=y)]

  set.seed(1)
  KFolds <- createFolds(1:nrow(data), k = k, list = TRUE, returnTrain = FALSE)
  
  opreds <- rep(NA, nrow(data))
  score  <- list()
  Kpreds <- list()
  for(i in 1:k){
    
    train_idx = unlist(KFolds[-i])
    valid_idx = unlist(KFolds[i])
    
    if(sum(sapply(data[,-target_idx], function(x) is.factor(x)))>0){
      train_pool <- catboost.load_pool(data = data_x[train_idx,], label = data_y[train_idx], cat_features = cat_features)
      valid_pool <- catboost.load_pool(data = data_x[valid_idx,], label = data_y[valid_idx], cat_features = cat_features)
    }else{
      train_pool <- catboost.load_pool(data = data_x[train_idx,], label = data_y[train_idx])
      valid_pool <- catboost.load_pool(data = data_x[valid_idx,], label = data_y[valid_idx])
    }
    
    ml_cat <- catboost.train(
      learn_pool = train_pool, 
      test_pool = valid_pool,
      params = params
    )
    
    opreds[valid_idx] = catboost.predict(ml_cat, valid_pool, prediction_type="Probability")
    score[[i]] = auc(data_y[valid_idx], opreds[valid_idx])
    
    if(sum(sapply(data_x, function(x) is.factor(x)))>0){
      test_pool <- catboost.load_pool(data = test, cat_features = cat_features)
    }else{
      test_pool <- catboost.load_pool(data = test)
    }
    
    Kpreds[[i]] = catboost.predict(ml_cat, test_pool, prediction_type="Probability") 
    cat(">> crossvalidation_score(AUC) :", score[[i]], "\n")
  }
  crossvalidation_score = do.call(rbind, score)
  cvpredict_score = auc(data_y, opreds)
  cat(">> cvpredict_score(AUC) : ", cvpredict_score, "\n")
  pred = expm1(rowMeans(do.call(cbind, Kpreds)))
  
  return(list(ztable=opreds, pred=pred, cvpredict_score=cvpredict_score, crossvalidation_score=crossvalidation_score))
}



