# title : catboost_cv_predict
# author : jacob
require(caret)


catboost_cv_predict <- function(data, test, k, y, params, train_dir){
  
  data <- as.data.frame(data)
  test <- as.data.frame(test)
  
  set.seed(1)
  KFolds <- createFolds(1:nrow(data), k = k, list = TRUE, returnTrain = FALSE)
  
  opreds <- rep(NA, nrow(data))
  score  <- list()
  Kpreds <- list()
  for(i in 1:k){
    train_idx = unlist(KFolds[i])
    valid_idx = unlist(KFolds[-i])
    
    target_idx <- which(colnames(data)==y)
    cat_features <- which(sapply(data[,-target_idx], is.factor))
    
    train_pool <- catboost.load_pool(data = data[train_idx,][,-target_idx], label = data[train_idx,][,target_idx], cat_features = cat_features)
    valid_pool <- catboost.load_pool(data = data[valid_idx,][,-target_idx], label = data[valid_idx,][,target_idx], cat_features = cat_features)
    
    params$train_dir <- train_dir
    params$random_seed <- 1
    
    ml_cat <- catboost.train(
      learn_pool = train_pool, 
      test_pool = valid_pool,
      params = params
    )
    
    opreds[valid_idx] = catboost.predict(ml_cat, valid_pool)
    score[[i]] = mse(data[valid_idx,][,target_idx], opreds[valid_idx])^0.5
    test_pool <- catboost.load_pool(data = test, cat_features = cat_features)
    Kpreds[[i]] = catboost.predict(ml_cat, test_pool) 
    cat(">> VALID score :", score[[i]], "\n")
  }
  VALID_score = do.call(rbind, score)
  FULL_score = mse(data[,target_idx], opreds)^0.5
  cat(">> FULL score : ", FULL_score, "\n")
  pred = expm1(rowMeans(do.call(cbind, Kpreds)))
  
  return(list(pred=pred, FULL_score=FULL_score, VALID_score=VALID_score))
}



