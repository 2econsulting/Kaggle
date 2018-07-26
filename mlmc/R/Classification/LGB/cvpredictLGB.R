# title : cvpredictLGB
# author : jacob

cvpredictLGB <- function(data, test, k, y, params){
  
  if(k<2) stop(">> k is very small \n")
  require(caret)
  require(Metrics)
  
  data <- as.data.frame(data)
  test <- as.data.frame(test)
  
  # convert char to factor(no need if use lgb.prepare_rules)

  data_y <- data[,y]
  data_x <- data[,which(colnames(data)!=y)]

  # ...
  rules <- lgb.prepare_rules(data = data_x)$rules
  target_idx   <- which(colnames(data)==y)
  cat_features <- names(which(sapply(data[,-target_idx], is.factor)))
  
  set.seed(1)
  KFolds <- createFolds(1:nrow(data), k = k, list = TRUE, returnTrain = FALSE)        
  
  opreds <- rep(NA, nrow(data))
  score  <- list()
  Kpreds <- list()
  for(i in 1:k){
    
    train_idx = unlist(KFolds[-i])
    valid_idx = unlist(KFolds[i])
    
    # dtrain
    dtrain <- lgb.Dataset(
      data = as.matrix(lgb.prepare_rules(data = data_x[train_idx,],  rules = rules)[[1]]), 
      label = data_y[train_idx], 
      colnames = colnames(data_x),
      categorical_feature = cat_features
    )
    
    # dvalid
    dvalid <- lgb.Dataset(
      data = as.matrix(lgb.prepare_rules(data = data_x[valid_idx,],  rules = rules)[[1]]), 
      label = data_y[valid_idx], 
      colnames = colnames(data_x),
      categorical_feature = cat_features
    )
    
    set.seed(1)
    ml_lgb <- lgb.train(
      params = params,
      data = dtrain,
      valids = list(test = dvalid),
      objective = "binary",
      eval = "auc", 
      nrounds = 1000,
      verbosity = -1, # verbose verbosity
      record = TRUE,
      eval_freq = 10,
      num_threads = detectCores(logical=F),
      early_stopping_rounds = 50
    )
    
    mvalid <- as.matrix(lgb.prepare_rules(data=data_x[valid_idx,], rules=rules)[[1]])
    opreds[valid_idx] = predict(ml_lgb, data=mvalid, n=ml_lgb$best_iter)
    score[[i]] = auc(data_y[valid_idx], opreds[valid_idx])

    mtest <- as.matrix(lgb.prepare_rules(data=test, rules=rules)[[1]])
    Kpreds[[i]] = predict(ml_lgb, data=mtest, n=ml_lgb$best_iter)   
    cat(">> crossvalidation_score :", score[[i]], "\n")
  }
  crossvalidation_score = do.call(rbind, score)
  cvpredict_score = auc(data_y, opreds)
  cat(">> cvpredict_score : ", cvpredict_score, "\n")
  pred = expm1(rowMeans(do.call(cbind, Kpreds)))
  
  return(list(ztable=opreds, pred=pred, cvpredict_score=cvpredict_score, crossvalidation_score=crossvalidation_score))
}

