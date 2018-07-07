# title : catboost_cv_predict
# author : jacob

catboost_cv_predict <- function(data, test, k, y, params, train_dir){
  
  p1 <- list()
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
    p1[[i]] <- data.frame(pred=catboost.predict(ml_cat, test_pool, prediction_type="Probability"))
    pred[[i]] <- data.frame(pred=catboost.predict(ml_cat, test_pool, prediction_type="Class"))
  }
  evalDF <- data.frame(k=1:k, eval=do.call(rbind, eval))
  predDF <- setNames(data.frame(do.call(cbind, pred)), paste0(rep("pred_",k),1:k))
  p1DF <- setNames(data.frame(do.call(cbind, p1)), paste0(rep("p1_",k),1:k))
  return(list(pred=predDF, eval=evalDF, p1=p1DF))
}
