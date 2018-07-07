# title : xgb_cv_predict
# author : jacob

xgb_cv_predict <- function(data, test, k, y, params){
  p1 <- list()
  pred <- list()
  eval <- list()
  for(i in 1:k){
    
    cat(">> fitting model:", i ,"\n")
    
    splits  <- splitFrame(dt = data, ratio = c(0.6, 0.2), seed = i+1)
    train   <- splits[[1]]
    valid   <- rbind(splits[[2]],splits[[3]]) 
    
    sparse_matrix_train <- sparse.model.matrix(Survived~.-1, data = train)
    dtrain <- xgb.DMatrix(data = sparse_matrix_train, label = train$Survived) 
    
    sparse_matrix_valid <- sparse.model.matrix(Survived~.-1, data = valid)
    dvalid <- xgb.DMatrix(data = sparse_matrix_valid, label = valid$Survived)
    
    watchlist <- list(eval = dvalid)
    
    bst <- xgb.train(
      list(
        max_depth = params["max_depth"][[1]],
        eta = params["eta"][[1]],
        alpha = params["alpha"][[1]],
        lambda = params["lambda"][[1]],
        subsample = params["subsample"][[1]],
        colsample_bytree = params["colsample_bytree"][[1]],
        min_child_weight = params["min_child_weight"][[1]],
        gamma = params["gamma"][[1]]
      ),
      data = dtrain, 
      watchlist = watchlist,
      early_stopping_rounds = 3,
      nround = 1000,
      verbose = 1,
      eval_metric = "logloss",
      objective = "binary:logistic"
    )
    
    # eval_metric
    eval[[i]] <- bst$best_score[[1]]
    
    # predict 
    sparse_matrix_test <- sparse.model.matrix(~.-1, data = test)
    dtest <- xgb.DMatrix(data = sparse_matrix_test)
    p1[[i]] <- predict(bst, newdata=dtest)
    pred[[i]] <- ifelse(predict(bst, newdata=dtest)>0.5,1,0)
  }
  evalDF <- data.frame(k=1:k, eval=do.call(rbind, eval))
  predDF <- setNames(data.frame(do.call(cbind, pred)), paste0(rep("pred_",k),1:k))
  p1DF <- setNames(data.frame(do.call(cbind, p1)), paste0(rep("p1_",k),1:k))
  return(list(pred=predDF, eval=evalDF, p1=p1DF))
}