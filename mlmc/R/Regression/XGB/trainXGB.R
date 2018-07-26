# title : trainXGB
# author : jacob 

trainXGB <- function(data, y){
  
  # split
  splits  <- splitFrame(dt = train0, ratio = c(0.6,0.2), seed = 1234)
  train   <- splits[[1]]
  valid   <- rbind(splits[[2]],splits[[3]]) 
  
  # xgb.DMatrix
  dtrain <- prepXGB(data=train, y=y)
  dvalid <- prepXGB(data=valid, y=y)
  watchlist <- list(eval = dvalid)
  
  # xgb 
  model <- xgb.train(
    data = dtrain, 
    watchlist = watchlist,
    early_stopping_rounds = 3,
    nround = 1000,
    verbose = 1,
    eval_metric = "logloss",
    objective = "binary:logistic"
  )
  return(model)
}

# example ----
# library(rAutoFE)
# data(churn)
# xgb <- trainXGB(data=churn, y="Churn.")