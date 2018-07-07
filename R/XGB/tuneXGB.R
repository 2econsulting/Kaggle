# title : tuneXGB
# author : jacob

tuneXGB <- function(data, y, params){
  
  # split
  splits  <- splitFrame(dt = data, ratio = c(0.6,0.2), seed = 1234)
  train   <- splits[[1]]
  valid   <- rbind(splits[[2]],splits[[3]]) 
  cat(">> number of valid:", nrow(valid))
  
  # xgb.DMatrix
  #sparse_matrix_train <- sparse.model.matrix(Survived~.-1, data = train)
  #dtrain <- xgb.DMatrix(data = sparse_matrix_train, label = train$Survived) 
  #sparse_matrix_valid <- sparse.model.matrix(Survived~.-1, data = valid)
  #dvalid <- xgb.DMatrix(data = sparse_matrix_valid, label = valid$Survived)
  dtrain <- prepXGB(data=train, y=y)
  dvalid <- prepXGB(data=valid, y=y)
  watchlist <- list(eval = dvalid)
  
  # gridOptions
  gridOptions <- expand.grid(params)
  
  # train 
  model <- list()
  perf <- numeric(nrow(gridOptions))
  for (i in 1:nrow(gridOptions)) {
    model[[i]] <- xgb.train(
      list(
        max_depth = gridOptions[i, "max_depth"],
        eta = gridOptions[i, "eta"],
        alpha = gridOptions[i, "alpha"],
        lambda = gridOptions[i, "lambda"],
        subsample = gridOptions[i, "subsample"],
        colsample_bytree = gridOptions[i, "colsample_bytree"],
        min_child_weight = gridOptions[i, "min_child_weight"],
        gamma = gridOptions[i, "gamma"]
      ),
      dtrain, 
      watchlist,
      early_stopping_rounds = 3,
      nrounds = 1000, 
      verbose = 1,
      eval_metric = "logloss",
      objective = "binary:logistic"
    )
    perf[i] <- model[[1]]$best_score[[1]]
  }
  
  # train log
  cat("Model ", which.min(perf), " is lowest loss: ", min(perf), sep = "","\n")
  print(gridOptions[which.min(perf), ])
  
  # best 
  bstModel <- model[[which.min(perf)]]
  bstGrid <- gridOptions[which.min(perf), ]
  
  return(list(bstModel=bstModel, bstGrid=bstGrid))
}

# example ----
# library(rAutoFE)
# data(churn)
# params <- list(
#  max_depth = c(6, 2, 3, 4, 5, 7, 9, 11, 13),
#  eta = c(0.3, 0.1, 0.5, 0.01, 0.03),
#  alpha = c(0, 0.01),
#  lambda = c(0, 0.01),
#  subsample = c(1, 0.9),
#  colsample_bytree = c(1, 0.9),
#  min_child_weight = c(1, 2, 3),
#  gamma = c(0, 1) 
# )
# xgb <- tuneXGB(data=churn, y="Churn.", params=params)