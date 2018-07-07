# title : h2o_lgb_cv_predict
# author : jacob

h2o_lgb_cv_predict <- function(data, test, k, y, params){
  
  if(class(data) != "H2OFrame"){
    data_hex <- as.h2o(data)
  }else{
    data_hex <- data
  }
  
  if(class(test) != "H2OFrame"){
    test_hex <- as.h2o(test)
  }else{
    test_hex <- test
  }
  
  p1 <- list()
  pred <- list()
  eval <- list()
  thred <- list()
  for(i in 1:k){
    
    cat(">> fitting model:", i ,"\n")
    
    splits  <- h2o.splitFrame(data_hex, ratio = c(0.6), seed = i)
    train_hex  <- splits[[1]]
    valid_hex  <- splits[[2]]
    
    y =  y
    x = colnames(data_hex)[colnames(data_hex)!=y]
    
    ml_lgb <- h2o.xgboost(
      x = x,
      y = y,
      training_frame = train_hex,
      validation_frame = valid_hex,
      seed = 1234,
      score_each_iteration = TRUE,
      stopping_rounds = 3,
      stopping_metric = "logloss",
      stopping_tolerance = 0.001,
      ntrees = 1000,
      tree_method = "hist",
      grow_policy = "lossguide",
      categorical_encoding = params["categorical_encoding"][[1]],
      max_depth = params["max_depth"][[1]],
      min_rows = params["min_rows"][[1]],
      learn_rate = params["learn_rate"][[1]],
      sample_rate = params["sample_rate"][[1]],
      col_sample_rate = params["col_sample_rate"][[1]],
      gamma = params["gamma"][[1]],
      reg_lambda = params["reg_lambda"][[1]],
      reg_alpha = params["reg_alpha"][[1]]
    )
    
    # predict and submit
    maxF1 <- h2o.F1(h2o.performance(ml_lgb, newdata = valid_hex))
    maxF1_thred <- maxF1[which.max(maxF1$f1),]$threshold
    thred[[i]] <- maxF1_thred
    eval[[i]] <- h2o.accuracy(h2o.performance(ml_lgb, newdata = valid_hex), maxF1_thred)[[1]]
    
    # predict & p1
    pred_table <- h2o.predict(ml_lgb, newdata=test_hex)
    pred_table <- as.data.frame(pred_table)
    pred[[i]] <- data.frame(pred = as.numeric(as.character(pred_table$predict)))
    p1[[i]] <- data.frame(pred = pred_table$p1)
  }
  evalDF <- data.frame(k=1:k, eval=do.call(rbind, eval))
  thredDF <- data.frame(k=1:k, thred=do.call(rbind, thred))
  predDF <- setNames(data.frame(do.call(cbind, pred)), paste0(rep("pred_",k),1:k))
  p1DF <- setNames(data.frame(do.call(cbind, p1)), paste0(rep("p1_",k),1:k))
  return(list(pred=predDF, eval=evalDF, thred=thredDF, p1=p1DF))
}