

# H2Oxgb_cv_predict
H2Oxgb_cv_predict <- function(data, test, k, y, param){
  
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
    
    ml_xgb <- h2o.xgboost(
      x = x,
      y = y,
      model_id = "H2Oxgb",
      training_frame = train_hex,
      validation_frame = valid_hex,
      stopping_rounds = 3,
      stopping_metric = "misclassification", # misclassification, loglogss, AUC
      stopping_tolerance = 0.001,
      seed = 1234,
      categorical_encoding = "AUTO", # SortByResponse, Enum, EnumLimited
      max_depth = 6,
      min_rows = 1,
      learn_rate = 0.3,
      sample_rate = 1,
      col_sample_rate = 1,
      reg_lambda = 0,
      reg_alpha = 0, 
      ntrees = 1000
    )
    
    # predict and submit
    maxF1 <- h2o.F1(h2o.performance(ml_xgb, newdata = valid_hex))
    maxF1_thred <- maxF1[which.max(maxF1$f1),]$threshold
    thred[[i]] <- maxF1_thred
    eval[[i]] <- h2o.accuracy(h2o.performance(ml_xgb, newdata = valid_hex), maxF1_thred)[[1]]
    
    # predict & p1
    pred_table <- h2o.predict(ml_xgb, newdata=test_hex)
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