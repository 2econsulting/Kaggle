# title : prepCatBoost
# author : jacob

prepCatBoost <- function(data, y){
  target_idx   <- which(colnames(data)==y)
  cat_features <- which(sapply(data[,-target_idx], is.factor))
  data_pool <- catboost.load_pool(
    data = data[,-target_idx], 
    label = data[,target_idx], 
    cat_features = cat_features
  )
  return(data_pool)
}

# example ----
# library(rAutoFE)
# data(churn)
# train_pool <- prepCatBoost(data=churn, y="Churn.")