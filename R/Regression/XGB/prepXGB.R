# title : prepXGB
# author : jaocb

prepXGB <- function(data, y){
  data_x <- data[,colnames(data)[colnames(data) != y]]
  tryCatch({data_y <- data[,y]}, error=function(e) data_y <- NULL)
  sparse_matrix_x <- sparse.model.matrix(~.-1, data = data_x)
  if(exists("data_y")){
    dmatrix <- xgb.DMatrix(data = sparse_matrix_x, label = data_y)
  }else{
    dmatrix <- xgb.DMatrix(data = sparse_matrix_x)
  }
  return(dmatrix)
}

# example ----
# library(rAutoFE)
# data(churn)
# dtrain <- prepXGB(data=churn, y="Churn.")