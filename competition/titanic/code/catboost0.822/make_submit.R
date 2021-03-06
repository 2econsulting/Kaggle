

fe_impute <- function(data){
  for(i in names(which(sapply(data, is.numeric)))){
    miss_index <- which(is.na(data[,i]))
    data[miss_index, i] <- median(data[, i], na.rm=TRUE)
  }
  return(data)
}

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

make_submit <- function(pred, name){
  PassengerId <- read.csv("./input/test.csv")$PassengerId
  output <- as.data.frame(cbind(PassengerId, pred))
  colnames(output) <- c("PassengerId", "Survived")
  write.csv(output, paste0("./", name, ".csv"), row.names = F)
  cat(">> done! \n")
}
