

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
  ID <- fread("./input/sample_submission.csv")$ID
  output <- as.data.frame(cbind(ID, pred))
  colnames(output) <- c("ID", "target")
  write.csv(output, paste0("./output/", name, ".csv"), row.names = F)
  cat(">> done! \n")
}
