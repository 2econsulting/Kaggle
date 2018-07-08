

make_submit <- function(pred, name){
  PassengerId <- read.csv("../../../input/test.csv")$PassengerId
  output <- as.data.frame(cbind(PassengerId, pred))
  colnames(output) <- c("PassengerId", "Survived")
  write.csv(output, paste0("./", name, ".csv"), row.names = F)
  cat(">> done! \n")
}
