# title : ensemble code
# author : jacob

# set
setwd("~/kitematic/Kaggle(home)/titanic/ztable")
files <- c(
  "R_TEST_CATBOOST_TUNE_P1.csv",
  "R_TEST_H2OXGB_TUNE_CVP1.csv",
  "R_TEST_H2OLGB_TUNE_CVP1.csv"
)

# read dataset
ztables <- list()
for(i in 1:length(files)){
  tmp <- read.csv(files[i])
  tmp$Survived <- ifelse(tmp$Survived>0.5,1,0)
  ztables[[files[i]]] <- tmp
}

# yhat count
bst <- sapply(ztables,"[","Survived")
sapply(bst, function(x) sum(x))

# ensemble & submit
pred <- rowSums(do.call(cbind,bst))>2
PassengerId <- read.csv("../input/test.csv")$PassengerId
output <- as.data.frame(cbind(PassengerId, pred))
colnames(output) <- c("PassengerId", "Survived")
write.csv(output, "../output/ensemble_v21.csv",row.names = F) # 0.83253
sum(pred)



