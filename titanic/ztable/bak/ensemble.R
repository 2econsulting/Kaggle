

setwd("~/GitHub/Kaggle/titanic/ztable")

files <- list.files()
files <- files[files!="README.md"]

ztables <- list()
for(i in 1:length(files)){
  ztables[[files[i]]] <- read.csv(files[i])
}

TOP1 <- ztables$R_TEST_CATBOOST_TUNE_P1.csv
table(round(TOP1$Survived,1))

CORR <- as.data.frame.matrix(cor(do.call(cbind, sapply(ztables,"[",2))))
TOP1_CORR <- data.frame(
  name=gsub(".csv.Survived","",rownames(CORR)),
  R_TEST_CATBOOST_TUNE_P1=CORR[,"R_TEST_CATBOOST_TUNE_P1.csv.Survived"]
)
TOP1_CORR[order(TOP1_CORR$R_TEST_CATBOOST_TUNE_P1),]

base <- ifelse(ztables$R_TEST_CATBOOST_TUNE_P1.csv$Survived>0.5,1,0)
TOP1 <- ztables$R_TEST_CATBOOST_TUNE_P1.csv$Survived
TOP2 <- ztables$PYTHON_TEST_LIGHTGBM_TUNE_CVPRED.csv$Survived
TOP3 <- ztables$PYTHON_TEST_CATBOOST_DEFAULT_CVPRED.csv$Survived

pred <- ifelse(rowSums(data.frame(TOP1*0.5, TOP2*0.25, TOP2*0.25))>0.5,1,0)
pred2 <- ifelse(rowSums(data.frame(TOP1*0.4, TOP2*0.3, TOP2*0.3))>0.5,1,0)
pred3 <- ifelse(rowSums(data.frame(TOP1*0.6, TOP2*0.2, TOP2*0.2))>0.5,1,0)
all.equal(pred,pred2)
all.equal(pred,pred3)

PassengerId <- read.csv("../input/test.csv")$PassengerId
output <- as.data.frame(cbind(PassengerId, pred))
colnames(output) <- c("PassengerId", "Survived")
write.csv(output, "../output/ensemble_v1.csv",row.names = F)

PassengerId <- read.csv("../input/test.csv")$PassengerId
output <- as.data.frame(cbind(PassengerId, pred2))
colnames(output) <- c("PassengerId", "Survived")
write.csv(output, "../output/ensemble_v2.csv",row.names = F)

PassengerId <- read.csv("../input/test.csv")$PassengerId
output <- as.data.frame(cbind(PassengerId, pred3))
colnames(output) <- c("PassengerId", "Survived")
write.csv(output, "../output/ensemble_v3.csv",row.names = F)
