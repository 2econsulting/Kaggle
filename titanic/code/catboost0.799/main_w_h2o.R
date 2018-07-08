

pred1 <- read.csv("./prob_LGB.csv")$Survived
pred2 <- read.csv("./prob_LGB_2.csv")$Survived
pred3 <- rowMeans(output$pred)

prop.table(table(pred1>0.5))
prop.table(table(pred2>0.5))
prop.table(table(pred3>0.5))

cor(pred1,pred2)
cor(pred1,pred3)

output <- data.frame(pred1,pred2,pred3)
head(output)

sum(pred1>0.5)
sum(pred2>0.5)
sum(pred3>0.5)
submit <- ifelse(rowMeans(output)>0.48,1,0)
sum(submit)
prop.table(table(rowMeans(output)>0.5))

make_submit(submit,"ensemble_v3")




