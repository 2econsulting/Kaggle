# title : fe_santander
# author : jacob
# log y
# transofom x 
# back-fitting 
# pca
# clustering

# TheFeatures
data <- data[, c("target",TheFeatures), with=FALSE]
test <- test[, c(TheFeatures), with=FALSE]

# add_statistics
data <- add_statistics(data)
test <- add_statistics(test)

# target log
data$target <- log1p(data$target)



