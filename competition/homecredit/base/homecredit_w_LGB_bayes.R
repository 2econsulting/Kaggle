# title : homecredit_w_LGB_bayes
# author : jacob

# tuning code
source(file.path(path_code,"LGB/tuneLGB.R"))
source(file.path(path_code,"LGB/cvpredictLGB.R"))
source(file.path(path_code,"LGB/bayesTuneLGB.R"))

# set file
table_nm = "kageyama"
file_data = file.path(table_nm,paste0(table_nm,"_train.csv"))
file_test = file.path(table_nm,paste0(table_nm,"_test.csv"))

# read data
data = fread(file.path(path_input, file_data))
test = fread(file.path(path_input, file_test))
submit = fread(file.path(path_input, 'sample_submission.csv'))

# sampling
set.seed(1)
sample_num =round(nrow(data)*sample_rate)

# ..
data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL
names <- which(sapply(data, class) != "numeric")
data[, (names) := lapply(.SD, as.numeric), .SDcols = names]

# params
params = list(
  max_depth = c(3L, 9L),
  subsample = c(0.6, 1),
  colsample_bytree = c(0.6, 1),
  num_leaves = c(15L, 511L), # this value should be less than 2^max_depth
  min_data = c(20L, 200L),
  lambda_l2 = c(0, 5)
)

# BayesianOptimization
optimalParams <- rBayesianOptimization::BayesianOptimization(
  FUN = function(...){bayesTuneLGB(data=head(data, sample_num), k=kfolds, ...)},
  bounds = params, 
  init_points = init_points, 
  n_iter = n_iter,
  acq = "ucb", 
  kappa = 2.576, 
  eps = 0.0, 
  verbose = TRUE
)

# ------------------------
# cvpredict catboost 
# ------------------------
params = as.list(optimalParams$Best_Par)
output <- cvpredictLGB(data, test, k=kfolds*2, y=y, params=params)
cat(">> cv_score :", output$score)

# save param
file_param = paste0("PARAM_LGBbayes",round(output$score,3)*10^3,table_nm,".Rda")
saveRDS(optimalParams$scores, file.path(path_output, file_param))
cat(">> best params saved! \n")

# save ztable
file_ztable = paste0("ZTABLE_LGBbayes",round(output$score,3)*10^3,table_nm,".csv")
fwrite(data.frame(ztable=output$ztable), file.path(path_output, file_ztable))
cat(">> ztable saved! \n")

# save submit
file_pred = paste0("SUBMIT_LGBbayes",round(output$score,3)*10^3,table_nm,".csv")
submit[,y] <- ifelse(output$pred>1, 1, output$pred)
fwrite(submit, file.path(path_output, file_pred))
cat(">> submit saved! \n")





