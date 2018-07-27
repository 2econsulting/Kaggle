# title : homecredit_w_LGB
# author : jacob

# tuning code
source(file.path(path_code,"LGB/tuneLGB.R"))
source(file.path(path_code,"LGB/cvpredictLGB.R"))

# set file
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

# ------------------------
#  optimal Depth Range
# ------------------------
params <- expand.grid(
  max_depth = seq(from=3, to=9, by=1)
)
optimalDepthRange <- tuneLGB(head(data, sample_num), y=y, params=params, k=kfolds, max_model=nrow(params))

# ------------------------
# optimal hyper-params
# ------------------------
params <- expand.grid(
  max_depth = head(optimalDepthRange$scores$max_depth, 3),
  subsample = seq(from=0.6, to=1, by=0.01),
  colsample_bytree = seq(from=0.6, to=1, by=0.01), 
  num_leaves = c(15, 31, 63, 127, 255, 511),
  min_data = seq(from=20, to=200, by=20)  # if dataset is small like 100 set to 1
)
lgb_tuning_rule = which(params$num_leaves < 2^unique(params$max_depth))
params <- params[lgb_tuning_rule, ]
optimalParams <- tuneLGB(head(data, sample_num), y=y, params=params, k=kfolds, max_model=max_model)

# ------------------------
# cvpredict catboost 
# ------------------------
params = as.list(head(optimalParams$scores[names(params)],1))
output <- cvpredictLGB(data, test, k=kfolds*2, y=y, params=params)
cat(">> cv_score :", output$score)

# save param
file_param = paste0("PARAM_LGB",round(output$score,3)*10^3,table_nm,".Rda")
saveRDS(optimalParams$scores, file.path(path_output, file_param))
cat(">> best params saved! \n")

# save ztable
file_ztable = paste0("ZTABLE_LGB",round(output$score,3)*10^3,table_nm,".csv")
fwrite(data.frame(ztable=output$ztable), file.path(path_output, file_ztable))
cat(">> ztable saved! \n")

# save submit
file_pred = paste0("SUBMIT_LGB",round(output$score,3)*10^3,table_nm,".csv")
submit[,y] <- ifelse(output$pred>1, 1, output$pred)
fwrite(submit, file.path(path_output, file_pred))
cat(">> submit saved! \n")


