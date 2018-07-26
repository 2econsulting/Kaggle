# title : home credit 
# author : jacob 

# library
library(data.table)

# base learners
setwd("~/GitHub/2econsulting/Kaggle/R/Classification/STACK")
source("../LGB/example/homecredit_w_lightgbm.R")
source("../CatBoost//example/homecredit_w_CatBoost.R")
source("../XGB/example/homecredit_w_XGB.R")

# --------------
# model ensemble 
# --------------

# make ztable
setwd("~/Kaggle/homecredit/output")
# ztable_lgb <- fread("./")
# ztable_xgb <- fread("./")
# ztable_cat <- fread("./")







