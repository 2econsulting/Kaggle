# title : add_statistics

add_statistics <- function(data){
  data[data==0] <- NA
  data[,np_nans:=apply(data, 1, function(x) sum(is.na(x)))]
  data[,the_median:=apply(data, 1, function(x) median(x, na.rm=T))]
  data[,the_mean:=apply(data, 1, function(x) mean(x, na.rm=T))]
  data[,the_sum:=apply(data, 1, function(x) sum(x, na.rm=T))]
  data[,the_std:=apply(data, 1, function(x) sd(x, na.rm=T))]
  data[,the_kur:=apply(data, 1, function(x) kurtosis(x, na.rm=T))]
  return(data)
}