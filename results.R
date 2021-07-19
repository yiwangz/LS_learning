##############################################################################################
##############################################################################################
data_acc <- data.frame("method"=c("OWL (super oracle - no missing)",
                                  "SS-learning (scaled tuning)",
                                  "OWL on S (last obs carry forward)",
                                  "OWL on S_imp (multiple imputation)",
                                  "OWL on S_M (completers)"),
                       "acc_training"=numeric(5),
                       "acc_independent"=numeric(5))

acc_all_super <- as.vector(as.matrix(read.csv("acc_all_super.txt", header = FALSE)))  
acc_all_tune <- as.vector(as.matrix(read.csv("acc_all_tune.txt", header = FALSE)))
acc_all_allCV <- as.vector(as.matrix(read.csv("acc_all_allCV.txt", header = FALSE)))
acc_all_impCV <- as.vector(as.matrix(read.csv("acc_all_impCV.txt", header = FALSE)))
acc_all_1CV <- as.vector(as.matrix(read.csv("acc_all_1CV.txt", header = FALSE)))

acc_indpt_super <- as.vector(as.matrix(read.csv("acc_indpt_super.txt", header = FALSE)))
acc_indpt_tune <- as.vector(as.matrix(read.csv("acc_indpt_tune.txt", header = FALSE)))
acc_indpt_allCV <- as.vector(as.matrix(read.csv("acc_indpt_allCV.txt", header = FALSE)))
acc_indpt_impCV <- as.vector(as.matrix(read.csv("acc_indpt_impCV.txt", header = FALSE)))
acc_indpt_1CV <- as.vector(as.matrix(read.csv("acc_indpt_1CV.txt", header = FALSE)))

data_acc[1,2] <- paste0(format(round(mean(acc_all_super),3), nsmall=3), " (",
                        format(round(sd(acc_all_super),3), nsmall=3), ")")
data_acc[2,2] <- paste0(format(round(mean(acc_all_tune),3), nsmall=3), " (",
                        format(round(sd(acc_all_tune),3), nsmall=3), ")")
data_acc[3,2] <- paste0(format(round(mean(acc_all_allCV),3), nsmall=3), " (",
                        format(round(sd(acc_all_allCV),3), nsmall=3), ")")
data_acc[4,2] <- paste0(format(round(mean(acc_all_impCV),3), nsmall=3), " (",
                        format(round(sd(acc_all_impCV),3), nsmall=3), ")")
data_acc[5,2] <- paste0(format(round(mean(acc_all_1CV),3), nsmall=3), " (",
                        format(round(sd(acc_all_1CV),3), nsmall=3), ")")

data_acc[1,3] <- paste0(format(round(mean(acc_indpt_super),3), nsmall=3), " (",
                        format(round(sd(acc_indpt_super),3), nsmall=3), ")")
data_acc[2,3] <- paste0(format(round(mean(acc_indpt_tune),3), nsmall=3), " (",
                        format(round(sd(acc_indpt_tune),3), nsmall=3), ")")
data_acc[3,3] <- paste0(format(round(mean(acc_indpt_allCV),3), nsmall=3), " (",
                        format(round(sd(acc_indpt_allCV),3), nsmall=3), ")")
data_acc[4,3] <- paste0(format(round(mean(acc_indpt_impCV),3), nsmall=3), " (",
                        format(round(sd(acc_indpt_impCV),3), nsmall=3), ")")
data_acc[5,3] <- paste0(format(round(mean(acc_indpt_1CV),3), nsmall=3), " (",
                        format(round(sd(acc_indpt_1CV),3), nsmall=3), ")")


################################################
data_evf <- data.frame("method"=c("OWL (super oracle - no missing)",
                                  "SS-learning (scaled tuning)",
                                  "OWL on S (last obs carry forward)",
                                  "OWL on S_imp (multiple imputation)",
                                  "OWL on S_M (completers)"),
                       "evf_B1"=numeric(5),
                       "evf_B2"=numeric(5),
                       "evf_B2"=numeric(5),
                       "evf_B4"=numeric(5),
                       "evf_B5"=numeric(5),
                       "evf_mean"=numeric(5))

evf_super <- as.matrix(read.csv("evf_B_super.txt", header = FALSE))
evf_tune <- as.matrix(read.csv("evf_B_tune.txt", header = FALSE))
evf_allCV <- as.matrix(read.csv("evf_B_allCV.txt", header = FALSE))
evf_impCV <- as.matrix(read.csv("evf_B_impCV.txt", header = FALSE))
evf_1CV <- as.matrix(read.csv("evf_B_1CV.txt", header = FALSE))

data_evf[1,2:6] <- sapply(1:5, function(i) paste0(format(round(mean(evf_super[,i]),2), nsmall=2), " (",
                                                  format(round(sd(evf_super[,i]),2), nsmall=2), ")"))
data_evf[2,2:6] <- sapply(1:5, function(i) paste0(format(round(mean(evf_tune[,i]),2), nsmall=2), " (",
                                                  format(round(sd(evf_tune[,i]),2), nsmall=2), ")"))
data_evf[3,2:6] <- sapply(1:5, function(i) paste0(format(round(mean(evf_allCV[,i]),2), nsmall=2), " (",
                                                  format(round(sd(evf_allCV[,i]),2), nsmall=2), ")"))
data_evf[4,2:6] <- sapply(1:5, function(i) paste0(format(round(mean(evf_impCV[,i]),2), nsmall=2), " (",
                                                  format(round(sd(evf_impCV[,i]),2), nsmall=2), ")"))
data_evf[5,2:6] <- sapply(1:5, function(i) paste0(format(round(mean(evf_1CV[,i]),2), nsmall=2), " (",
                                                  format(round(sd(evf_1CV[,i]),2), nsmall=2), ")"))

data_evf[1,7] <- paste0(format(round(mean(as.vector(evf_super)),2), nsmall=2)," (",
                        format(round(sd(as.vector(evf_super)),2), nsmall=2), ")")
data_evf[2,7] <- paste0(format(round(mean(as.vector(evf_tune)),2), nsmall=2)," (",
                        format(round(sd(as.vector(evf_tune)),2), nsmall=2), ")")
data_evf[3,7] <- paste0(format(round(mean(as.vector(evf_allCV)),2), nsmall=2)," (",
                        format(round(sd(as.vector(evf_allCV)),2), nsmall=2), ")")
data_evf[4,7] <- paste0(format(round(mean(as.vector(evf_impCV)),2), nsmall=2)," (",
                        format(round(sd(as.vector(evf_impCV)),2), nsmall=2), ")")
data_evf[5,7] <- paste0(format(round(mean(as.vector(evf_1CV)),2), nsmall=2)," (",
                        format(round(sd(as.vector(evf_1CV)),2), nsmall=2), ")")

################################################
data_time <- data.frame("method"=c("OWL (super oracle - no missing)",
                                   "SS-learning (scaled tuning)",
                                   "OWL on S (last obs carry forward)",
                                   "OWL on S_imp (multiple imputation)",
                                   "OWL on S_M (completers)"),
                        "time"=numeric(5))

time_super <- as.matrix(read.csv("time_super.txt", header = FALSE))
time_tune <- as.matrix(read.csv("time_tune.txt", header = FALSE))
time_allCV <- as.matrix(read.csv("time_allCV.txt", header = FALSE))
time_impCV <- as.matrix(read.csv("time_impCV.txt", header = FALSE))
time_1CV <- as.matrix(read.csv("time_1CV.txt", header = FALSE))

data_time[1,2] <- paste0(format(round(mean(time_super),2), nsmall=2), " (",
                         format(round(sd(time_super),2), nsmall=2), ")")
data_time[2,2] <- paste0(format(round(mean(time_tune),2), nsmall=2), " (",
                         format(round(sd(time_tune),2), nsmall=2), ")")
data_time[3,2] <- paste0(format(round(mean(time_allCV),2), nsmall=2), " (",
                         format(round(sd(time_allCV),2), nsmall=2), ")")
data_time[4,2] <- paste0(format(round(mean(time_impCV),2), nsmall=2), " (",
                         format(round(sd(time_impCV),2), nsmall=2), ")")
data_time[5,2] <- paste0(format(round(mean(time_1CV),2), nsmall=2), " (",
                         format(round(sd(time_1CV),2), nsmall=2), ")")


################################################
data_par <- read.csv("par_tune.txt", header = FALSE)
data_par <- as.vector(as.matrix(data_par))
theta1 <- mean(data_par)
sd(data_par)
time_vec <- seq(0.2,0.8,0.2)
lambda_vec <- round(theta1*exp(log(1/theta1)*time_vec),3)


setwd("~/Box/Synergistic_Self_Learning/SS_learning_longitudinal/simulation_20210503/results")
write.table(data_acc, file="data_acc.txt", col.names = TRUE, row.names = FALSE, sep=",")
write.table(data_evf, file="data_evf.txt", col.names = TRUE, row.names = FALSE, sep=",")
write.table(data_time, file="data_time.txt", col.names = TRUE, row.names = FALSE, sep=",")







