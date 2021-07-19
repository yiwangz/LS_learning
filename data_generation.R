library(MASS)
library(mice)
##############################################################################################
##############################################################################################
cov_structure <- "auto-regressive" 

m <- 5
n <- 1000
nindpt <- 5000
d <- 10
rho <- 0.5
rho_x <- 0.2

num_rep <- 200

for(rep in 1:num_rep){
  set.seed(1234+rep)
  
  cov_x <- matrix(rho_x, d, d)
  diag(cov_x) <- 1
  Xindpt_norm <- mvrnorm(n=nindpt, mu=numeric(d), Sigma=cov_x)
  X_norm <- mvrnorm(n=n, mu=numeric(d), Sigma=cov_x)
  Xindpt <- pnorm(Xindpt_norm, mean=0, sd=1, lower.tail=TRUE, log.p=FALSE)
  X <- pnorm(X_norm, mean=0, sd=1, lower.tail=TRUE, log.p=FALSE)
  
  A <- rep(c(1,-1), each=n/2)
  
  t_basis <- matrix(0, nrow=m, ncol=m+2)
  for(i in 1:m){
    basis_func <- NULL
    for(j in 2:(m-1)){
      basis_func <- c(basis_func, max(0, ((i-j)/m)^3))
    }
    t_basis[i,] <- c(1, i/m, (i/m)^2, (i/m)^3, basis_func)
  }
  
  cov_mat <- matrix(0, ncol=m, nrow=m)
  if(cov_structure == "exchangable"){
    cov_mat[,] <- rho
    diag(cov_mat) <- 1
  }else if(cov_structure == "auto-regressive"){
    diag(cov_mat) <- 1
    for(i in 1:(m-1)){
      for(j in (i+1):m){
        cov_mat[i,j] <- rho^(abs(i-j)) 
        cov_mat[j,i] <- rho^(abs(i-j)) 
      }
    }
  }
  
  b_random <- rnorm(n, mean=0, sd=0.5)
  
  B_mat <- matrix(0, nrow=n, ncol=m)
  for(i in 1:n){
    mu <- numeric(m)
    for(j in 1:m){
      mu[j] <- 0.01+0.02*X[i,1]+t_basis[j,]%*%c(3,0.5,0.5,-3.5,-2,-2,-0.1)-  
        0.1*(0.4*X[i,5]+0.6*X[i,6]-X[i,7])*t_basis[j,]%*%c(3,0.5,0.5,-3.5,-2,-2,-0.1)+
        2*A[i]*(1+X[i,1]-log(X[i,2]+1)+2*X[i,3]^3-exp(X[i,4]))+b_random[i]
    }
    B_mat[i,] <- mvrnorm(n=1, mu=mu, Sigma=cov_mat)
  }
  
  if(min(B_mat) < 0){
    B_mat <- B_mat+abs(min(B_mat))+0.001
  }
  
  Tindpt_label <- ifelse(1+Xindpt[,1]-log(Xindpt[,2]+1)+2*Xindpt[,3]^3-exp(Xindpt[,4])>0, 1, -1)
  T_label <- ifelse(1+X[,1]-log(X[,2]+1)+2*X[,3]^3-exp(X[,4])>0, 1, -1)
  
  miss_prop <- matrix(0, nrow=n, ncol=m)
  miss_mat <- matrix(0, nrow=n, ncol=m)
  for(i in 1:n){
    for(j in 1:m){
      if(j == 1){
        miss_prop[i,j] <- 0.01
      }
      else{
        miss_prop[i,j] <- exp(1+2*j/m-0.005*B_mat[i,j-1])/
          (1+exp(1+2*j/m-0.005*B_mat[i,j-1]))
      }
      miss_mat[i,j] <- rbinom(1,1,miss_prop[i,j])
    }
  }
  apply(miss_mat, 2, sum)
  
  index <- numeric(n)  
  for(i in 1:n){
    obs_num <- sum(which(miss_mat[i,]==0))
    if(obs_num == 0){
      index[i] <- NA
    }
    else{
      index[i] <- which(miss_mat[i,]==0)[length(which(miss_mat[i,]==0))]
    }
  }
  table(index) 
  sum(table(index))
  
  ##############################################################################################
  ## MICE imputation
  B_obs <- B_mat
  B_obs[miss_mat == 1] = NA
  
  imp_data <- mice(cbind(X, A, B_obs),m=5,meth='pmm',seed=500)
  data_comp1 <- complete(imp_data,1)
  data_comp2 <- complete(imp_data,2)
  data_comp3 <- complete(imp_data,3)
  data_comp4 <- complete(imp_data,4)
  data_comp5 <- complete(imp_data,5)
  
  ##############################################################################################
  ## save results
  data_indpt <- cbind(Xindpt, Tindpt_label)
  colnames(data_indpt) <- c(paste0("X", seq(1,d,1)),"True_label")
  
  data <- cbind(X, A, T_label, B_mat, miss_mat, index-1, 
                data_comp1[,(d+2):(d+1+m)],
                data_comp2[,(d+2):(d+1+m)],
                data_comp3[,(d+2):(d+1+m)],
                data_comp4[,(d+2):(d+1+m)],
                data_comp5[,(d+2):(d+1+m)])
  colnames(data) <- c(paste0("X", seq(1,d,1)), "Assignment", "True_label", 
                      paste0("Benefit",seq(1,m,1)), paste0("miss", seq(1,m,1)), "index",
                      paste0("Imputed1_",seq(1,m,1)),
                      paste0("Imputed2_",seq(1,m,1)),
                      paste0("Imputed3_",seq(1,m,1)),
                      paste0("Imputed4_",seq(1,m,1)),
                      paste0("Imputed5_",seq(1,m,1)))
  
  write.table(data_indpt, file = paste0("data_indpt",rep,".txt"), col.names = TRUE, row.names = FALSE, sep = ",")
  write.table(data, file = paste0("data",rep,".txt"), col.names = TRUE, row.names = FALSE, sep = ",")
}






