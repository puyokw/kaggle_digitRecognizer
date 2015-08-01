library(xgboost)
library(Metrics)
library(readr)
library(Rtsne)
train <- read_csv("train.csv")
test <- read_csv("test.csv")

x <- rbind(train[,-1],test)
set.seed(131)
tsne <- Rtsne(x, dims = 3, perplexity = 30, initial_dims = 50,
theta = 0.5, check_duplicates = TRUE, pca = TRUE, verbose=TRUE, max_iter = 500)

tsneTrain <- cbind(dim1=tsne$Y[,1],dim2=tsne$Y[,2])
tsneTrain <- cbind(tsneTrain,dim3=tsne$Y[,3])
tsneTrain <- as.data.frame(tsneTrain)
tsneTest <- tsneTrain[(nrow(train)+1):nrow(tsneTrain),]
tsneTrain <- tsneTrain[1:nrow(train),]

train <- cbind(train,tsneTrain)
test <- cbind(test,tsneTest)

y <- train$label
x <- rbind(train[,-1],test)
x <- as.matrix(x)
x <- matrix(as.numeric(x),nrow(x),ncol(x))
 
trind <- 1:length(y)
teind <- (nrow(train)+1):nrow(x)
pred <- 1:nrow(test)
pred[1:nrow(test)]<-0
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss", 
              "max_depth" = 5, 
              "eta" = 0.1, # defualt 0.3 shrinkage
              "min_child_weight" = 10, # 1
              #"max_delta_step" = 0, # default 0 ; it can set 1-10
              #"subsample" = 0.9, # 1
			  #"colsample_bytree" = 0.9 # 1
			  "lamdba"=1e-5,
			  "alpha"=1e-5,
			  "num_class"=10
              )
cv.nround <- 250 #search
set.seed(131)
bst.cv <- xgb.cv(param=param, data = x[trind,], label = y,  nfold = 10, nrounds=cv.nround)
print(minNum <- which.min(bst.cv$test.mlogloss.mean))
print( sprintf("%d minNum=%d test-mlogloss %s+%s",i,minNum,
bst.cv$test.mlogloss.mean[minNum],bst.cv$test.mlogloss.std[minNum]))

nround <- minNum
bst <- xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
predTest <- predict(bst,x[teind,])
predTest <- matrix(predTest,10,length(predTest)/10)
predTest <- t(predTest)
predTest <- as.data.frame(predTest)

pred <- max.col(predTest)-1
submit <- data.frame(ImageId=1:length(pred), Label=pred)
write_csv(submit,"xgbTsne.csv")
