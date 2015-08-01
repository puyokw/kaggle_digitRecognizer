library(randomForest)
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

set.seed(131)
model <- randomForest(as.factor(label)~., data=train, ntree=100, do.trace=TRUE)

plot(model)
print(model)
pred <- predict(model,test,type="prob")
pred <- as.data.frame(pred)

pred <- max.col(pred)-1
submit <- data.frame(ImageId=1:length(pred), Label=pred)
write_csv(submit,"rfTsne.csv")
