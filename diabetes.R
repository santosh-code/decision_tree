library(readr)
diabetes<-read.csv(file.choose())
str(diabetes)
diabetes$Class.variable<-as.factor(diabetes$Class.variable)


# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
set.seed(100)
train <- sample(nrow(diabetes), 0.7*nrow(diabetes), replace = FALSE)
TrainSet <- diabetes[train,]
ValidSet <- diabetes[-train,]
summary(TrainSet)
summary(ValidSet)


model1 <- randomForest(TrainSet$Class.variable ~ ., data = TrainSet, importance = TRUE)
model1

plot(model1)
####test acc
pred<-predict(model1,ValidSet)
test_acc<-mean(pred==ValidSet$Class.variable)
test_acc
###train_acc
pred<-predict(model1,TrainSet )
train_acc<-mean(pred==TrainSet$Class.variable)
train_acc


model2 <- randomForest(TrainSet$Class.variable ~ ., data = TrainSet,method='rpart')
model2

####test acc
pred<-predict(model2,ValidSet)
test_acc<-mean(pred==ValidSet$Class.variable)
test_acc
###train_acc
pred<-predict(model2,TrainSet )
train_acc<-mean(pred==TrainSet$Class.variable)
train_acc