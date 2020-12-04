View(Fraud_check)
str(Fraud_check)
Fraud_check$Taxable.Income[Fraud_check$Taxable.Income > 30000] <- 'Good'
Fraud_check$Taxable.Income[Fraud_check$Taxable.Income <= 30000] <- 'Risky'


Fraud_check$Taxable.Income<-as.factor(Fraud_check$Taxable.Income)

fraud_split<-Fraud_check[order(runif(600)),]
str(fraud_split)
library(C50)

train<-fraud_split[1:500,]
test<-fraud_split[501:600,]


dt_model<-C5.0(train[,-3],train$Taxable.Income)
windows()
plot(dt_model)

pred<-predict(dt_model,test[,-3])
acc_test<-mean(pred==test$Taxable.Income)
acc_test
pred<-predict(dt_model,train[,-3])
acc_train<-mean(pred==train$Taxable.Income)
acc_train
#####pruning
View(Fraud_check)
Fraud_check$Taxable.Income[Fraud_check$Taxable.Income > 30000] <- 'Good'
Fraud_check$Taxable.Income[Fraud_check$Taxable.Income <= 30000] <- 'Risky'

library(caTools)
set.seed(0)
split<-sample.split(Fraud_check$Taxable.Income,SplitRatio = 0.8)
train<-subset(Fraud_check,split==TRUE)
test<-subset(Fraud_check,split==FALSE)

library(rpart)
model<-rpart(train$Taxable.Income~.,data = train,method="class",
             control=rpart.control(cp=0),maxdepth=3)
summary(model)

###plot decison tree
library(rpart.plot)
rpart.plot(model,box.palette = "auto",digits = -3)
######compute aaccuracy for test
pred<-predict(model,test[,-3],type="class")
pred

##check acc
acc<-mean(pred==test$Taxable.Income)
acc  ####test acc=0.741
######compute aaccuracy for train
pred1<-predict(model,train[,-3],type="class")
pred1

##check acc
acc1<-mean(pred==train$Taxable.Income)
acc1  ####train acc=0.747

###for mincp value
printcp(model)
plotcp(model)

mincp<-model$cptable[which.min(model$cptable[,"xerror"]),"CP"]
#######
model_pruned<-prune(model,cp=mincp)
rpart.plot(model_pruned,box="auto",digits = -3)
