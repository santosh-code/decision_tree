library(readr)
cloth_mn<-read.csv(file.choose())
str(cloth_mn)

summary(cloth_mn$Sales)
sales_cut<-cut(cloth_mn$Sales,breaks =c(0,5,10,15,20),labels = c("A","B","C","D"),right=F)
sales_cut<-as.data.frame(sales_cut)
final<-cbind(sales_cut,cloth_mn)
str(final)
####sufflling rows randomly
cloth_mn_rndm<-final[order(runif(400)),]
str(cloth_mn_rndm)


####split data into train & test
cloth_mn_rndm_train<-cloth_mn_rndm[1:300,]
cloth_mn_rndm_test<-cloth_mn_rndm[1:150,]

prop.table(table(cloth_mn_rndm_train$sales_cut))
prop.table(table(cloth_mn_rndm_test$sales_cut))
prop.table(table(cloth_mn_rndm$sales_cut))

?C5.0
install.packages("C50")
library(C50)
cloth_model <- C5.0(cloth_mn_rndm_train[,-c(1,2)],cloth_mn_rndm_train$sales_cut )

# Display detailed information about the tree
summary(cloth_model)
#####Decision tree
windows()
plot(cloth_model)
#####
train_res <- predict(cloth_model, cloth_mn_rndm_train)
train_acc <- mean(cloth_mn_rndm_train$sales_cut==train_res)
train_acc
#####train acc=0.9233
test_res <- predict(cloth_model, cloth_mn_rndm_test)
test_acc <- mean(cloth_mn_rndm_test$sales_cut==test_res)
test_acc
#####test acc=0.92
