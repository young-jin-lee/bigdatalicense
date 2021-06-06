test<-read.csv("testdata.csv")
train<-read.csv("traindata.csv")
dim(train)

summary(train)

colSums(is.na(train))
colSums(!is.na(train))

for(i in 1:ncol(train)){
  train[is.na(train[,i]), i] <- mean(train[,i], na.rm = TRUE)
}

for(i in 1:ncol(test)){
  test[is.na(test[,i]), i] <- mean(test[,i], na.rm = TRUE)
}

train_n<-train[ , ! apply( train , 2 , function(x) any(is.na(x)) ) ]
test_n<-test[ , ! apply( test , 2 , function(x) any(is.na(x)) ) ]

train_n$rand<-runif(nrow(train_n))

Train<-train_n[train_n$rand<0.9,]
Cal<-train_n[train_n$rand>=0.9,]
Test<-test_n

Train$churn<-ifelse(Train$churn==-1,0,Train$churn) 
Cal$churn<-ifelse(Cal$churn==-1,0,Cal$churn)

#############data preprocessing###################

model<-glm(formula= churn~Var1+Var2+Var3+Var4+Var5+Var6+Var7+Var13+Var126+Var173+Var163+Var160+Var153+Var149+Var147+Var146+Var144+Var142+Var140+Var138+Var134+Var132+Var131+Var126+Var125+Var123+Var122+Var119,family = binomial(link="logit"),data=Train)

library("ROCR")
p<-predict(model,Cal,type = "response")
pr<-prediction(p,Cal$churn)
auc<-performance(pr,measure = "auc")
auc<-auc@y.values[[1]]

pred<-predict(model,Test,type = "response")
save(pred,file = "pred_21500576.Rdata")
