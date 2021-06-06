
# load data
load("donors.RData")
library(tibble)
library(rpart)
library(rpart.plot)
library(class)
library(ROCR)
library(ROSE)
# functions to use 


get_accuracy <- function(tble)
{
  return ((tble[1,1] + tble[2,2]) / sum(tble))
}

get_precision <- function(tble)
{
  return (tble[2,2] / (tble[2,1] + tble[2,2]))
}

get_recall <- function(tble)
{
  return (tble[2,2] / (tble[1,2] + tble[2,2]))
}

calAUC <- function(predCol, targetCol){
  perf <- performance(prediction(predCol, targetCol),
                      'auc')
  as.numeric(perf@y.values)
}

# data exploration
class(donors_train)
str(donors_train)

summary(donors_train$age)
hist(donors_train$age)
table(donors_train$donated, donors_train$frequency)
summary(donors_train$donated)

# data preprocessing


sum(donors_train$age < 5)
donors_train <- donors_train[donors_train$age >= 5, ]
donors_test <- donors_test[donors_test$age >= 5, ]

donors_train$age_group <- cut(donors_train$age, breaks = c(-Inf, 19,29,39,49,59,69,79,89,Inf))
levels(donors_train$age_group) <-  c('<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79','80-89','90+')
donors_test$age_group <- cut(donors_test$age, breaks = c(-Inf, 19,29,39,49,59,69,79,89,Inf))
levels(donors_test$age_group) <-  c('<20', '20-29', '30-39', '40-49', '50-59','60-69', '70-79','80-89','90+')

donors_train <- as.data.frame(sapply(donors_train, function(x){as.factor(x)}))
donors_test <- as.data.frame(sapply(donors_test, function(x){as.factor(x)}))

sum(is.na(donors_train))
donors_train <- donors_train[complete.cases(donors_train), ]

sum(is.na(donors_test))
donors_test <- donors_test[complete.cases(donors_test), ]




summary(donors_train$donated)


age_train <- as.numeric(donors_train$age)
age_test <- as.numeric(donors_test$age)
donors_train$age <- NULL
donors_test$age <- NULL

# Modeling
model <- rpart(donated ~ ., data = donors_train, method = "class", 
               control= rpart.control(maxdepth = 10, minsplit=1, minbucket = 1, cp=0))
model

#### frequency, money, age, wealth rating, 
model <- rpart(donated ~ frequency + age_group + pet_owner + interest_religion , data = donors_train, method = "class", 
               control= rpart.control(maxdepth = 10, minsplit=2, minbucket = 1, cp=0))
model

# Visualization
rpart.plot(model)
rpart.plot(model, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)
?rpart.plot
printcp(model)
plotcp(model)

# Predict on the training dataset
donors_train$pred <- predict(model, donors_train, type = "class")

# performance on the training dataset
(conf.table <- table(donors_train$donated, donors_train$pred))
mean(donors_train$donated == donors_train$pred)
(accuracy <- get_accuracy(conf.table))
(precision <- get_precision(conf.table))
(recall <- get_recall(conf.table))


# Make predictions on the test dataset
donors_test$pred <- predict(model, donors_test, type = "class")

#### Evaluating DT model on test
table(donors_test$donated, donors_test$pred)
(conf.table <- table(pred = donors_test$donated, actual = donors_test$pred))
(accuracy <- get_accuracy(conf.table))
(precision <- get_precision(conf.table))
(recall <- get_recall(conf.table))

donors_train$pred <- NULL 
donors_test$pred <- NULL




#################### k NN 
# data preprocessing
load("donors.RData")
sum(donors_train$age < 5)
donors_train <- donors_train[donors_train$age >= 5, ]
donors_test <- donors_test[donors_test$age >= 5, ]

#donors_train <- as.data.frame(sapply(donors_train, function(x){as.factor(x)}))
#donors_test <- as.data.frame(sapply(donors_test, function(x){as.factor(x)}))

donors_train$wealth_rating <- as.factor(donors_train$wealth_rating)
donors_test$wealth_rating <- as.factor(donors_test$wealth_rating)
sum(is.na(donors_train))
donors_train <- donors_train[complete.cases(donors_train), ]
sum(is.na(donors_test))
donors_test <- donors_test[complete.cases(donors_test), ]

summary(donors_train$donated)
# changing value for clear interpretation
donors_train$donated <- ifelse(donors_train$donated == 1, 'y', 'n')
donors_test$donated <- ifelse(donors_test$donated == 1, 'y', 'n')
summary(donors_train)

wealth_train <- as.data.frame(predict(dummyVars(~ wealth_rating, data = donors_train), donors_train))
wealth_test <- as.data.frame(predict(dummyVars(~ wealth_rating, data = donors_test), donors_test))

donors_train$wealth_rating <- NULL
donors_test$wealth_rating <- NULL

donors_train_label <- donors_train[, 1]
donors_test_label <- donors_test[, 1]

donors_train$donated <- NULL
donors_test$donated <- NULL

donors_train <- as.data.frame(sapply(donors_train, function(x){as.numeric(x)}))
donors_test <- as.data.frame(sapply(donors_test, function(x){as.numeric(x)}))

donors_train <- cbind(donors_train, wealth_train)
donors_test <- cbind(donors_test, wealth_test)

summary(donors_train)
sqrt(nrow(donors_train))

minmax_norm <- function(x) {
  (x-min(x))/(max(x)-min(x))
}
donors_train <- as.data.frame(sapply(donors_train, minmax_norm))
donors_test <- as.data.frame(sapply(donors_test, minmax_norm))

################################### predict

calAUC <- function(predCol, targetCol){
  perf <- performance(prediction(predCol, targetCol), 'auc')
  as.numeric(perf@y.values)
}
threshold <<- 0.1
sequence_loop <- seq(1,501, 30)
sequence_loop
for (val_k in sequence_loop)
{
  test_pred <- knn(train = donors_train, test = donors_test, cl = train_label, k =
                     val_k, prob = TRUE)
  donors_test_pred_prob <- ifelse(test_pred == 'n',
                                  attributes(test_pred)$prob,
                                  1-attributes(test_pred)$prob)
  test_pred_new <- ifelse(donors_test_pred_prob > threshold,
                          'n', 'y')
  cmat <- table(test_label, test_pred_new)
  print(val_k)
  print('auc')
  print(calAUC(donors_test_pred_prob, test_label == 'n'))
  
  #print('accuracy')
  #print(mean(test_label == test_pred_new))
  #print('precision')
  #print(cmat[2,2] / sum(cmat[,2]))
  #print('recall')
  #print(cmat[2,2] / sum(cmat[2,]))
  
}

# dataframe
x <- data.frame("k" = sequence_loop, "accuracy" = c(0.5145128,0.5368964, 0.5535936,0.5732447,0.5712314,
                                                    0.5647629,0.5701298,0.5728502,0.5718113,
                                                    0.56858,0.5704697,0.5724106,0.5734582,0.5698607,
                                                    0.5722982,0.5748927,0.5744487))

ggplot(x, aes(x= k, y = accuracy))+
  geom_line()+theme_bw()+
  ggtitle("Finding the optimal k")# k : 451 accuracy: 0.5748927


test_pred <- knn(train = donors_train, test = donors_test, cl = donors_train_label,
                 k = 451, prob = TRUE)

head(test_pred)
head(attributes(test_pred)$prob)

# converting all Prob to P(n)
donors_test_pred_prob <- ifelse(test_pred == 'n',
                              attributes(test_pred)$prob,
                              1-attributes(test_pred)$prob)
head(donors_test_pred_prob)
summary(donors_test_pred_prob)

plot(performance(prediction(donors_test_pred_prob, donors_test_label == 'n'),
                 'tpr', 'fpr'))

# AUC for our kNN
calAUC(donors_test_pred_prob, donors_test_label == 'n')

threshold <- 0.97
test_pred_new <- ifelse(donors_test_pred_prob > threshold,
                             'n', 'y')

(cmat <- table(donors_test_label, test_pred_new))


#accuracy
mean(donors_test_label == test_pred_new)
#precision
cmat[2,2] / sum(cmat[,2])
#recall
cmat[2,2] / sum(cmat[2,])


##################################### naive bayes
calAUC <- function(predCol, targetCol){
  perf <- performance(prediction(predCol, targetCol),
                      'auc')
  as.numeric(perf@y.values)
}
load("donors.RData")


# preprocessing
sum(donors_train$age < 5)
donors_train <- donors_train[donors_train$age >= 5, ]
donors_test <- donors_test[donors_test$age >= 5, ]

sum(is.na(donors_train))
donors_train <- donors_train[complete.cases(donors_train), ]
sum(is.na(donors_test))
donors_test <- donors_test[complete.cases(donors_test), ]

str(donors_train)
donors_train <- as.data.frame(sapply(donors_train, function(x){as.factor(x)}))
donors_test <- as.data.frame(sapply(donors_test, function(x){as.factor(x)}))
donors_train$age <- as.integer(donors_train$age)
donors_test$age <- as.integer(donors_test$age)

donors_train$donated <- ifelse(donors_train$donated == 1, 'y', 'n')
donors_test$donated <- ifelse(donors_test$donated == 1, 'y', 'n')

donors_train$donated <- as.factor(donors_train$donated)
donors_test$donated <- as.factor(donors_test$donated)

# explore
summary(donors_train)
table(donated = donors_train$donated, donors_train$veteran)
table(donated = donors_train$donated, donors_train$bad_address)
table(donated = donors_train$donated, donors_train$has_children)
table(donated = donors_train$donated, donors_train$wealth_rating)
table(donated = donors_train$donated, donors_train$interest_veterans)
table(donated = donors_train$donated, donors_train$interest_religion)
table(donated = donors_train$donated, donors_train$pet_owner)
table(donated = donors_train$donated, donors_train$catalog_shopper)
table(donated = donors_train$donated, donors_train$recency)
table(donated = donors_train$donated, donors_train$frequency)
table(donated = donors_train$donated, donors_train$money)

#donors_train[duplicated(donors_train), ]
#donors_train <- donors_train[!duplicated(donors_train[,-1]), ]

# predict
set.seed(2018)
threshold <- 0.95 
(donors_model_lap <- naive_bayes(donated ~ ., data = donors_train, laplace = 1))
(pred <- predict(donors_model_lap, donors_test, type = 'prob'))
pred[,1]
(pred_new <- ifelse(pred[,1] > threshold,
                        'n', 'y'))

(cmat <- table(donors_test$donated, pred_new))

# ROC curve
plot(performance(prediction(pred[,1], donors_test$donated == 'n'),
                 'tpr', 'fpr'))

# AUC 
calAUC(pred[,1], donors_test$donated == 'n')

#accuracy
mean(donors_test$donated == pred_new)

#confusion matrix
(cmat <- table(donors_test$donated, pred_new))

#precision
cmat[2,2] / sum(cmat[,2])

#recall
cmat[2,2] / sum(cmat[2,])



# practice

a <- c(rep("A", 3), rep("B", 3), rep("C",2))
b <- c(1,1,2,4,1,1,2,2)
(df <-data.frame(a,b))
sum(duplicated(df))
df[duplicated(df), ]
df[!duplicated(df[,-1]), ]
