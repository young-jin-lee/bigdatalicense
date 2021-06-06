
# Libraries
library(tibble)
library(ROSE)
library(ROCR)
library(rpart)
library(rpart.plot)
library(ROSE)
library(ggplot2)
library(caret)
library(class)
library(naivebayes)
library(reshape2)
library(e1071)
require(caTools)

# Functions to use
calcRMSE <- function(label, estimation)
{
  return(sqrt(mean((label - estimation) ** 2)))
}
calcR2 <- function(label, estimation)
{
  RSS = sum((label - estimation) ** 2)
  SStot = sum((label - mean(label)) ** 2)
  return(1-RSS/SStot)
}
calAUC <- function(predCol, targetCol){
  perf <- ROCR::performance(ROCR::prediction(predCol, targetCol),'auc')
  as.numeric(perf@y.values)
}

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
minmax_norm <- function(x) {
  (x-min(x))/(max(x)-min(x))
}

fillNAwithMostFreq <- function(x){
  
  return (x[is.na(x)] <- names(sort(table(x), decreasing=T)[1]))
}

# Load Data
colnames(bank.train)

train <- bank.train
test <- bank.test.nolabel
str(train)
sapply(train, function(x){sum(x == "unknown")}) # detect by column
sapply(test, function(x){sum(x == "unknown")}) # detect by column

train[ train == "unknown" ] <- NA
test[ test == "unknown" ] <- NA

sapply(train, function(x){sum(is.na(x))}) # detect by column
sapply(test, function(x){sum(is.na(x))}) # detect by column

str(train)



sapply(train, function(x){fillNAwithMostFreq(x)})

train$job[is.na(train$job)] <- "admin."
train$marital[is.na(train$marital)] <- "married"
train$education[is.na(train$education)] <-"university.degree"
train$default[is.na(train$default)] <-"no"
train$housing[is.na(train$housing)] <-"yes"
train$loan[is.na(train$loan)] <-"no"


test$job[is.na(test$job)] <- "admin."
test$marital[is.na(test$marital)] <- "married"
test$education[is.na(test$education)] <-"university.degree"
test$default[is.na(test$default)] <-"no"
test$housing[is.na(test$housing)] <-"yes"
test$loan[is.na(test$loan)] <-"no"
str(train)


########### split train and validation

set.seed(101) 
sample = sample.split(train$age, SplitRatio = .75)
validation  = subset(train, sample == FALSE)
train = subset(train, sample == TRUE)

########### naive bayes - NO GOOD

(model <- naive_bayes(y ~ ., data = train, laplace = 1))

(train.pred <- predict(model, train, type = 'prob'))
(validation.pred <- predict(model, validation, type = 'prob'))


calAUC(train.pred[,2], train$y == "yes")
calAUC(validation.pred[,2], validation$y == 'yes')

########### logistic regression - VERY GOOD

y <- "y"
x <- c("age"      ,      "job"   ,         "marital"     ,   "education"   ,   "housing"    ,    "loan"        ,  
       "contact"    ,    "month"     ,     "day_of_week"  ,  "duration"   ,    "campaign"   ,    "pdays"       ,  
       "previous"    ,   "poutcome"   ,    "emp.var.rate"  , "cons.price.idx", "cons.conf.idx"  ,"euribor3m"   ,  
       "nr.employed")

colnames(train)
fmla <- paste(y, paste(x, collapse = "+"), sep = "~")
str(train)
model <- glm(fmla, data= train, family = binomial(link = 'logit'))
summary(model)
coefficients(model)

train$pred <- predict(model, newdata= train, type = 'response')
validation$pred <- predict(model, newdata= validation, type = 'response')

aggregate(pred ~ y, data = train, mean)
aggregate(pred ~ y, data = validation, mean)

(ctab.train <- table(pred = train$pred >0.5, atRisk = train$y))
(precision <- ctab.train[2,2] / sum(ctab.train[2,]))
(recall <- ctab.train[2,2] / sum(ctab.train[,2]))
(enrich <- precision/mean(as.numeric(train$y)))

(ctab.validation <- table(pred = validation$pred >0.5, atRisk = validation$y))
(precision <- ctab.validation[2,2] / sum(ctab.validation[2,]))
(recall <- ctab.validation[2,2] / sum(ctab.validation[,2]))
(enrich <- precision/mean(as.numeric(validation$y)))

calAUC(train$pred, train$y == "yes")
calAUC(validation$pred, validation$y == "yes")

test$pred <- predict(model, newdata= test, type = 'response')

colnames(train)
str(train)

calAUC(train$pred, train$y == "yes")
calAUC(validation$pred, validation$y == "yes")

#### SUBMIT

pred_c <- test$pred
length(pred_c)
save(pred_c, file = "classification.RData")
######################################################


train <- student.train
test <- student.test.nolabel


str(train)

sum(is.na(train))
sapply(train, function(x){table(x)}) # table by column(be careful)


set.seed(101) 
sample = sample.split(train$age, SplitRatio = .75)
validation  = subset(train, sample == FALSE)
train = subset(train, sample == TRUE)
??step
# Linear Regression - NONONONO
y <- "G3"
x <- c("school"   ,  "sex"      ,  "I(age)^2"     , "famsize"  ,  "Pstatus"  ,  "Medu"     ,  "Fedu"      ,
       "Mjob"     ,  "Fjob"    ,   "reason"   ,"studytime" , "failures" ,  "schoolsup" ,
       "famsup"  ,   "paid"   ,    "activities" ,"nursery"  ,  "higher"    , "internet"  , "romantic" ,  "famrel"    ,
       "freetime"  , "goout"   ,   "Dalc"      , "Walc"   ,    "health"   ,  "absences" ,  "class"   )
(fmla <- paste(y, paste(x, collapse = "+"), sep = "~"))

(model <- lm(fmla, train))
#step <- stepAIC(model, direction="both")
#step$anova

summary(model)
colnames(train)
train$pred <- predict(model, newdata = train)
validation$pred <- predict(model, newdata = validation)

calcRMSE(train$G3, train$pred)
calcR2(train$G3, train$pred)

calcRMSE(validation$G3, validation$pred)
calcR2(validation$G3, validation$pred)

colnames(train)

# Regression Tree - NOT GOOD ENOUGH
?rpart

model <- rpart(G3 ~ ., data = train, method = "anova", 
               control= rpart.control(maxdepth = 5, minsplit=3, minbucket = 1, cp=0.026))
plotcp(model)
train$pred <- predict(model,train,type = "vector")
validation$pred <- predict(model, validation,type = "vector")
calcRMSE(train$G3, train$pred)
calcR2(train$G3, train$pred)

calcRMSE(validation$G3, validation$pred)
calcR2(validation$G3, validation$pred)


# Random Forest - VERY GOOD !
model <- randomForest(x = train[,c(1:30,32)], y = train$G3, 
                      importance=TRUE, # to inspect variable importance
                      mtry = 31,
                      ntree=2500) # the number of tree to grow

train$pred <- predict(model,train,type = "response")
validation$pred <- predict(model, validation,type = "response")

calcRMSE(train$G3, train$pred)
calcR2(train$G3, train$pred)

calcRMSE(validation$G3, validation$pred)
calcR2(validation$G3, validation$pred)

pred_r <- predict(model,
                  test,
                  type = "response")

length(pred_r)

save(pred_r, file = "regression.RData")

load('regression.RData')
length(pred_r)
load('classification.RData')
length(pred_c)
save(pred_c, pred_r, file = "st21300551.RData")


'''
# grid serach 

dependent <<- train$G3
independent <<-train[,c(1:30,32)]

set.seed(20180613)
K = 10
R = 3
??cvFolds

cv <- cvFolds(nrow(train), K=K, R=R)
(grid <- expand.grid(ntree = c(1000,1500,2000), mtry = c(15:30)))
result <- foreach(g = 1:nrow(grid), .combine = rbind) %do%
{
  foreach(r = 1:R, .combine = rbind) %do%
  {
    foreach(k = 1:K, .combine = rbind) %do%
    {
      validation_idx <- cv$subsets[which(cv$which == k), r]
      train <- train[-validation_idx, ]
      validation <- train[-validation_idx, ]
      
      # TRAIN
      model <- randomForest(x = independent, y = dependent,
                            ntree = grid[g, "ntree"],
                            mtry = grid[g, "mtry"])
      # PREDICTION
      predicted <- predict(model, newdata = independent)
      # EVALUDATION
      rmse <- rmse(predicted,dependent)
      return(data.frame(g =g, precision = rmse))
    }
  }
}
result
library(plyr)
ddply(result, .(g), summarize, mean_precision = mean(precision))
'''

