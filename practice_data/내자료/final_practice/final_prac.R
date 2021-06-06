
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

library(caret)
library(stringr)
library(dplyr)
library(mlr)
library(tidyr)
library(vtreat)
library(cvTools)
library(foreach)
library(randomForest)
require(caTools)
library(lubridate)
library(tibble)
library(ROSE)
library(ROCR)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(naivebayes)
library(reshape2)
library(e1071)

bank.df <- read.csv("bank-additional.csv", sep=";", header=T) # classification
vegas.df <- read.csv("LasVegasTripAdvisorReviews-Dataset.csv",sep = ',', header=T, fill = T) # classification
occupancy.df <- read.table("occupancy_datatraining.txt", sep=",", header=T) # classfication
risk.df <- read.csv("risk_factors_cervical_cancer.csv", sep = ",", fill = T, header=T) # classification
student.df <- read.csv("student.csv", sep = ",", header=T) # classification

str(bank.df)
sum(is.na(bank.df))
table(bank.df$pdays)
str(vegas.df)
sum(is.na(vegas.df))
str(occupancy.df)
sum(is.na(occupancy.df))
str(risk.df)
sum(is.na(risk.df))
str(student.df)
sum(is.na(student.df))



################################################################################## BANK

str(bank.df)
dim(bank.df)
sum(bank.df == "unknown")

sapply(bank.df, function(x){table(x)}) # table by column(be careful)

sapply(bank.df, function(x){sum(x == "unknown")}) # detect by column

# bank.df <- as.data.frame(sapply(bank.df, function(x){str_replace(x, "unknown", "NA")})) # replace str with str in each column
sapply(bank.df, function(x){sum(x == "NA")}) # detect by column

# bank.df[ bank.df == "NA" ] <- NA
bank.df[ bank.df == "unknown" ] <- NA
sapply(bank.df, function(x){sum(is.na(x))}) # detect by column
apply(bank.df, 1, function(x){sum(is.na(x))}) # detect by row

table(bank.df$default)
bank.df$default <- NULL


bank.df <- bank.df[complete.cases(bank.df), ]
str(bank.df)

# split 

set.seed(101) 
sample = sample.split(bank.df$age, SplitRatio = .75)
train = subset(bank.df, sample == TRUE)
test  = subset(bank.df, sample == FALSE)

# 칼럼 별로 이상한 값 찾기

y <- "y"
x <- c("age"      ,      "job"   ,         "marital"     ,   "education"   ,   "housing"    ,    "loan"        ,  
       "contact"    ,    "month"     ,     "day_of_week"  ,  "duration"   ,    "campaign"   ,    "pdays"       ,  
       "previous"    ,   "poutcome"   ,    "emp.var.rate"  , "cons.price.idx", "cons.conf.idx"  ,"euribor3m"   ,  
       "nr.employed")
fmla <- paste(y, paste(x, collapse = "+"), sep = "~")
model <- glm(fmla, data= train, family = binomial(link = 'logit'))
summary(model)

train$pred <- predict(model, newdata= train, type = 'response')
test$pred <- predict(model, newdata= test, type = 'response')

aggregate(pred ~ y, data = train, mean)
aggregate(pred ~ y, data = test, mean)


(ctab.test <- table(pred = test$pred >0.5, atRisk = test$y))

(precision <- ctab.test[2,2] / sum(ctab.test[2,]))

(recall <- ctab.test[2,2] / sum(ctab.test[,2]))

(enrich <- precision/mean(as.numeric(test$y)))


coefficients(model)


calAUC(train$pred, train$y == "yes")
calAUC(test$pred, test$y == "yes")




########################################################################################## vegas
str(vegas.df)
head(vegas.df)

# replace NA with mean in each column

sapply(bank.df, function(x){table(x)}) # table by column(be careful)

vegas.df$User.country <- NULL
set.seed(101) 
sample = sample.split(vegas.df$Nr..reviews, SplitRatio = .75)
train = subset(vegas.df, sample == TRUE)
test  = subset(vegas.df, sample == FALSE)
colnames(train)
y <- "Score"
x <- c("Nr..reviews"   ,    "Nr..hotel.reviews", "Helpful.votes"        ,    "Period.of.stay"  , 
       "Traveler.type"   ,  "Pool"           ,   "Gym"            ,   "Tennis.court","Spa",
       "Casino"        ,    "Free.internet"  ,   "Hotel.name"     ,   "Hotel.stars"    ,   "Nr..rooms"     ,   
       "User.continent" ,   "Member.years"   ,   "Review.month"   ,   "Review.weekday"  )
(fmla <- paste(y, paste(x, collapse = "+"), sep = "~"))
str(vegas.df)
table(vegas.df$Score)
model <- rpart(fmla, data= train, method = 'anova', cp =0)
model
pred <- predict(model, test, type = "vector")
pred
pred <- round(pred,digits=0)

calcRMSE(test$Score, pred)

sum(pred == test$Score)




################################################################## OCCUPANCY

str(occupancy.df)



head(occupancy.df$date)
tail(occupancy.df$date)
a=ymd_hms(occupancy.df$date)

year <- year(a)
table(year)
month <- month(a)
table(month)
day <- day(a)
table(day)
hour <- hour(a)
table(hour)
minute <- minute(a)
table(minute)
second <- second(a)
table(second)

occupancy.df$date <- NULL
occupancy.df$hour <- hour
occupancy.df$minute <- minute

set.seed(101) 
sample <- sample.split(occupancy.df$Temperature, SplitRatio = .75)
train <- subset(occupancy.df, sample == TRUE)
test  <- subset(occupancy.df, sample == FALSE)
y <- "Occupancy"
x <- c( "Temperature"  , "Humidity"    ,  "Light"     ,    "CO2"    ,       "HumidityRatio",
        "hour"      ,    "minute"  )
(fmla <- paste(y, paste(x, collapse = "+"), sep = "~"))

colnames(train)

model <- glm(fmla, data= train, family = binomial(link = 'logit'))

train
test
table(train$hour)
table(test$hour)

train$pred <- predict(model, newdata= train, type = 'response')
test$pred <- predict(model, newdata= test, type = 'response')

aggregate(pred ~ Occupancy, data = train, mean)
aggregate(pred ~ Occupancy, data = test, mean)


(ctab.test <- table(pred = test$pred >0.01, Occupancy = test$Occupancy))

(precision <- ctab.test[2,2] / sum(ctab.test[2,]))

(recall <- ctab.test[2,2] / sum(ctab.test[,2]))

(enrich <- precision/mean(as.numeric(test$Occupancy)))


coefficients(model)


calAUC(train$pred, train$Occupancy == 1)
calAUC(test$pred, test$Occupancy == 1)







###################################################################################### RISK

sum(train$Smokes..years. == '22')

str(risk.df)

sum(risk.df == "?")
sapply(risk.df, function(x){sum(x == "?")}) # detect by column
risk.df$STDs..Time.since.first.diagnosis <- NULL
risk.df$STDs..Time.since.last.diagnosis <- NULL

risk.df[risk.df == "?" ] <- NA

sapply(risk.df, function(x){sum(is.na(x))}) # detect by column
apply(risk.df, 1, function(x){sum(is.na(x))}) # detect by row
risk.df <- risk.df[complete.cases(risk.df), ]

sapply(risk.df, function(x){table(x)}) # table by column(be careful)

risk.df$STDs.AIDS <- NULL
risk.df$STDs.cervical.condylomatosis <- NULL

str(risk.df)
colnames(risk.df)

risk.df<- as.data.frame(sapply(risk.df, function(x){as.factor(x)})) # drop levels

risk.df[,c(1:4,9,11,13,24)]<- as.data.frame(sapply(risk.df[,c(1:4,9,11,13,24)], function(x){as.integer(x)})) # drop levels


set.seed(101) 
sample <- sample.split(risk.df$Age, SplitRatio = .75)
train <- subset(risk.df, sample == TRUE)
test  <- subset(risk.df, sample == FALSE)
y <- "Hinselmann"
x <- c( "Age"                             ,   "Number.of.sexual.partners"       ,  
        "First.sexual.intercourse"        ,   "Num.of.pregnancies"            ,    
        "Smokes"                          ,   "Smokes..years."                ,    
        "Smokes..packs.year."             ,   "Hormonal.Contraceptives"       ,    
        "Hormonal.Contraceptives..years."  ,  "IUD"                           ,    
        "IUD..years."                     ,   "STDs"                          ,    
        "STDs..number."                   ,   "STDs.condylomatosis"              , 
        "STDs.vaginal.condylomatosis"     ,   "STDs.vulvo.perineal.condylomatosis",
        "STDs.syphilis"                   ,   "STDs.pelvic.inflammatory.disease",
        "STDs.genital.herpes"             ,   "STDs.molluscum.contagiosum"    ,    
        "STDs.HIV"                        ,   "STDs.Hepatitis.B"              ,    
        "STDs.HPV"                        ,   "STDs..Number.of.diagnosis"     ,    
        "Dx.Cancer"                       ,   "Dx.CIN"       ,                     
        "Dx.HPV"                          ,   "Dx"  )
(fmla <- paste(y, paste(x, collapse = "+"), sep = "~"))


colnames(train[,-c(30,31,32)])
train <- train[,-c(30,31,32)]
test <- test[,-c(30,31,32)]
str(train)
str(test)
model <- rpart(fmla, data= train)
model <- glm(fmla, data= train, family = binomial(link = 'logit'))


train$pred <- predict(model, newdata= train, type = 'prob')
test$pred <- predict(model, newdata= test, type = 'prob')

aggregate(pred ~ Hinselmann, data = train, mean)
aggregate(pred ~ Hinselmann, data = test, mean)


(ctab.test <- table(pred = test$pred >0.01, Hinselmann = test$Hinselmann))

(precision <- ctab.test[2,2] / sum(ctab.test[2,]))

(recall <- ctab.test[2,2] / sum(ctab.test[,2]))

(enrich <- precision/mean(as.numeric(test$Hinselmann)))


coefficients(model)


table(test$Hinselmann)
calAUC(train$pred, train$Hinselmann == 1)
calAUC(test$pred, test$Hinselmann == 1)


####################################################################### STUDENT

str(student.df)

sapply(student.df, function(x){table(x)}) # table by column(be careful)

sum(is.na(student.df))

set.seed(101) 
sample <- sample.split(student.df$school, SplitRatio = .75)
train <- subset(student.df, sample == TRUE)
test  <- subset(student.df, sample == FALSE)

table(student.df, student.df$G3)
y <- "G3"
x <- c("school"   ,  "sex"      ,  "age"       , "address"   , "famsize"  ,  "Pstatus"  ,  "Medu"     ,  "Fedu"      ,
        "Mjob"     ,  "Fjob"    ,   "reason"   ,  "guardian" ,  "traveltime" ,"studytime" , "failures" ,  "schoolsup" ,
        "famsup"  ,   "paid"   ,    "activities" ,"nursery"  ,  "higher"    , "internet"  , "romantic" ,  "famrel"    ,
       "freetime"  , "goout"   ,   "Dalc"      , "Walc"   ,    "health"   ,  "absences" ,  "G1"       ,  "G2"   )
(fmla <- paste(y, paste(x, collapse = "+"), sep = "~"))

(model <- lm(fmla, train))
summary(model)
colnames(train)
train[,c(1:32)]
train$pred <- predict(model, newdata = train)
test$pred <- predict(model, newdata = test)
calcRMSE(test$G3, test$pred)
sd(test$G3)
calcR2(test$G3, test$pred)


# random forest
rf.final <- randomForest(x = train[,c(1:32)], y = train$G3, 
                         importance=TRUE, # to inspect variable importance
                         mtry = 24,
                         ntree=1000) # the number of tree to grow

rf.pred <- predict(rf.final,
                   test,
                   type = "response")
calcRMSE(test$G3, rf.pred)
calcR2(test$G3, rf.pred)




########################################################### replace with column mean

# Lets say I have a dataframe , df as following -
df <- data.frame(a=c(2,3,4,NA,5,NA),b=c(1,2,3,4,NA,NA))
df
# create a custom function
fillNAwithMean <- function(x){
  na_index <- which(is.na(x))        
  mean_x <- mean(x, na.rm=T)
  x[na_index] <- mean_x
  return(x)
}

(df <- apply(df,2,fillNAwithMean))

