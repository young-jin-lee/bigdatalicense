setwd("/Users/dof0717/Desktop/2018_1/IntroToDataScience/midterm/midterm_practice")

library(MASS)
library(doBy)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)
library(ggplot2)

############################################################################ cars04

cars04 <- read.csv(file = 'cars04.csv', stringsAsFactors = F)

# Data transformation

summary(cars04)
str(cars04)
cars04[cars04$name == 'Infiniti G35 4dr',]
cars04$name <- as.factor(cars04$name)
cars04$ncyl <- as.factor(cars04$ncyl)
head(cars04)
colnames(cars04)
dim(cars04)

# NA
sum(is.na(cars04))
sapply(cars04, function(x){sum(is.na(x))})
cars04.no.na <- cars04[complete.cases(cars04),]
sum(is.na(cars04.no.na))

# Outlier
idx <- sapply(cars04.no.na, is.numeric)
cars04.numeric <- cars04.no.na[,idx]

par(mfrow = c(2,2))
name <- colnames(cars04.numeric)
name
i <<- 1
box <- function(x){
  boxplot(x, main = name[i])
  i <<- i + 1
}
sapply(cars04.numeric, function(x){box(x)})


# logical fields true 값들은 몇개인가
sapply(cars04[,c(2:8)], function(x){sum(x)})

# cyl 개수 별로 무게는 얼마나 차이가 나는가
weight.vec <- tapply(cars04.no.na$weight, cars04.no.na$ncyl, mean)
weight.df <- as.data.frame(weight.vec)
weight.df[which.max(weight.df$weight.vec),]

# 스포츠카와 스포츠카가 아닌 것의 딜러코스트는 어떻게 차이가 나는가
tapply(cars04.no.na$dealer_cost, cars04.no.na$sports_car , mean)

# horsepower가 가장 높은 5개의 케이스
cars04.no.na[which.maxn(cars04.no.na$horsepwr,5),]

# 비주얼 빈 cut & quantile의 조합
hist(cars04.no.na$dealer_cost)

(cut_points <- quantile(cars04.no.na$dealer_cost, c(0, 0.25, 0.75, 1)))
(cars04.no.na$pricepoint <- cut(cars04.no.na$dealer_cost, breaks = cut_points, include.lowest = T))
head(cars04.no.na[,c('dealer_cost', 'pricepoint')])
(levels(cars04.no.na$pricepoint) <- c('Plow25pec', 'Pnormal', 'Phigh25perc'))
head(cars04.no.na[,c('dealer_cost', 'pricepoint')],8)
table(cars04.no.na$pricepoint)

(cut_points <- quantile(cars04.no.na$weight, c(0,0.25,0.75,1)))
(cars04.no.na$weightpoint <- cut(cars04.no.na$weight, breaks = cut_points, include.lowest = T))
head(cars04.no.na[,c('weight', 'weightpoint')])
(levels(cars04.no.na$weightpoint) <- c('Wlow25pec', 'Wnormal', 'Whigh25perc'))
head(cars04.no.na[,c('weight', 'weightpoint')],8)

table(cars04.no.na$pricepoint, cars04.no.na$sports_car)

prop.table(table(cars04.no.na$pricepoint, cars04.no.na$sports_car),2) # 행퍼센트

# weight가 상위 10%와 하위 10%의 가격 차이

(cut_points <- quantile(cars04.no.na$weight, c(0,0.10,0.90,1)))
(cars04.no.na$weightpoint2 <- cut(cars04.no.na$weight, breaks = cut_points, include.lowest = T))
head(cars04.no.na[,c('weight', 'weightpoint2')])
(levels(cars04.no.na$weightpoint2) <- c('Wlow10pec', 'Wnormal', 'Whigh10perc'))
head(cars04.no.na[,c('weight', 'weightpoint2')],8)

tapply(cars04.no.na$dealer_cost, cars04.no.na$weightpoint2, mean)

# dodge만 뽑아
idx <- str_detect(cars04.no.na$name, "Dodge")
dodge.df <- cars04.no.na[idx,]
dodge.df

# 브랜드별로 평균 가격은 어떻게 되는지
temp  <- cars04.no.na %>% separate(name, c('brand', 'model'))
cars04.no.na$brand <- temp[,1]
head(cars04.no.na$brand)
avg <- tapply(cars04.no.na$dealer_cost, cars04.no.na$brand, mean) 
avg.df <- as.data.frame(avg)
avg.df
avg.df[which.maxn(avg.df$avg,5),] 


############################################################################ census 

census <- read.csv(file = 'census-retail.csv', stringsAsFactors = F)
summary(census)
str(census)

census <- census[complete.cases(census),]

census <- census %>% gather(MONTH, VAL, -YEAR) 
census <- census[order(census$YEAR, decreasing = F), ]
str(census)


avg <- tapply(census$VAL, census$MONTH, mean)
avg.df <- as.data.frame(avg)
avg.df
avg.df[which.maxn(avg.df$avg,5),] 

############################################################################ comics 이거는 넘겨

comics <- read.csv(file = 'comics.csv', stringsAsFactors = T)
summary(comics)
sapply(comics, function(x) {sum(is.na(x))})

delete.na <- function(DF, n=0) {
  print(sum(rowSums(is.na(comics)) >= n))
  DF[rowSums(is.na(DF))  < n,]
}
find.na <- function(DF, n=0) {
  print(sum(rowSums(is.na(comics)) >= n))
}
find.na(comics,2)

############################################################################ immigration

immigration <- read.csv(file = 'immigration.csv')
summary(immigration)
str(immigration)
head(immigration)

############################################################################ life

life <- read.csv(file = 'life_exp_raw.csv',stringsAsFactors = F)
summary(life)
str(life)
colnames(life)
sum(is.na(life))

life$State <- as.factor(life$State)
life$County <- as.factor(life$County)

life$State <- tolower(life$State)
life$County <- tolower(life$County)

names(life)
life.raw <- life[,-c(6,7,9,10)]

levels(life$State)

sum(life$State == levels(life$State)[1])

state[1]

x <- as.vector(rep(state[1], sum(life$State == levels(life$State)[1])))
y <- as.vector(rep(state[2], sum(life$State == levels(life$State)[2])))
x
y
c(x,y)



apply(life[,c(5:10)], 2, max)

life2 <- life %>% unite(area, State, County, sep = "-")

max_female <- aggregate(Female.life.expectancy..years. ~ area + Year , life2, FUN = mean)
max_female <- max_female[order(max_female$Female.life.expectancy..years., decreasing = T), ]
max_female[which.maxn(max_female$Female.life.expectancy..years.,5),]


max_male <- aggregate(Male.life.expectancy..years. ~ area + Year , life2, FUN = mean)
max_male <- max_male[order(max_male$Male.life.expectancy..years., decreasing = T), ]
max_male[which.maxn(max_male$Male.life.expectancy..years.,5),]

############################################################################ student

students <- read.csv(file = 'students_with_dates.csv')
summary(students)
str(students)
glimpse(students)
students$dob <- as.character(students$dob) # 날짜
students$nurse_visit <- as.character(students$nurse_visit) # 날짜 시간

students$dob <- ymd(students$dob)
students$nurse_visit <- ymd_hms(students$nurse_visit)
students <- students %>% separate(dob, c("year", "month", "day"), sep = "-")

(x <- aggregate(absences ~ year + sex, data = students, FUN = mean))
x %>% spread(sex, absences)


y <- tapply(students$absences, students$year, function(x){x>0})
sapply(y,sum)

############################################################################ income

income <- read.csv(file = 'us_income_raw.csv', stringsAsFactors = FALSE)
summary(income)
str(income)
head(income)
tail(income)
dim(income)
footnote <- income[c(9415:9427),]
income <- income[-c(9415:9427),]

income$Income <- as.integer(income$Income)
sum(is.na(income))
#income <- income[complete.cases(income), ] # long인 상태에서 NA를 지우면 안되지 ...

income.wide <- income[,c(1,2,4,5)]
income.wide <- income.wide %>% spread(Description, Income)
income.wide <- income.wide[complete.cases(income.wide), ]

income.long <- income.wide %>% gather(Description, Income, -GeoName, -GeoFips)
income.long <- income.long[order(income.long$GeoFips, decreasing = F), ]

sapply(income.wide[,c(3:5)], mean )
#tapply(income.long$Income, income.long$Description, mean) # long으로 mean을 계산하면 또 안되지 .

############################################################################ weather

weather <- read.csv(file = 'weather.csv', stringsAsFactors = FALSE) # 일단 string으로 읽어와서

# Explore data
dim(weather)
names(weather)
summary(weather) # value가 될 것 중에 factor가 되어야할 것이 뭔지 파악해
weather$measure <- as.factor(weather$measure) # 바꿔
str(weather)
glimpse(weather)

head(weather)
tail(weather)

weather[weather$month == 2, ]
weather[weather$month == 6,]
weather[weather$month == 12,]


# Tidying data
weather$X <- NULL

levels(weather$measure) # 여기서 value가 integer가 아닌 것들을 파악해. 파악된 것들은 따로 뽑아놔야해

weather <- weather %>% gather(day, Value, -year, -month, -measure)
weather$day <- gsub("X", "", weather$day)

temp <- weather[weather$measure == "Events",]

weather$Value <- as.integer(weather$Value)
sum(is.na(weather))
weather <- weather[complete.cases(weather), ]

temp$Value[temp$Value == ""] <- 'Sunny'
temp <- temp[complete.cases(temp), ]

weather <- rbind(weather, temp)

weather2 <- weather %>% unite(date, year,month,day)
weather2$date <- ymd(weather2$date)

weather2 <- weather2[order(weather2$date, decreasing = F), ]
