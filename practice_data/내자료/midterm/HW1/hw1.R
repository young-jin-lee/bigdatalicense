
bank.df <- read.table("bank_hw.csv", fill = TRUE, sep = ",", header = T)
save(bank.df, file = "bank.RData")
str(bank.df)
summary(bank.df)
colnames(bank.df)

# 1
sum(bank.df$age < 30)
sum(bank.df$age > 50)

# 2
bank.df$balance_kw <- bank.df$balance * 1200

# 3
sum(bank.df$y == "yes")
sum(bank.df$y == "yes") / dim(bank.df)[1]

# 4
bank.df$pdays <- replace(bank.df$pdays, bank.df$pdays == -1, NA)
sum(is.na(bank.df$pdays))

# 5
summary(bank.df$job)

# 6
bank.df$age_group2 <- cut(bank.df$age, breaks = c(-Inf, 19,29,39,49,59, Inf))
levels(bank.df$age_group2) <-  c('<20', '20-29', '30-39', '40-49', '50-59', '60+')
cnt.df <- as.data.frame(table(bank.df$age_group2))
cnt.df
largest <- max(cnt.df$Freq)
cnt.df$Var1[cnt.df$Freq == largest]

# 7
prop.df <- prop.table(table(bank.df$y,bank.df$age_group2),2)[2,]
prop.df
prop.df <- as.data.frame(prop.df)
colnames(prop.df) <- "col"
highest <- max(prop.df)
rownames(prop.df)[prop.df$col == highest]

# 8
mean(bank.df$duration[bank.df$contact == 'cellular'])
mean(bank.df$duration[bank.df$contact == 'telephone'])
mean(bank.df$duration[bank.df$contact == 'unknown'])

# 9
bank.df <- bank.df[order(bank.df$age),]

# 10
save(bank.df, file = "SolutionToHW1.RData")

