library(tidyverse)
library(lattice)
library(caret)
library(tidyr)
library(ModelMetrics)
library(purrr)
library(rpart)
library(randomForest)
library(lubridate)
library(e1071)

setwd("~/2.Education/1.R/Harvardx/Captsone/CYO")
dat <- read.csv("Admission_Predict_Ver1.1.csv")

## Data visualization

dat %>% ggplot(aes(Serial.No.)) + geom_bar()
summary(dat$Serial.No.)
dat %>% ggplot(aes(GRE.Score)) + geom_bar()
summary(dat$GRE.Score)
dat %>% ggplot(aes(TOEFL.Score)) + geom_bar()
summary(dat$TOEFL.Score)
dat %>% ggplot(aes(University.Rating)) + geom_bar()
summary(dat$University.Rating)
dat %>% ggplot(aes(SOP)) + geom_bar()
summary(dat$SOP)
dat %>% ggplot(aes(LOR)) + geom_bar()
summary(dat$LOR)
dat %>% ggplot(aes(CGPA)) + geom_bar()
summary(dat$CGPA)
dat %>% ggplot(aes(Research)) + geom_bar()
summary(dat$Research)
dat %>% ggplot(aes(Chance.of.Admit)) + geom_bar()
summary(dat$Chance.of.Admit)

## boxplot of each attribute vs Chance of Admit
boxplot(dat$Chance.of.Admit~dat$Serial.No., ylab= "Chance.of.Admit", xlab = "Serial.No")
boxplot(dat$Chance.of.Admit~dat$GRE.Score, ylab= "Chance.of.Admit", xlab = "GRE.Score")
boxplot(dat$Chance.of.Admit~dat$TOEFL.Score, ylab= "Chance.of.Admit", xlab = "TOEFL.Score")
boxplot(dat$Chance.of.Admit~dat$University.Rating, ylab= "Chance.of.Admit", xlab = "University.Rating")
boxplot(dat$Chance.of.Admit~dat$SOP, ylab= "Chance.of.Admit", xlab = "SOP")
boxplot(dat$Chance.of.Admit~dat$LOR, ylab= "Chance.of.Admit", xlab = "LOR")
boxplot(dat$Chance.of.Admit~dat$CGPA, ylab= "Chance.of.Admit", xlab = "CGPA")
boxplot(dat$Chance.of.Admit~dat$Research, ylab= "Chance.of.Admit", xlab = "Research")

## normalization
dat.av <- sapply(dat,mean, na.rm = TRUE)
dat.sd <- sapply(dat, sd, na.rm = TRUE)
dat.med <- sapply(dat, median, na.rm = TRUE)
skewness <- (dat.av - dat.med) / dat.sd
dat.norm <- (dat - dat.av) / dat.sd

## running a correlation matrix
datcor <- cor(dat.norm, use = "complete.obs")
plot(dat.norm, use = "complete.obs")

## running a PCA
datpca <- prcomp(dat / dat.sd)
summary(datpca)
datpca$rotation


## dividing into training set and test set
set.seed(1)
test_index <- createDataPartition(y = dat$Serial.No., times = 1, p = 0.1, list = FALSE)
train <- dat[-test_index,]
test <- dat[test_index,]



## method 1: linnear regression
fit_lm <- lm(Chance.of.Admit ~ ., data = train)
dat_hat_lm <- predict(fit_lm, newdata=test)
rmse(test$Chance.of.Admit,dat_hat_lm)
summary(fit_lm)

## method 2: naive Bayes
fit_nb <- naiveBayes(Chance.of.Admit ~ ., data = dat)
dat_hat_nb <- predict(fit_nb, newdata=test)
rmse(test$Chance.of.Admit,dat_hat_nb)
print(fit_nb)

#############################################################################
###Third method: classification tree
fit_rt <- rpart(Chance.of.Admit ~ ., data = train)
test_hat_rt <- predict(fit_rt, test)
rmse_acc <- rmse(test$Chance.of.Admit,test_hat_rt)
printcp(fit_rt)
summary(fit_rt)
plot(fit_rt)
##rmse_reported: 0.9571388

#############################################################################
###Fourth method: random forest tree
fit_rf <- randomForest(Chance.of.Admit ~ . , data = train)
test_hat_rf <- predict(fit_rf, test)
rmse_acc <- rmse(test$Chance.of.Admit,test_hat_rf)
plot(fit_rf)
print(fit_rf)
summary(fit_rf)
