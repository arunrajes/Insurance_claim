library(readr)
library(dplyr)
library(caret)
library(fastDummies)
library(tidyr)
library(DMwR)

#Reading csv file

ic<-read_csv("C:/Users/gssaruba/Downloads/Insurance_Claim_Prediction.csv")

#Re-arrange column names in the given dataset
names(ic)<-c("ID","Number_of_Months_as_Customer","Insurance_Type","Doctor_Visited_Before_Days",
             "Premium_Amount","Illness_Type","Hospital_Location","Hostpital_Type","Hospital_Brand",
             "No_Days_Stay_Recommended","Items_Not_Covered","Average_Room_Rent","No_Times_Visited_Doctor_in_Last_Month",
             "Treatment_Type","Hospital_Rating","Admitted_Under","Hospital_Since","Age_Bracket","Amount_WithDrawn")


#convert the datatype of required columns into factor
col_names <-sapply(ic, function(col) length(unique(col)) < 5)
ic[ , col_names] <- lapply(ic[ , col_names] , factor)

ic$Insurance_Type<-as.factor(ic$Insurance_Type)
ic$Items_Not_Covered<-as.factor(ic$Items_Not_Covered)
ic$Admitted_Under<-as.factor(ic$Admitted_Under)
ic$Hospital_Since<-as.numeric(ic$Hospital_Since)
ic$Established_years<-2019-ic$Hospital_Since
ic$Age_Bracket<-as.factor(ic$Age_Bracket)

#Converting into Ordered factor and removing the Primary Key column ID
ic<-subset(ic,select=-c(ID,Hospital_Since))
ic$Hospital_Rating<-as.factor(ic$Hospital_Rating)
ic$Age_Bracket<-as.factor(ic$Age_Bracket)


names(ic)
ic<-ic[,c(17,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18)]


#Preprocessing - Scaling
ic<-dummy_cols(ic,select_columns = c("Insurance_Type" ,"Illness_Type","Hospital_Location","Hostpital_Type","Hospital_Brand","Items_Not_Covered","Treatment_Type","Hospital_Rating","Admitted_Under","Age_Bracket"),remove_first_dummy = TRUE)
ic<-subset(ic,select= -c(Insurance_Type,Illness_Type,Hospital_Location,Hostpital_Type,Hospital_Brand,Items_Not_Covered,Treatment_Type,Hospital_Rating,Admitted_Under,Age_Bracket))
names(ic)

ylog<-log(ic$Amount_WithDrawn)
y<-as.data.frame(ylog)
colnames(y)<-"Amount_WithDrawn"
head(y)
ic_new<-ic[,2:40]

#scaling of Numeric variables
index<-names(ic_new)
temp<-scale(ic_new[,index])
temp<-as.data.frame(temp)
ic_new[,index]<-temp

which(is.na(ic_new))
ic2<-cbind(y,ic_new)

#Check the mean and std.dev once scaling is done
sapply(ic_new[,1:39],mean)
sapply(ic_new[,1:39],sd)

set.seed(222)
ind<-sample(2,nrow(ic2),replace=T,prob=c(0.7,0.3))
ic_train<-ic2[ind==1,]
ic_test<-ic2[ind==2,]

train_x<-as.matrix(ic_train[,2:40])
train_y<-as.matrix(ic_train$Amount_WithDrawn)

test_x<-as.matrix(ic_test[,2:40])
test_y<-as.matrix(ic_test$Amount_WithDrawn)

library(keras)
model <- keras_model_sequential() %>%
# network architecture
layer_dense(units = 50, activation = "relu", input_shape = ncol(train_x)) %>%
layer_dense(units=20,activation="relu")%>%
layer_dense(units = 5, activation = "relu")
summary(model)
  
# backpropagation
model%>%compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mse")
  )

# model%>%compile(
#   optimizer = "adagrad",
#   loss = "RMSE",
#   metrics = c("RMSE")
# )

model%>%summary()
# train our model

learn <- model %>% fit(
  x = train_x,
  y = train_y,
  epochs = 15,
  batch_size = 32,
  validation_split = 0.2
)
plot(learn)

summary(learn)

test_predictions<-model%>% predict(test_x)
test_predictions[,1]

modelvalues<-data.frame(obs=test_y,pred=test_predictions[,1])
defaultSummary(modelvalues)
regr.eval(test_y,test_predictions[,1])


