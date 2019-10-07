#R Code for KNN lazyClassifier
library(readr)
library(dplyr)
library(caret)
library(fastDummies)

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
plot_str(ic)

#Establish new feature - Years of Establishment
ic$Hospital_Since<-as.numeric(ic$Hospital_Since)
ic$Established_years<-as.numeric(2019-ic$Hospital_Since)

#Converting few columns to factor columns
ic$Admitted_Under<-as.factor(ic$Admitted_Under)
ic$Insurance_Type<-as.factor(ic$Insurance_Type)
ic$Items_Not_Covered<-as.factor(ic$Items_Not_Covered)
ic$Hospital_Rating<-as.factor(ic$Hospital_Rating)
ic$Age_Bracket<-as.factor(ic$Age_Bracket)

#Primary Key column ID, Variable - Hospital Since
ic<-subset(ic,select=-c(ID,Hospital_Since))


#Converting into classification problem
ic2<-ic%>%mutate(Amount_WithDrawn_Class= ifelse(ic$Amount_WithDrawn>168107,'High','Low'))
table(ic2$Amount_WithDrawn_Class)
ic2<-subset(ic2,select=-c(Amount_WithDrawn))


ic_base<-ic2

#Note: Use the model which is scaled. seems little faster . avg time 10 to 15 minutes

#-------Train the model using KNN3-unscaled data Caret Package-----------------------
set.seed(330)
#Create Test and Train data split
trainIndex<-createDataPartition(ic_base$Amount_WithDrawn_Class,p=0.7,list=FALSE)

ic_base_train<-ic_base[trainIndex,]
ic_base_test<-ic_base[-trainIndex,]


ic_base_train$Amount_WithDrawn_Class<-as.factor(ic_base_train$Amount_WithDrawn_Class)
ic_base_test$Amount_WithDrawn_Class<-as.factor(ic_base_test$Amount_WithDrawn_Class)

knnFit_unsc<-knn3(Amount_WithDrawn_Class ~.,data=ic_base_train,k=11)
knnFit_unsc
tst_pred_unsc<-predict(knnFit_unsc,newdata=ic_base_test,type="class")
confusionMatrix(table(predicted=tst_pred_unsc,actual=ic_base_test$Amount_WithDrawn_Class))



#------Build Model using Scaling of X variables in Caret Package----------------------------------

set.seed(400)
#Create Test and Train data split
trainIndex<-createDataPartition(ic_base$Amount_WithDrawn_Class,p=0.7,list=FALSE)

ic_base_train2<-ic_base[trainIndex,]
ic_base_test2<-ic_base[-trainIndex,]


ic_base_train2$Amount_WithDrawn_Class<-as.factor(ic_base_train2$Amount_WithDrawn_Class)
ic_base_test2$Amount_WithDrawn_Class<-as.factor(ic_base_test2$Amount_WithDrawn_Class)


#Center, Scale for X values
values<-preProcess(ic_base_train2,method=c("center","scale"))
x_train<-predict(values,ic_base_train2)
x_test<-predict(values,ic_base_test2)

#Build the Model

knnFit_sc<-knn3(Amount_WithDrawn_Class~.,data=x_train,k=11)
knnFit_sc
head(knnFit_sc$learn$X)
tst_pred_sc<-predict(knnFit_sc,newdata=x_test,type="class")
tst_pred_sc

confusionMatrix(table(predicted=tst_pred_sc,actual=ic_base_test2$Amount_WithDrawn_Class))
