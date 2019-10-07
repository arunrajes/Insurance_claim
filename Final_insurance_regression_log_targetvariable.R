#R Code for linear regression log transformation of Y Parameter & Y + Continuous Parameters
#R Code for Ridge and lasso Regression on Y Parameter linear Regression
library(olsrr)
library(readr)
library(DataExplorer)
library(skimr)
library(dplyr)
library(fastDummies)
library(caret)
library(leaps)
library(tidyr)
library(DMwR)

#Reading csv file

ic<-read_csv("C:/Users/gssaruba/Downloads/Insurance_Claim_Prediction.csv")

#Re-arrange column names in the given dataset
names(ic)<-c("ID","Number_of_Months_as_Customer","Insurance_Type","Doctor_Visited_Before_Days",
             "Premium_Amount","Illness_Type","Hospital_Location","Hostpital_Type","Hospital_Brand",
             "No_Days_Stay_Recommended","Items_Not_Covered","Average_Room_Rent","No_Times_Visited_Doctor_in_Last_Month",
             "Treatment_Type","Hospital_Rating","Admitted_Under","Hospital_Since","Age_Bracket","Amount_WithDrawn")

#Initial datatype structure and summary
plot_str(ic)
skim(ic)


#convert the datatype of required columns into factor
col_names <-sapply(ic, function(col) length(unique(col)) < 5)
ic[ , col_names] <- lapply(ic[ , col_names] , factor)
plot_str(ic)

#Establish new feature - Years of Establishment
ic$Hospital_Since<-as.numeric(ic$Hospital_Since)
ic$Established_years<-as.numeric(2019-ic$Hospital_Since)
ic$Admitted_Under<-as.factor(ic$Admitted_Under)
ic$Insurance_Type<-as.factor(ic$Insurance_Type)
ic$Items_Not_Covered<-as.factor(ic$Items_Not_Covered)

#Converting into Ordered factor and removing the Primary Key column ID, Variable - Hospital Since
ic<-subset(ic,select=-c(ID,Hospital_Since))
ic$Hospital_Rating<-ordered(ic$Hospital_Rating,levels=c(2,3,4,5))
ic$Age_Bracket<-ordered(ic$Age_Bracket,levels=c("20-30","30-40","40-50","50+"))

#After Basic Cleaning - Data type structure
plot_str(ic)
skim(ic)
summary(ic)

ic<-ic[,c(17,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18)]
plot_str(ic)

ic<-ic[,c(1,2,4,5,10,12,13,18,3,6,7,8,9,11,14,15,16,17)]
plot_str(ic)

#Establishing 5Fold Cross Validation

custom<-trainControl(method="repeatedcv",
                     number=5,
                     repeats=3,
                     verboseIter = T)


#Preprocessing - Scaling
ic<-dummy_cols(ic,select_columns = c("Insurance_Type" ,"Illness_Type","Hospital_Location","Hostpital_Type","Hospital_Brand","Items_Not_Covered","Treatment_Type","Hospital_Rating","Admitted_Under","Age_Bracket"),remove_first_dummy = TRUE)
ic<-subset(ic,select= -c(Insurance_Type,Illness_Type,Hospital_Location,Hostpital_Type,Hospital_Brand,Items_Not_Covered,Treatment_Type,Hospital_Rating,Admitted_Under,Age_Bracket))
names(ic)<-c("Amount_WithDrawn","Number_of_Months_as_Customer",         
             "Doctor_Visited_Before_Days","Premium_Amount",                       
             "No_Days_Stay_Recommended","Average_Room_Rent",                    
             "No_Times_Visited_Doctor_in_Last_Month","Established_years",                    
             "Insurance_Type_POL02","Insurance_Type_POL03",                 
             "Insurance_Type_POL04","Insurance_Type_POL05",                 
             "Illness_Type_Acute","Hospital_Location_Town",              
             "Hospital_Location_Semi_Urban","Hospital_Location_Bigger_Towns",       
             "Hostpital_Type_Single_Specialty","Hostpital_Type_Nursing_Home",          
             "Hostpital_Type_Childrens_Hospital","Hospital_Brand_Corporate_Hospital",    
             "Hospital_Brand_Legacy_Hospital","Items_Not_Covered_Physiotherapy",      
             "Items_Not_Covered_Extra_Attender","Items_Not_Covered_Catering",           
             "Items_Not_Covered_Radiology","Items_Not_Covered_Pharmacy",          
             "Treatment_Type_Sugery","Treatment_Type_Observation",           
             "Treatment_Type_Trauma_Care","Hospital_Rating_3",                    
             "Hospital_Rating_5","Hospital_Rating_2",                    
             "Admitted_Under_Pulmonology","Admitted_Under_Neurology",             
             "Admitted_Under_Oncology","Admitted_Under_Ortho_Care",            
             "Admitted_Under_Endocrinology","Age_Bracket_20_to_30",                    
             "Age_Bracket_50_plus","Age_Bracket_40_to_50")
ic_base<-ic

#------------------Improving Base Model-2 with Log of Y parameter--------------------------------
y_amt_wd<-log(ic_base$Amount_WithDrawn)
ic_new<-ic_base[,2:40]


#scaling of Numeric variables
index<-names(ic_new)
temp2<-scale(ic_new[,index])
temp3<-as.data.frame(temp2)
ic_new[,index]<-temp3

names(ic_new)



ic2<-cbind(y_amt_wd,ic_new)
names(ic2)[1]<-"Amount_WithDrawn"
plot_histogram(ic2)


#Split of training and test dataset & Custom Control Parameter
set.seed(222)
ind<-sample(2,nrow(ic2),replace=T,prob=c(0.7,0.3))
ic_train<-ic2[ind==1,]
ic_test<-ic2[ind==2,]

#--------------------------Base log Model-2 with no cross validation----------------------------------------------------------
set.seed(223)
model1<-lm(Amount_WithDrawn~.,ic_train)
summary(model1)


#--------------------------Building Base log Model-2 with cross validation-----------------------------------------------------
set.seed(224)
model2<-train(Amount_WithDrawn~.,ic_train,method="lm",trControl=custom)
model2$results
summary(model2)
plot(varImp(model2))

#-------------------------Predicting the Base log Model-2-------------------------------------------------------

model2_predict<-predict(model2,ic_test)
model2_predict
modelvalues<-data.frame(obs=ic_test$Amount_WithDrawn,pred=model2_predict)
defaultSummary(modelvalues)
regr.eval(ic_test$Amount_WithDrawn,model2_predict)

#*************************CSV file output*****************************************************************************
model2_predict<-predict(model2,ic_test)
y2<-model2_predict
y2<-exp(y2)
y2<-as.data.frame(y2)
colnames(y2)<-"Predicted Y"
y_amt<-as.data.frame(exp(ic_test$Amount_WithDrawn))
ic_test2<-as.data.frame(unscale(ic_test[,2:40],temp2))
colnames(y_amt)<-"Amount_WithDrawn"
write.csv(cbind(y2,y_amt,ic_test2),file ="linear_model_logtransform.csv")

#--------------------------Building log Model of all continuous variables-----------------------------------

#applying log transformation to all continuous variable
i1<-ic_base[,1:8]
plot_histogram(i1)
i2<-sapply(i1,log1p)
plot_histogram(i2)

#applying scaling to all categorical variable
i3<-ic_base[,9:40]
str(i3)
index<-names(i3)
temp2<-scale(i3[,index])
temp3<-as.data.frame(temp2)
i3[,index]<-temp3
sapply(i3,mean)
sapply(i3,sd)

#Merging into single data frame
i4<-as.data.frame(cbind(i2,i3))
plot_str(i4)



#Split of training and test dataset & Custom Control Parameter
set.seed(225)
ind<-sample(2,nrow(i4),replace=T,prob=c(0.7,0.3))
ic_train<-i4[ind==1,]
ic_test<-i4[ind==2,]

#Train the model
set.seed(224)
model3<-train(Amount_WithDrawn~.,ic_train,method="lm",trControl=custom)
model3$results
summary(model3)
plot(varImp(model3))

#Predict the model
model3_predict<-predict(model3,ic_test)
model3_predict
modelvalues<-data.frame(obs=ic_test$Amount_WithDrawn,pred=model3_predict)
defaultSummary(modelvalues)
regr.eval(ic_test$Amount_WithDrawn,model3_predict)

#----------------------------Lasso Regression-----------------------------------------------------------------
#Train the model using lasso Regression
set.seed(228)
model4<-train(Amount_WithDrawn~.,ic_train,method="lasso",trControl=custom)
model4$results
summary(model4)
plot(varImp(model4))
model4$bestTune

#Predict the model using lasso model
model4_predict<-predict(model4,ic_test)
model4_predict
modelvalues<-data.frame(obs=ic_test$Amount_WithDrawn,pred=model4_predict)
defaultSummary(modelvalues)
regr.eval(ic_test$Amount_WithDrawn,model4_predict)

#---------------------------Ridge Regression------------------------------------------------------------------
#Train the model using ridge Regression
set.seed(230)
model5<-train(Amount_WithDrawn~.,ic_train,method="ridge",trControl=custom)
model5$results
summary(model5)
model5$bestTune
plot(varImp(model5))

#Predict the model using ridge regression model
model5_predict<-predict(model5,ic_test)
model5_predict
modelvalues<-data.frame(obs=ic_test$Amount_WithDrawn,pred=model5_predict)
defaultSummary(modelvalues)
regr.eval(ic_test$Amount_WithDrawn,model5_predict)

#----------------------------elastic Net Regression----------------------------------------------
#Train the model using elastic Net Regression
set.seed(230)
model6<-train(Amount_WithDrawn~.,ic_train,method="enet",trControl=custom)
model6$results
model6$bestTune
summary(model6)
plot(varImp(model6))

#Predict the model using elastic Net regression model
model6_predict<-predict(model6,ic_test)
model6_predict
modelvalues<-data.frame(obs=ic_test$Amount_WithDrawn,pred=model6_predict)
defaultSummary(modelvalues)
regr.eval(ic_test$Amount_WithDrawn,model4_predict)



