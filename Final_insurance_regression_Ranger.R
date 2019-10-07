#R Code for Ranger- RF Model with Scaling, and without Scaling, 5 Fold Cross validated 
library(readr)
library(DataExplorer)
library(skimr)
library(dplyr)
library(fastDummies)
library(caret)
library(DMwR)
library(ranger)



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

#Rearranging Columns 
ic<-ic[,c(17,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18)]
ic<-ic[,c(1,2,4,5,10,12,13,18,3,6,7,8,9,11,14,15,16,17)]


ic_raw<-ic



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

#------------------Build RF Model with Scaling of X and Y parameter----------------------------------------------------
ic_new<-ic_base[,1:40]


#scaling of all variables
index<-names(ic_new)
temp2<-scale(ic_new[,index])
temp3<-as.data.frame(temp2)
ic_new[,index]<-temp3

names(ic_new)
ic2<-ic_new
plot_histogram(ic2)


#Split of training and test dataset & Custom Control Parameter
set.seed(222)
ind<-sample(2,nrow(ic2),replace=T,prob=c(0.7,0.3))
ic_train<-ic2[ind==1,]
ic_test<-ic2[ind==2,]
rf_sim<-ranger(Amount_WithDrawn~.,data=ic_train,importance="permutation",splitrule = "variance")
print(rf_sim)
importance(rf_sim)



rf_predict<-predict(rf_sim,data=ic_test)
rf_predict$predictions
modelvalues<-data.frame(obs=ic_test$Amount_WithDrawn,pred=rf_predict$predictions)
defaultSummary(modelvalues)
regr.eval(ic_test$Amount_WithDrawn,rf_predict$predictions)

#------------Build the model using Caret Package - Cross Validation & No Scaling--------------------------------
trainIndex<-createDataPartition(ic_raw$Amount_WithDrawn,p=0.7,list=FALSE)

ic_raw_train<-ic_raw[trainIndex,]
ic_raw_test<-ic_raw[-trainIndex,]

# Not to run as of Now: it takes 14 hours to 16 hours to run the model 
#Establishing 3Fold Cross Validation
# 
# custom<-trainControl(method="repeatedcv",
#                       number=3,
#                       repeats=2,
#                       allowParallel = TRUE,
#                       verboseIter = TRUE)
#  hyper_grid<-expand.grid(
#    
#    mtry=seq(9,39,by=10),
#    min.node.size = seq(3,9,by=2),
#    splitrule=c("variance")
#  )
#  nrow(hyper_grid)
# model_rf<-train(Amount_WithDrawn~.,data = ic_raw,method="ranger",tuneGrid=hyper_grid,trControl=custom)
# model_rf
# 
# str(model_rf)
# plot(model_rf)
# model_rf$finalModel
#                 
# 
# Running model with ic_raw_train data with tuned hyperparameters
# 
# 
# hyper_grid<-expand.grid(
#   
#   mtry=seq(18,22,by=1),
#   min.node.size = 9,
#   splitrule=c("variance")
# )
# 
# 
# model_rf_final<-train(Amount_WithDrawn~.,data = ic_raw_train,method="ranger",tuneGrid=hyper_grid)
# 
# model_rf_final
# model_rf_final$resample
# str(model_rf_final)
# plot(model_rf_final)
# model_rf_final$finalModel
# model_rf_predict<-predict(model_rf_final,data=ic_test)

#------------------------Build the model using Ranger Package - No Scaling, No CrossValidation-----------------

model_rf_sim<-ranger(log(Amount_WithDrawn)~.,data=ic_raw_train,mtry=17,importance='impurity',min.node.size=9,splitrule = "variance",verbose = TRUE)
model_rf_sim

ic_raw_test2<-ic_raw_test[,2:18]

model_rf_sim_predict<-predict(model_rf_sim,data=ic_raw_test2)
model_rf_sim_predict$predictions

saveRDS(model_rf_sim,file="C:/Users/gssaruba/Documents/R_Scripts/rangermodel.rds")


modelvalues<-data.frame(obs=log(ic_raw_test$Amount_WithDrawn),pred=model_rf_sim_predict$predictions)
defaultSummary(modelvalues)
regr.eval(log(ic_raw_test$Amount_WithDrawn),model_rf_sim_predict$predictions)

Predicted<-exp(model_rf_sim_predict$predictions)
Ranger_output<-cbind(ic_raw_test,Predicted)
write.csv(Ranger_output,"Insurance_ranger_output.csv")

library(vip)
vip(model_rf_sim)
#****************************************************************************************************************
#-------------------------------Model Intrepretation with LIME--------------------------------
#***************************************************************************************************************
library(lime)

model_type.ranger<-function(x,...){
  return('regression')
}

model_rf_sim_lime<-ranger(Amount_WithDrawn~.,data=ic_raw_train,mtry=17,importance='impurity',min.node.size=9,splitrule = "variance",verbose = TRUE)
model_type(model_rf_sim_lime)

local_obs<-ic_raw_train[1:4,2:18]

predict_model.ranger <- function(x, newdata, ...) {
  # Function performs prediction and returns data frame with Response
  pred <- predict(x, newdata)
  return(as.data.frame(pred$predictions))
}

predict_model(model_rf_sim_lime, newdata = local_obs)

explainer_ranger<-lime(local_obs,model_rf_sim_lime)
explanation_ranger <- explain(local_obs, explainer_ranger,n_features=17,n_permutations = 1000,dist_fun="euclidean")
plot_features(explanation_ranger, ncol = 2)
