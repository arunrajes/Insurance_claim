#R Code for Naive Bayes Classifier
library(readr)
library(dplyr)
library(caret)
library(naivebayes)



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

#Establish new feature - Years of Establishment
ic$Hospital_Since<-as.numeric(ic$Hospital_Since)
ic$Established_years<-as.numeric(2019-ic$Hospital_Since)

#Converting few columns into factor columns
ic$Admitted_Under<-as.factor(ic$Admitted_Under)
ic$Insurance_Type<-as.factor(ic$Insurance_Type)
ic$Items_Not_Covered<-as.factor(ic$Items_Not_Covered)
ic$Hospital_Rating<-as.factor(ic$Hospital_Rating)
ic$Age_Bracket<-as.factor(ic$Age_Bracket)

#Removing ID and Hospital Since column
ic<-subset(ic,select=-c(ID,Hospital_Since))

#Establishing 3-Fold Cross Validation with Hyperparameter Tune Grid


custom<-trainControl(method="repeatedcv",
                     number=3,
                     repeats=2,
                     allowParallel = TRUE,
                     verboseIter = T)

search_grid<-expand.grid (
  laplace=seq(1,5,by=0.5),
  usekernel=TRUE,
  adjust=seq(1,5,by=1)
)

#Converting into classification problem
ic2<-ic%>%mutate(Amount_WithDrawn_Class= ifelse(ic$Amount_WithDrawn>168107,'High','Low'))
table(ic2$Amount_WithDrawn_Class)
ic2<-subset(ic2,select=-c(Amount_WithDrawn))

ic_base<-ic2
ic_base<-as.data.frame(ic_base)

#-------Train the model using Naive Bayes Classifier in Caret Package-------------

#Create Test and Train data split
set.seed(330)
trainIndex<-createDataPartition(ic_base$Amount_WithDrawn_Class,p=0.7,list=FALSE)
ic_base_train<-ic_base[trainIndex,]
ic_base_test<-ic_base[-trainIndex,]

ic_base_train$Amount_WithDrawn_Class<-as.factor(ic_base_train$Amount_WithDrawn_Class)
ic_base_test$Amount_WithDrawn_Class<-as.factor(ic_base_test$Amount_WithDrawn_Class)

y_train<-ic_base_train$Amount_WithDrawn_Class
x_train<-subset(ic_base_train,select=-c(Amount_WithDrawn_Class))

y_test<-ic_base_test$Amount_WithDrawn_Class
x_test<-subset(ic_base_test,select=-c(Amount_WithDrawn_Class))


#Build the model using training data - Hyper Paramter Tuning
nb.ml<-train(x_train,y_train,method="naive_bayes",trControl = custom,tuneGrid = search_grid)
summary(nb.ml)
nb.ml$bestTune
nb.ml$results
nb.ml$results %>%top_n(5, wt = Accuracy)%>%arrange(desc(Accuracy))
plot(nb.ml)
confusionMatrix(nb.ml)

#Test the model using test data
pred<-predict(nb.ml,newdata=x_test)
confusionMatrix(pred,y_test)

