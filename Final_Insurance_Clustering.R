#R Code for clustering
library(readr)
library(dplyr)
library(fastDummies)
library(ClusterR)
library(cluster)
library(factoextra)




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
ic$Established_years<-2019-ic$Hospital_Since


#Converting into factor and removing the Primary Key column ID, Variable - Hospital Since
ic<-subset(ic,select=-c(ID,Hospital_Since))
ic$Hospital_Rating<-as.factor(ic$Hospital_Rating)
ic$Age_Bracket<-as.factor(ic$Age_Bracket)
ic$Items_Not_Covered<-as.factor(ic$Items_Not_Covered)
ic$Admitted_Under<-as.factor(ic$Admitted_Under)
ic$Insurance_Type<-as.factor(ic$Insurance_Type)

#After Basic Cleaning - removing Target Variable
ic_new<-subset(ic,select=-c(Amount_WithDrawn))

plot_str(ic_new)
#Preprocessing - Scaling
ic_new<-dummy_cols(ic_new,select_columns = c("Insurance_Type" ,"Illness_Type","Hospital_Location","Hostpital_Type","Hospital_Brand","Items_Not_Covered","Treatment_Type","Hospital_Rating","Admitted_Under","Age_Bracket"),remove_first_dummy = TRUE)
ic_new<-subset(ic_new,select= -c(Insurance_Type,Illness_Type,Hospital_Location,Hostpital_Type,Hospital_Brand,Items_Not_Covered,Treatment_Type,Hospital_Rating,Admitted_Under,Age_Bracket))
ic_base<-ic_new

#scaling of all variables in dataset
index<-names(ic_new)
temp<-scale(ic_new[,index])
temp1<-as.data.frame(temp)
ic_new[,index]<-temp1

#Reducing the Dimensions of the dataset - Principal Component Analysis
ic_new_pca<-prcomp(ic_new)
summary(ic_new_pca)

#Screeplot - PCA
screeplot(ic_new_pca, type = "l", npcs = 39, main = "Screeplot of the all 39 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)

ic_modified<-ic_new_pca$x
modified_pca_subset<-as.data.frame(ic_modified[,1:17])


#Clustering using ClusterR package - 2 Principal Components

ic_modified2<-as.data.frame(ic_modified[,1:2])
km<-KMeans_rcpp(ic_modified2,5,num_init = 20,max_iters = 100,initializer = "kmeans++",verbose = TRUE)
Optimal_Clusters_KMeans(ic_modified2,max_clusters = 15,criterion = "WCSSE",plot_clusters = TRUE)
km$centroids

p1<-plot_2d(ic_modified2,km$clusters,km$centroids)






#Running Kmeans from base package and identifying optimum clusters
wss <- function(k) {
  kmeans(modified_pca_subset, k, nstart = 20,iter.max=100,algorithm = "MacQueen" ,trace = TRUE)$tot.withinss
}

k.values <- 1:30
wss_values <- map_dbl(k.values, wss)

#Drwaing a cluster of 10

k2<-kmeans(modified_pca_subset, 10, nstart = 20,iter.max=100,algorithm = "MacQueen")
p2<-fviz_cluster(k2, data = modified_pca_subset)

#Plotting the clusters - Elbow Method
plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")





