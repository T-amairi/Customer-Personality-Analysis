########################################
#            Preparing Data            #
########################################

### Retrieving the dataset ###
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #set path
data = read.csv2("marketing_campaign.csv", sep= "\t") #load data
head(data) #check data 

### Cleaning ###
library(dplyr)
print(sum(is.na(data))) #check for missing values
data = na.omit(data) #drop missing values because few NA items
data = filter(data,data$Income < 15e+04) #filter huge income
str(data)

### Setting variables ###
str(data) #check data type

data$ID = NULL #ID is useless
print(unique(data$Z_CostContact)) #contains only one value 
print(unique(data$Z_Revenue)) #contains only one value
data$Z_CostContact = NULL #useless
data$Z_Revenue = NULL #useless

#we kill keep response instead of all AcceptedCmp* columns
data$AcceptedCmp1 = NULL
data$AcceptedCmp2 = NULL
data$AcceptedCmp3 = NULL
data$AcceptedCmp4 = NULL
data$AcceptedCmp5 = NULL

#get values from marital status and education
print(unique(data$Marital_Status))
print(unique(data$Education))

#check boxplot of marital status
barplot(table(data$Marital_Status))

#remove rows with YOLO and Absurd as marital status (few samples)
data = data[data$Marital_Status != "YOLO",]
data = data[data$Marital_Status != "Absurd",]

#to many Education lvl : we will summarize them in two categories :
#PG = postgraduate and UG = undergraduate
data$Education[data$Education == "2n Cycle"] = "UG"
data$Education[data$Education == "Basic"] = "UG"
data$Education[data$Education == "Graduation"] = "PG"
data$Education[data$Education == "Master"] = "PG"
data$Education[data$Education == "PhD"] = "PG"

#same thing for Marital_Status : single or couple
data$Marital_Status[data$Marital_Status == "Divorced"] = "Single"
data$Marital_Status[data$Marital_Status == "Widow"] = "Single"
data$Marital_Status[data$Marital_Status == "Together"] = "Couple"
data$Marital_Status[data$Marital_Status == "Married"] = "Couple"
data$Marital_Status[data$Marital_Status == "Alone"] = "Single"

#merge kid and teen variables into one column
data$Child = data$Kidhome + data$Teenhome
data$Kidhome = NULL
data$Teenhome = NULL

#change date of birth into age
data$Age = 2022 - data$Year_Birth
data$Year_Birth = NULL

#change the date of customer's enrollment into seniority
data$Dt_Customer = substr(data$Dt_Customer,7,10)
data$Seniority  = 2022 - as.numeric(data$Dt_Customer)
data$Dt_Customer = NULL

#split products in two categories : necessary and optional products
data$MntOptional = data$MntWines + data$MntGoldProds + data$MntSweetProducts
data$MntNecessary = data$MntFishProducts + data$MntFruits + data$MntMeatProducts
data$MntWines = NULL
data$MntFishProducts = NULL
data$MntFruits = NULL
data$MntGoldProds = NULL
data$MntMeatProducts = NULL
data$MntSweetProducts = NULL

#merge all the purchases
data$TotalPurchases = data$NumWebPurchases + data$NumCatalogPurchases + data$NumStorePurchases
data$NumWebPurchases = NULL
data$NumCatalogPurchases = NULL
data$NumStorePurchases = NULL

isFactor <- function(x) #to set a column as factor 
{
  if(length(unique(x)) == 2)
  {
    as.factor(x)
  }
    
  else x
}

#set factors
data[] = lapply(data,isFactor) 

#order by column names
data = data[,order(names(data))]
attach(data)

#final check
str(data)

########################################
#      UNSUPERVISED CLASSIFICATION     #
########################################

library(FactoMineR)
library(factoextra)
options(ggrepel.max.overlaps = Inf)

#get the sub dataframe for the FAMD
varFAMD = names(data) %in% c("Response", "Complain")
dataFAMD = data[!varFAMD]

#FAMD
famd = FAMD(dataFAMD,graph = FALSE)

#plot
fviz_eig(famd,addlabels = TRUE) #% of variance for each dim
gradients = c("#00AFBB","#E7B800","#FC4E07")
fviz_famd_var(famd,"quanti.var",col.var = "cos2",gradient.cols = gradients,repel = TRUE) #quanti cos2
fviz_famd_var(famd,"quali.var",col.var = "cos2",gradient.cols = gradients,repel = TRUE) #quali cos2
fviz_famd_ind(famd, col.ind = "cos2",gradient.cols = gradients,labels=FALSE) #indiv cos2

#HCPC
hcpc = HCPC(famd,consol = TRUE,iter.max = 1000,nb.clust = -1,graph = FALSE)

#plot
plot(hcpc,choice = "tree",labels=FALSE)
plot(hcpc,choice = "map",ind.names = FALSE,draw.tree = FALSE)

#get clusters spec
print(hcpc$desc.var$quanti)
print(hcpc$desc.var$category)

########################################
#      SUPERVISED CLASSIFICATION       #
########################################

#get dataset for training and testing
set.seed(1)
n <- nrow(data)
p <- ncol(data)-1
test.ratio <- .2 # ratio of test/train samples
n.test <- round(n*test.ratio)
tr <- sample(1:n,n.test)
data.test <- data[tr,]
data.train <- data[-tr,]

#check the distribution of response
print(table(data.train$Response)) #=> dataset imbalanced

#balance the data.train set
library(DMwR)
data.train = SMOTE(Response ~., data.train)

#AFD / LDA / QDA
print(table(data.train$Response)) #prior prob not equal => can't use AFD

#shapiro test
print(shapiro.test(data.train$Income))
#log transformation ?
print(shapiro.test(log(data.train$Income))) #can't use LDA nor QDA

#create data frame with dummies for categorial variables
matrix.train.dum = model.matrix(Response ~., data.train)[,-1]
matrix.test.dum = model.matrix(Response ~., data.test)[,-1]
data.train.dum = data.frame(matrix.train.dum)
data.test.dum = data.frame(matrix.test.dum)
data.train.dum$Response = data.train$Response
data.test.dum$Response = data.test$Response

#lda/qda
library(MASS)
LDA = lda(Response~.,data=data.train.dum)
QDA = qda(Response~.,data=data.train.dum)

#predict
predict.LDA = predict(LDA,newdata=data.test.dum)$class
predict.QDA = predict(QDA,newdata=data.test.dum)$class

#get acc
LDA.acc = mean(predict.LDA  == data.test.dum$Response)
QDA.acc = mean(predict.QDA == data.test.dum$Response)

#get auc
library(pROC)
predict.LDA = predict(LDA,newdata=data.test.dum)$posterior[,2]
predict.QDA = predict(QDA,newdata=data.test.dum)$posterior[,2]
LDA.roc = roc(data.test.dum$Response,predict.LDA)
QDA.roc = roc(data.test.dum$Response,predict.QDA)

#KNN
library(class)
#scale
data.train.dum.scale = data.frame(scale(matrix.train.dum,center=TRUE,scale=TRUE))
data.test.dum.scale = data.frame(scale(matrix.test.dum,center=TRUE,scale=TRUE))

i=1
knn.tmp=1
l = round(sqrt(nrow(data.train.dum.scale))) + 1
for(i in 1:l) #test multiple k value to get the best accuracy
{
  knn.i = knn(train=data.train.dum.scale,test=data.test.dum.scale,cl=data.train$Response,k=i)
  knn.tmp[i] = sum(data.test$Response == knn.i)/nrow(data.test)
}

#get acc
knn.acc = max(knn.tmp)

#get best k value
knn.k = which(knn.tmp == knn.acc)[1]

#get auc
knn.opt = knn(train=data.train.dum.scale,test=data.test.dum.scale,cl=data.train$Response,k=knn.k,prob=TRUE)
knn.roc = roc(data.test.dum$Response,attributes(knn.opt)$prob)

#plot k-value
plot(knn.tmp,type="b",xlab="k-value",ylab="Accuracy")

#CART
library(rpart)
library(rpart.plot)
cart = rpart(Response~.,data.train,control=rpart.control(cp=0))

#get best cp
cp.opt = cart$cptable[which.min(cart$cptable[,"xerror"]),"CP"]

#get best tree
cart.opt = prune(cart,cp.opt)

#save plot of the tress as a jpeg
tiff("cart.jpeg", units="in", width=5, height=5, res=600)
rpart.plot(cart.opt, type=4)
dev.off()

#get importance
barplot(cart.opt$variable.importance, las=3)

#predict
predict.cart = predict(cart.opt, newdata=data.test, type="class")

#get acc
cart.acc = mean(predict.cart == data.test$Response)

#get auc
predict.cart = predict(cart.opt, data.test, type="prob")[,2]
cart.roc = roc(data.test$Response,predict.cart)

#RANDOM FOREST
library(randomForest)
RF = randomForest(Response~.,data.train)

#get importance
ord=order(RF$importance,decreasing = TRUE)
barplot(RF$importance[ord],names.arg=rownames(RF$importance)[ord],las=3)

#predict
predict.RF = predict(RF, newdata=data.test, type="class")

#get acc
RF.acc = mean(predict.RF == data.test$Response)

#get auc
predict.RF = predict(RF, data.test, type="prob")[,2]
RF.roc = roc(data.test$Response,predict.RF)

#LASSO
library(glmnet)
#v-fold because we have a lot of rows
lasso.cv = cv.glmnet(matrix.train.dum,data.train$Response,family="binomial",type.measure = "class")

#predict
predict.lasso = predict(lasso.cv,newx = matrix.test.dum,s = 'lambda.min',type = "class")

#get acc
lasso.cv.acc = mean(predict.lasso == data.test$Response)

#get auc
predict.lasso = predict(lasso.cv,newx = matrix.test.dum,s = 'lambda.min',type = "response")
lasso.cv.roc = roc(data.test$Response,predict.lasso)

#get odd ratio
opt.lamb = lasso.cv$lambda.min
print(exp(coef(lasso.cv,s = opt.lamb)))

#Comparaison
result=matrix(NA, ncol=6, nrow=2)
rownames(result)=c('accuracy', 'AUC')
colnames(result)=c('LDA','QDA','KNN','CART','RF','LASSO')
result[1,]= c(LDA.acc,QDA.acc,knn.acc,cart.acc,RF.acc,lasso.cv.acc)
result[2,]=c(LDA.roc$auc,QDA.roc$auc,knn.roc$auc,cart.roc$auc,RF.roc$auc,lasso.cv.roc$auc)
print(result)

#plot
plot(LDA.roc, xlim=c(1,0))
plot(QDA.roc, add=TRUE, col=2)
plot(knn.roc, add=TRUE, col=3)
plot(cart.roc, add=TRUE, col=4)
plot(RF.roc,add=TRUE, col=5)
plot(lasso.cv.roc, add=TRUE, col=6)
legend('bottomright', col=1:6, paste(c('LDA','QDA', 'KNN','CART','RF','LASSO')), lwd=1)
