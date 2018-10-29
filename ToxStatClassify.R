
library(ggplot2)
library(lattice)
library(caret)
library(C50)
library(kernlab)
library(mlbench)
library(randomForest)
library(caretEnsemble)
library(MASS)
library(klaR)
library(nnet)
library(caretEnsemble)

phpcDat<- read.csv("Griznic_phch.csv", header = TRUE,sep=';')
phpcDat<- data.matrix(phpcDat[1:dim(phpcDat)[1],])
 
#summary statistics
summary(phpcDat)
sum(is.na(phpcDat))

## calculate correlation matrix
correlationMatrix <- cor(phpcDat[1:dim(phpcDat)[1],2:dim(phpcDat)[2]])
heatmap(correlationMatrix)
 
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
highlyCorrelated 

 
##data split
set.seed(1188)
# Stratified sampling
TrainingDataIndex <- createDataPartition(phpcDat[1,], p=0.75, list = FALSE)
# Create Training Data 
trainingData <- phpcDat[,TrainingDataIndex]
testData <- phpcDat[,-TrainingDataIndex]
TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=10)

trainingData<- trainingData[,-1]
testData<- testData[,-1]

row.names(trainingData)<- c('Class',paste('descr',1:5,sep=''))
row.names(testData)<- c('Class',paste('descr',1:5,sep=''))

trainingData<- t(trainingData)
testData<- t(testData)

## Model training
## classification with SVM
SVModel <- train(as.factor(Class) ~ ., data = trainingData,
          method = "svmPoly",
          trControl= TrainingParameters,
          tuneGrid = data.frame(degree = 1,
          scale = 1,C = 1),
          preProcess = c("pca","scale","center"),
          na.action = na.omit)

SVMPredictions <-predict(SVModel,testData)
# Create confusion matrix
cmSVM <-confusionMatrix(SVMPredictions, as.factor(as.data.frame(testData)$Class))
print(cmSVM)


#importance <- varImp(SVModel, scale=FALSE)
#vplot(importance)

## classification with DT
DecTreeModel <- train(as.factor(Class) ~ ., data = trainingData, 
                      method = "C5.0",
                      preProcess=c("scale","center"),
                      trControl= TrainingParameters,
                      na.action = na.omit
)

#Predictions
DTPredictions <-predict(DecTreeModel, testData, na.action = na.pass)
# Print confusion matrix and results
cmTree <-confusionMatrix(DTPredictions, as.factor(as.data.frame(testData)$Class))
print(cmTree)

## classification with Naive B
#Naive algorithm
NaiveModel <- train(trainingData[,-1], as.factor(as.data.frame(trainingData)$Class), 
                    method = "nb",
                    preProcess=c("scale","center"),
                    trControl= TrainingParameters,
                    na.action = na.omit
)

#Predictions
NaivePredictions <-predict(NaiveModel, testData, na.action = na.pass)
cmNaive <-confusionMatrix(NaivePredictions, as.factor(as.data.frame(testData)$Class))


##classification with neural networks
# train model with neural networks
NNModel <- train(trainingData[,-1], as.factor(as.data.frame(trainingData)$Class),
                 method = "nnet",
                 trControl= TrainingParameters,
                 #preProcess=c("scale","center"),
                 na.action = na.omit,
)

NNPredictions <-predict(NNModel, testData)
# Create confusion matrix
cmNN <-confusionMatrix(NNPredictions, as.factor(as.data.frame(testData)$Class))
print(cmNN)



##classification with ensemble models
# Create models
econtrol <- trainControl(method="cv", number=10, savePredictions=TRUE, classProbs=TRUE)
model_list <- caretList(as.factor(Class) ~ ., data = as.data.frame(trainingData),
                        methodList=c("svmPoly", "nnet", "C5.0", "nb"),
                        preProcess=c("scale","center"),
                        trControl = econtrol
)


results <- resamples(model_list)

# Measure model correlation
mcr <-modelCor(results)
print (mcr)


##
