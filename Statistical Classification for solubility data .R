
library(caret)
library(AppliedPredictiveModeling)
library(Hmisc)

#Solubility data by 
#Tetko et al. (2001) and Huuskonen (2000) 
#Consists if a set of compounds with corresponding experimental solubility values using complex sets of descriptors. 
#Authors used linear regression and neural network models to estimate the relationship between chemical structure and solubility.
data(solubility)
ls()

#a concise statistical description of a sample of predictors
ps = sample(length(solTrainX),16)
describe(solTrainX[,ps])

featurePlot(x = solTrainX[,ps], y = solTrainY,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth")) 

controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)

featurePlot(x = solTrainX[,ps], y = solTrainY,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth")) 



verbose = F 
trainingData = solTrainX
trainingData$Solubility = solTrainY

testData = solTestX
#testData$Solubility = solTestY

trainingData.trans = solTrainXtrans
trainingData.trans$Solubility = solTrainY

testData.trans = solTestXtrans

controlObject <- trainControl(method = "repeatedcv", repeats = 5, number = 10)

set.seed(669); ptm <- proc.time()
linearReg <- train(Solubility ~  . , data = trainingData, method = "lm", trControl = controlObject) 


set.seed(669); ptm <- proc.time()
enetGrid <- expand.grid(.lambda = c(0, .001, .01, .1), .fraction = seq(0.05, 1, length = 20))
enetModel <- train(Solubility ~ . , data = trainingData , method = "enet", preProc = c("center", "scale"), tuneGrid = enetGrid, trControl = controlObject)

set.seed(669); ptm <- proc.time()
plsModel <- train(Solubility ~ . , data = trainingData , method = "pls", preProc = c("center", "scale"), tuneLength = 15, trControl = controlObject)

set.seed(669); ptm <- proc.time()
svmRModel <- train(Solubility ~ . , data = trainingData, method = "svmRadial",
                   tuneLength = 15, preProc = c("center", "scale"),  trControl = controlObject)

set.seed(669); ptm <- proc.time()
treebagModel <- train(Solubility ~ . , data = trainingData, method = "treebag", trControl = controlObject)

set.seed(669); ptm <- proc.time()
ctreeModel <- train(Solubility ~ . , data = trainingData, method = "ctree", tuneLength = 10, trControl = controlObject)

set.seed(669); ptm <- proc.time()
rpartModel <- train(Solubility ~ . , data = trainingData , method = "rpart", tuneLength = 30, trControl = controlObject)

##plot performance
allResamples <- resamples(list("Linear Reg" = linearReg, "Linear Reg (Trans)" = linearReg.trans, 
                               "SVM" = svmRModel , "SVM (Trans)" = svmRModel.trans , 
                               "PLS" = plsModel , "PLS (Trans)" = plsModel.trans , 
                               "Elastic Net" = enetModel , "Elastic Net (Trans)" = enetModel.trans , 
                               "Bagged Tree" = treebagModel , "Bagged Tree (Trans)" = treebagModel.trans , 
                               "Cond Inf Tree" = ctreeModel , "Cond Inf Tree (Trans)" = ctreeModel.trans , 
                               "CART" = rpartModel , "CART (Trans)" = rpartModel.trans
))
parallelplot(allResamples)

parallelplot(allResamples , metric = "Rsquared")

perf.grid[order(perf.grid$RMSE.test, decreasing=F),]


##SVM plots 
predicted.SVM.trans = predict(svmRModel.trans , solTestXtrans) 
residualValues.SVM <- solTestY - predicted.SVM.trans
summary(residualValues.SVM)

sd(residualValues.SVM)

axisRange <- extendrange(c(solTestY, predicted.SVM.trans))
plot(solTestY, predicted.SVM.trans, ylim = axisRange, xlim = axisRange)
abline(0, 1, col = "darkgrey", lty = 2)

# Predicted values versus residuals
plot(predicted.SVM.trans, residualValues.SVM, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

##LM plots
predicted.lin_reg.trans = predict(linearReg.trans, solTestXtrans) 
residualValues.reg_lin <- solTestY - predicted.lin_reg.trans
summary(residualValues.reg_lin)

sd(residualValues.reg_lin)

axisRange <- extendrange(c(solTestY, predicted.lin_reg.trans))
plot(solTestY, predicted.lin_reg.trans, ylim = axisRange, xlim = axisRange)
abline(0, 1, col = "darkgrey", lty = 2)

plot(predicted.lin_reg.trans, residualValues.reg_lin, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)