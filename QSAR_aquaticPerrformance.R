library(caret)
library(recipes)
library(dplyr)
library(QSARdata)

data(AquaticTox)
tox <- AquaticTox_moe2D
ncol(tox)
#These data were compiled and described by He and Jurs (2005). 
#The data set consists of 322 compounds that were experimentally assessed for toxicity. 
#The outcome is the negative log of activity (but is labled as "activity").
#The structures and outcomes were obtained from http://www.qsarworld.com/index.php.


## Add the outcome variable to the data frame
tox$Activity <- AquaticTox_Outcome$Activity

tox <- tox %>%
  select(-Molecule) %>%
  ## Suppose the easy of manufacturability is 
  ## related to the molecular weight of the compound
  mutate(manufacturability  = 1/moe2D_Weight) %>%
  mutate(manufacturability = manufacturability/sum(manufacturability))

model_stats <- function(data, lev = NULL, model = NULL) {
  
  stats <- defaultSummary(data, lev = lev, model = model)
  
  wt_rmse <- function (pred, obs, wts, na.rm = TRUE) 
    sqrt(weighted.mean((pred - obs)^2, wts, na.rm = na.rm))
  
  res <- wt_rmse(pred = data$pred,
                 obs = data$obs, 
                 wts = data$manufacturability)
  c(wRMSE = res, stats)
}

tox_recipe <- recipe(Activity ~ ., data = tox) %>%
  add_role(manufacturability, new_role = "performance var")

#Browse[1]> head(data)

tox_recipe <- tox_recipe %>% step_nzv(all_predictors())
tox_recipe

tox_recipe <- tox_recipe %>% 
  step_pca(contains("VSA"), prefix = "surf_area_",  threshold = .95) 

tox_recipe <- tox_recipe %>% 
  step_center(all_predictors()) %>%
  step_scale(all_predictors())
tox_recipe

tox_ctrl <- trainControl(method = "cv", summaryFunction = model_stats)
set.seed(888)
tox_svm <- train(tox_recipe, tox,
                 method = "svmRadial", 
                 metric = "wRMSE",
                 maximize = FALSE,
                 tuneLength = 10,
                 trControl = tox_ctrl)
tox_svm

predictors(tox_svm)

tox_svm$recipe

