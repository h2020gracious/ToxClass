library(caret)
library(AppliedPredictiveModeling)
# dummy data set to measure performance in Statistical Classification Models

set.seed(975)
training <- quadBoundaryFunc(500)
testing <- quadBoundaryFunc(1000)


set.seed(615)
# Generate a 200x4 data matrix, 2 continuous & 2 categorical variables
dat <- easyBoundaryFunc(200, interaction = 3, intercept = 3)
dat$X1 <- scale(dat$X1)
dat$X2 <- scale(dat$X2)
dat$Data <- "Original"
dat$prob <- NULL

# Feature Selection
# Generate a 500x3 data matrix, 2 continuous & 1 categorical variables
set.seed(874)
reliefEx3 <- easyBoundaryFunc(500)
reliefEx3$X1 <- scale(reliefEx3$X1)
reliefEx3$X2 <- scale(reliefEx3$X2)
reliefEx3$prob <- NULL
apply(reliefEx3,2,range)


