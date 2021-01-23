library(caret)
library(e1071)
library(IRdisplay)
library(knitr)
library(kableExtra)
library(DMwR)

set.seed(8888)

draw_confusion_matrix <- function(cm) {

  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)

  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Negative Class (-1)', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Positive Class (+1)', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Negative Class (-1)', cex=1.2, srt=90)
  text(140, 335, 'Positive Class (+1)', cex=1.2, srt=90)

  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='black')
  text(195, 335, res[2], cex=1.6, font=2, col='black')
  text(295, 400, res[3], cex=1.6, font=2, col='black')
  text(295, 335, res[4], cex=1.6, font=2, col='black')

  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$overall[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$overall[1]), 3), cex=1.2)
  text(30, 85, names(cm$overall[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$overall[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)

  # add in the accuracy information 
  text(30, 35, names(cm$byClass[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$byClass[1]), 3), cex=1.4)
  text(70, 35, names(cm$byClass[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$byClass[2]), 3), cex=1.4)
}  

path <- '../data/hepatitisC.csv'
hepc.backup <- read.csv(path)
hepc <- read.csv(path)

# taking missing columns
missing_cols <- colnames(hepc)[colSums(is.na(hepc)) > 0]
missing_cols

# impute missing value with median for each columns by category
for (col in missing_cols){
    hepc[, col] <- ave(hepc[, col], 
                       hepc$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- hepc$Category != 'suspect Blood Donor'
hepc <- hepc[mask,]

# change the category
mask <- hepc$Category != 'Blood Donor'
hepc[mask, 'general_category'] <- 'Hepatitis'
hepc[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
hepc$Category <- factor(hepc$Category)
hepc$general_category <- factor(hepc$general_category)

summary(hepc$Category)
summary(hepc$general_category)


hepc$general_category <- c(-1, 1)[unclass(as.factor(hepc$general_category))]
hepc$general_category <- as.factor(hepc$general_category)

set.seed(8888)
trainIndex <- createDataPartition(hepc$general_category, p = .7,
                                  list = FALSE,
                                  times = 1)
train <- hepc[trainIndex, ]
valid <- hepc[-trainIndex,]

library(ROSE)

all_features <- c('Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT', 'general_category')
selected_features <- c('AST', 'ALP', 'ALT', 'GGT', 'BIL', 'general_category')

train.1 <- train[, all_features]
train.2 <- train[, selected_features]
train.3 <- ROSE(general_category~., data=train.1, seed=8888)$data
train.4 <- SMOTE(general_category~., data=train.1, seed=8888, perc.over=600, perc.under=100)

table(train.1$general_category)

table(train.2$general_category)

table(train.3$general_category)

table(train.4$general_category)

classifier.all <- svm(formula= general_category ~ ., 
                  data=train.1,
                  type='C-classification',
                  kernel='linear',
                  cross=5
                 )

y_pred <- predict(classifier.all, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier.selected <- svm(formula= general_category ~ ., 
                  data=train.2,
                  type='C-classification',
                  kernel='linear',
                  cross=5
                 )

y_pred <- predict(classifier.selected, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

weights <- 100 / table(train$general_category)

classifier2.1 <- svm(formula= general_category ~ ., 
                   data=train.2,
                   type='C-classification',
                   kernel='linear',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier2.1, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier2.1.cost.1000 <- svm(formula= general_category ~ ., 
                            data=train.2,
                            type='C-classification',
                            kernel='linear',
                            cross=5,
                            class.weights=weights,
                            cost=1000
                           )

y_pred <- predict(classifier2.1.cost.1000, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier2.2 <- svm(formula= general_category ~ ., 
                   data=train.2,
                   type='C-classification',
                   kernel='polynomial',
                   cross=5,
                   degree=2,
                   class.weights=weights
                  )

y_pred <- predict(classifier2.2, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier2.3 <- svm(formula= general_category ~ ., 
                   data=train.2,
                   type='C-classification',
                   kernel='polynomial',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier2.3, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier2.3.cost.1000 <- svm(formula= general_category ~ ., 
                             data=train.2,
                             type='C-classification',
                             kernel='polynomial',
                             cross=5,
                             class.weights=weights,
                             C=1000
                            )

y_pred <- predict(classifier2.3.cost.1000, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier2.4 <- svm(formula= general_category ~ ., 
                   data=train.2,
                   type='C-classification',
                   kernel='sigmoid',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier2.4, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier2.5 <- svm(formula= general_category ~ ., 
                   data=train.2,
                   type='C-classification',
                   kernel='radial',
                   cross=5
                  )

y_pred <- predict(classifier2.5, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

train.3 <- train.3[, selected_features]

classifier3.1.no_weights <- svm(formula= general_category ~ ., 
                   data=train.3,
                   type='C-classification',
                   kernel='linear',
                   cross=5
                  )

y_pred <- predict(classifier3.1.no_weights, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

weights <- 100 / table(train.3$general_category)

weights[1] <- 0.4
weights[2] <- 0.6
weights

classifier3.1 <- svm(formula= general_category ~ ., 
                   data=train.3,
                   type='C-classification',
                   kernel='linear',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier3.1, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier3.2 <- svm(formula= general_category ~ ., 
                   data=train.3,
                   type='C-classification',
                   kernel='polynomial',
                   degree=2,
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier3.2, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier3.3 <- svm(formula= general_category ~ ., 
                   data=train.3,
                   type='C-classification',
                   kernel='polynomial',
                   degree=3,
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier3.3, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier3.4 <- svm(formula= general_category ~ ., 
                   data=train.3,
                   type='C-classification',
                   kernel='sigmoid',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier3.4, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier3.5 <- svm(formula= general_category ~ ., 
                   data=train.3,
                   type='C-classification',
                   kernel='radial',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier3.5, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier4.1.no_weights <- svm(formula= general_category ~ ., 
                   data=train.4,
                   type='C-classification',
                   kernel='linear',
                   cross=5
                  )

y_pred <- predict(classifier4.1.no_weights, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

weights <- 100 / table(train.4$general_category)

# weights[1] <- 0.4
# weights[2] <- 0.6
weights

classifier4.1 <- svm(formula= general_category ~ ., 
                   data=train.4,
                   type='C-classification',
                   kernel='linear',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier4.1, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier4.2 <- svm(formula= general_category ~ ., 
                   data=train.4,
                   type='C-classification',
                   kernel='polynomial',
                   degree=2,
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier4.2, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier4.3 <- svm(formula= general_category ~ ., 
                   data=train.4,
                   type='C-classification',
                   kernel='polynomial',
                   degree=4,
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier4.3, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier4.4 <- svm(formula= general_category ~ ., 
                   data=train.4,
                   type='C-classification',
                   kernel='sigmoid',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier4.4, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

classifier4.5 <- svm(formula= general_category ~ ., 
                   data=train.4,
                   type='C-classification',
                   kernel='radial',
                   cross=5,
                   class.weights=weights
                  )

y_pred <- predict(classifier4.5, newdata=valid)
cm <- confusionMatrix(y_pred, valid[, 'general_category'], positive="1")
cm
draw_confusion_matrix(cm)

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category != 'suspect Blood Donor'
test <- test[mask,]


# change the category
mask <- test$Category != 'Blood Donor'
test[mask, 'general_category'] <- 'Hepatitis'
test[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
test$Category <- factor(test$Category)
test$general_category <- factor(test$general_category)


test$general_category <- c(-1, 1)[unclass(as.factor(test$general_category))]
test$general_category <- as.factor(test$general_category)


y_pred <- predict(classifier2.1, test)

test[, 'predict'] = y_pred

test <- test[-trainIndex, ]

mask <- test$Category == 'Cirrhosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Fibrosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'])
draw_confusion_matrix(cm)

mask <- test$Category == 'Hepatitis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Blood Donor'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category == 'suspect Blood Donor'
test <- test[mask,]
y_pred <- predict(classifier2.1, test)
test[, 'predict'] = y_pred

test

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category != 'suspect Blood Donor'
test <- test[mask,]


# change the category
mask <- test$Category != 'Blood Donor'
test[mask, 'general_category'] <- 'Hepatitis'
test[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
test$Category <- factor(test$Category)
test$general_category <- factor(test$general_category)


test$general_category <- c(-1, 1)[unclass(as.factor(test$general_category))]
test$general_category <- as.factor(test$general_category)


y_pred <- predict(classifier2.5, test)

test[, 'predict'] = y_pred
test <- test[-trainIndex, ]

mask <- test$Category == 'Cirrhosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Fibrosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Hepatitis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Blood Donor'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category == 'suspect Blood Donor'
test <- test[mask,]
y_pred <- predict(classifier2.5, test)
test[, 'predict'] = y_pred

test

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category != 'suspect Blood Donor'
test <- test[mask,]


# change the category
mask <- test$Category != 'Blood Donor'
test[mask, 'general_category'] <- 'Hepatitis'
test[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
test$Category <- factor(test$Category)
test$general_category <- factor(test$general_category)


test$general_category <- c(-1, 1)[unclass(as.factor(test$general_category))]
test$general_category <- as.factor(test$general_category)


y_pred <- predict(classifier3.1, test)

test[, 'predict'] = y_pred
test <- test[-trainIndex, ]

mask <- test$Category == 'Cirrhosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Fibrosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Hepatitis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Blood Donor'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category == 'suspect Blood Donor'
test <- test[mask,]
y_pred <- predict(classifier3.1, test)
test[, 'predict'] = y_pred

test

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category != 'suspect Blood Donor'
test <- test[mask,]


# change the category
mask <- test$Category != 'Blood Donor'
test[mask, 'general_category'] <- 'Hepatitis'
test[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
test$Category <- factor(test$Category)
test$general_category <- factor(test$general_category)


test$general_category <- c(-1, 1)[unclass(as.factor(test$general_category))]
test$general_category <- as.factor(test$general_category)


y_pred <- predict(classifier3.5, test)

test[, 'predict'] = y_pred
test <- test[-trainIndex, ]

mask <- test$Category == 'Cirrhosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Fibrosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Hepatitis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Blood Donor'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category == 'suspect Blood Donor'
test <- test[mask,]
y_pred <- predict(classifier3.5, test)
test[, 'predict'] = y_pred

test



test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category != 'suspect Blood Donor'
test <- test[mask,]


# change the category
mask <- test$Category != 'Blood Donor'
test[mask, 'general_category'] <- 'Hepatitis'
test[!mask, 'general_category'] <- 'Blood Donor'

# remove unwanted category
test$Category <- factor(test$Category)
test$general_category <- factor(test$general_category)


test$general_category <- c(-1, 1)[unclass(as.factor(test$general_category))]
test$general_category <- as.factor(test$general_category)


y_pred <- predict(classifier4.5, test)

test[, 'predict'] = y_pred
test <- test[-trainIndex, ]

mask <- test$Category == 'Cirrhosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Fibrosis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Hepatitis'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

mask <- test$Category == 'Blood Donor'
cm <- confusionMatrix(test[mask, 'predict'], test[mask, 'general_category'], positive="1")
draw_confusion_matrix(cm)

test <- hepc.backup
missing_cols <- colnames(test)[colSums(is.na(test)) > 0]

# impute missing value with median for each columns by category
for (col in missing_cols){
    test[, col] <- ave(test[, col], 
                       test$Category, 
                       FUN = function(x) ifelse(is.na(x), median(x, na.rm=TRUE), x)
                      )
}

# filter the suspect blood donor
mask <- test$Category == 'suspect Blood Donor'
test <- test[mask,]
y_pred <- predict(classifier4.5, test)
test[, 'predict'] = y_pred

test


