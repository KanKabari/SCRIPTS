# Load required packages
library(tidyverse) # for data manipulation
library(tm) # for text mining
library(textstem) # for stemming and lemmatization

#Load the Dataset
imdb <- read.csv("IMDB_dataset.csv")
View(imdb)

# Convert labels to a factor
imdb$sentiment <- as.factor(imdb$sentiment)

# Create a Corpus
TextDoc <- Corpus(VectorSource(review))

# Replacing "<", ">" and "/" with space
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
TextDoc <- tm_map(TextDoc, toSpace, "<")
TextDoc <- tm_map(TextDoc, toSpace, ">")
TextDoc <- tm_map(TextDoc, toSpace, "/")

# Convert the text to lower case
TextDoc <- tm_map(TextDoc, content_transformer(tolower))

# Remove punctuations
TextDoc <- tm_map(TextDoc, removePunctuation)

# Remove numbers
TextDoc <- tm_map(TextDoc, removeNumbers)

# Remove english common stopwords
TextDoc <- tm_map(TextDoc, removeWords, stopwords("english"))

# Eliminate extra white spaces
TextDoc <- tm_map(TextDoc, stripWhitespace)

# Lemmatize the text (reduce words to their base or dictionary form)
TextDoc <- tm_map(TextDoc, content_transformer(lemmatize_strings))

#-------------------------------------------------------------------------------
# Document-term Matrix
# Create a Document-Term Matrix with TF-IDF
dtm <- DocumentTermMatrix(TextDoc) # "control = list(weighting = weightTfIdf)"
dtm <- removeSparseTerms(dtm, 0.99)

# Convert DTM to a matrix
tfidf_matrix <- as.data.frame(as.matrix(dtm))
#tfidf_matrix <- as.matrix(dtm)
View(tfidf_matrix)

# Add the sentiment labels to the dtm
tfidf_matrix$sentiment <- imdb$sentiment
length(tfidf_matrix$sentiment) #confirm it corresponds with the number of reviews

#-------------------------------------------------------------------------------
# Applying PCA

# Confirm the position of the "sentiment" column in the dataframe.
which(colnames(tfidf_matrix) == "sentiment")

# To apply principal components analysis we use the command princomp().
#
# The last column contains the class variable so we exclude it and also check how long it takes to perform a PCA on the data.
system.time(tfidf_pca <- princomp(tfidf_matrix[, -ncol(tfidf_matrix)], cor = TRUE)) 
View(tfidf_pca)
# It take 196 seconds (over 3 minutes).

# Store the summary in an object
s <- summary(tfidf_pca)

#-------------------------------------------------------------------------------
# Extract PCA scores from the summary
scores <- s$scores

# select the optimal number of PCs to be used
top19_pcs <- scores[, 1:19] 

# Convert to a dataframe
top19_data <- data.frame(top19_pcs)

# Combine top PCs with the response variable
top19_data$sentiment <- imdb$sentiment
dim(top19_data)
#-------------------------------------------------------------------------------

#. Machine Learning
# Load libraries required for machine learning
library(caret) # for LR and other ML purposes
library(randomForest) # for RF
library(e1071) # for SVM
library(MASS) # for LDA
library(nnet) # for NN
library(rpart) # for DT

# Split the data into training and test sets
set.seed(50)
trainIndex <- createDataPartition(top19_data$sentiment, p = 0.8, list = FALSE)
trainData <- top19_data[trainIndex, ]
testData <- top19_data[-trainIndex, ]

# Separate features and labels
train_x <- trainData[, -ncol(top19_data)] 
train_y <- trainData$sentiment 
test_x <- testData[, -ncol(top19_data)] 
test_y <- testData$sentiment 

#----------------------------------------------------------------------
# Set-up parallel processing (it works with the caret package)
library(parallel) # for detecting number of cores and parallelising BART model
library(doParallel) # for parallelising all models except BART

# Detect the number of available cores
detectCores()

# Register parallel backend
registerDoParallel(cores = 4)

# Train the models and record the time taken
system.time(logistic_model <- train(sentiment ~ .,
                                    data = trainData,
                                    method = "glm"))

system.time(rf_model <- train(sentiment ~ .,
                              data = trainData,
                              method = "rf"))

system.time(svm_model <- train(sentiment ~ .,
                               data = trainData,
                               method = "svmLinear"))

system.time(lda_model <- train(sentiment ~ .,
                               data = trainData,
                               method = "lda"))

system.time(tree_model <- train(sentiment ~ .,
                                data = trainData,
                                method = "rpart"))

system.time(nn_model <- train(sentiment ~ .,
                              data = trainData,
                              method = "nnet"))
#-------------------------------------------------------------------------------
# Evaluate the traditional ML models
logistic_predictions <- predict(logistic_model, test_x)
rf_predictions <- predict(rf_model, test_x)
svm_predictions <- predict(svm_model, test_x)
lda_predictions <- predict(lda_model, test_x)
tree_predictions <- predict(tree_model, test_x)
nn_predictions <- predict(nn_model, test_x)

#-------------------------------------------------------------------------------
# Print the confusion matrix for each model
cat("Logistic Regression Confusion Matrix:\n")
print(confusionMatrix(logistic_predictions, test_y))

cat("Random Forest Confusion Matrix:\n")
print(confusionMatrix(rf_predictions, test_y))

cat("SVM Confusion Matrix:\n")
print(confusionMatrix(svm_predictions, test_y))

cat("LDA Confusion Matrix:\n")
print(confusionMatrix(lda_predictions, test_y))

cat("Decision Tree Confusion Matrix:\n")
print(confusionMatrix(tree_predictions, test_y))

cat("Neural Network Confusion Matrix:\n")
print(confusionMatrix(nn_predictions, test_y))

# Close the doParallel processing
stopImplicitCluster()
# Though the cluster will be closed. automatically at the conclusion of the R 
# session, it is better form to dos so explicitly.

#-------------------------------------------------------------------------------
# Train the BART model and determine the accuracy

library(BART)

# For binary classification, the outcome y.train is a vector containing zeros(0) and ones(1)
trainData <- trainData %>% 
  mutate(sentiment = ifelse(sentiment == "positive", 1, 0))

testData <- testData %>% 
  mutate(sentiment = ifelse(sentiment == "positive", 1, 0))

test_y <- testData$sentiment
train_y <- trainData$sentiment

# Train the BART model for classification
require(parallel)
system.time(bart_model <- mc.pbart(x.train = train_x, y.train = train_y,
                                   x.test = test_x, ndpost = 1000,
                                   mc.cores = 4, seed = 99))

# Predict on the test set
bart_pred <- predict(bart_model, newdata = test_x)
str(bart_pred)

# Check the summary of the probability values on the test set after predicting
summary(bart_pred$prob.test.mean)

# Using the median as a threshold, convert probabilities to binary class labels
bart_pred_class <- ifelse(bart_pred$prob.test.mean > 0.5031930, 1, 0)

# Confusion matrix
conf_matrix <- table(bart_pred_class, test_y)
conf_matrix

# Calculate accuracy
accuracy <- mean(bart_pred_class == test_y)
accuracy
#-------------------------------------------------------------------------------

# Visualizations for Figures 3 to 6 in the dissertation

pc_used <- c(2, 3, 4, 5, 6, 7, 8, 9, 10, 
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
             30, 40, 50, 60, 70, 80, 90, 100, 150, 200)
# pc2_to_10 <- c(2, 3, 4, 5, 6, 7, 8, 9, 10)
# pc11_to_20 <- c(11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
# pc30_to_200 <- c(30, 40, 50, 60, 70, 80, 90, 100, 150, 200)

accuracy <- c(72.76, 72.87, 73.40, 77.26, 77.40, 81.62, 81.78, 82.36, 83.30,
              84.43, 84.78, 84.95, 85.86, 85.82, 85.97, 85.92, 85.85, 86.15, 85.95,
              86.65, 87.05, 87.00, 87.09, 87.25, 87.15, 87.26, 87.30, 87.38, 87.37)
# accuracy2_to_10 <- c(72.76, 72.87, 73.40, 77.26, 77.40, 81.62, 81.78, 82.36, 83.30)
# accuracy11_to_20 <- c(84.43, 84.78, 84.95, 85.86, 85.82, 85.97, 85.92, 85.85, 86.15, 85.95)
# accuracy30_to_200 <- c(86.65, 87.05, 87.00, 87.09, 87.25, 87.15, 87.26, 87.30, 87.38, 87.37)

timings <- c(6.030, 6.430, 6.739, 7.001, 7.366, 10.687, 8.966, 9.467, 10.214,
             10.339, 11.102, 11.296, 11.851, 12.273, 13.008, 13.490, 14.396, 14.746, 15.398,
             21.988, 29.775, 39.149, 48.990, 60.576, 73.486, 87.755, 102.601, 197.215, 352.984)
# timings2_to_10 <- c(6.030, 6.430, 6.739, 7.001, 7.366, 10.687, 8.966, 9.467, 10.214)
# timings11_to_20 <- c(10.339, 11.102, 11.296, 11.851, 12.273, 13.008, 13.490, 14.396, 14.746, 15.398)
# timings30_to_200 <- c(21.988, 29.775, 39.149, 48.990, 60.576, 73.486, 87.755, 102.601, 197.215, 352.984)

par(mfrow = c(1, 2))
plot(pc_used, accuracy,
     type = "b",
     pch = 4,
     xlab = "Number of principal components",
     ylab = "Accuracy (%)")

abline(v = 19)

plot(pc_used, timings,
     type = "b",
     pch = 4,
     xlab = "Number of principal components",
     ylab = "Training Time (sec)")

abline(v = 19)

par(mfrow = c(1, 1))
#
#
# Visualisation for Figure 7
models <- c('L.R', 'R.F', 'S.V.M', 'L.D.A', 'D.T', 'N.N', 'B.A.R.T')
training_time <- c(4.411, 1261.526, 1872.394, 2.767, 7.737, 183.225, 40.668)
accuracy <- c(86.15, 85.49, 86.19, 86.14, 75.30, 85.82, 85.21)

# Create scatter plot
plot(training_time, accuracy, 
     pch=19, col="blue", cex=2, 
     xlab="Training Time (sec)", 
     ylab="Accuracy (%)",
     cex.lab=1.5,   # Increase the size of axis labels
     cex.axis=1.2)

# Add grid lines
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")

# Add text labels for each model
text(training_time, accuracy, labels=models, pos=4, cex=1.2, col="black", offset=0.5)
#-------------------------------------------------------------------------------

# Figure 8: Feature Importance Rankings 

# 1. Extract Feature Importance
# BART does not directly output feature importance, but we can approximate it by counting 
# how frequently features are used to split trees (variable selection frequencies)
feature_importance <- bart_model$varcount.mean  # This gives the average counts of splits

# 2. Create a DataFrame for Better Visualization
feature_importance_df <- data.frame(
  Feature = colnames(train_x),
  Importance = feature_importance
)

# Sort by importance
feature_importance_df <- feature_importance_df[order(-feature_importance_df$Importance), ]

# 3. Visualize the Feature Importance Rankings
ggplot(feature_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Features", y = "Importance")

# 4. Analyze Top Influential Features
# Print top N important features
top_n <- 10  # Number of top features to show
cat("\nTop", top_n, "Features Influencing Sentiment Predictions:\n")
print(head(feature_importance_df, top_n))
#
#-------------------------------------------------------------------------------
# Figure 9: Uncertainty estimates of BART model
# Check prob.test.mean
#
pi_hat <- apply(bart_pred$prob.test,
                2,
                mean)

length(pi_hat)

o <- order(pi_hat)

range(pi_hat - bart_pred$prob.test.mean)

pi_lower <- apply(bart_pred$prob.test,
                  2,
                  quantile,
                  prob = 0.025)

pi_upper <- apply(bart_pred$prob.test,
                  2,
                  quantile,
                  prob = 1 - 0.025)

plot(pi_hat[o])

l_lower <- loess.smooth(1:10000,
                        pi_lower[o],
                        span = 0.25)

l_upper<- loess.smooth(1:10000,
                       pi_upper[o],
                       span = 0.25)
# Plot
matplot(1:10000,
        cbind(pi_hat, pi_lower, pi_upper)[o,],
        type = "n",
        lty = 1,
        col = "black",
        ylim = c(-0.5,1.5),
        xlab = "Observation number",
        ylab = "Predicted probability and credible interval")

abline(h = c(0, 1))

test_y_ordered <- test_y[o]

points(1:10000,
       jitter(test_y_ordered),
       col = ifelse(test_y_ordered == 1, "green", "red"),
       pch = 4)

matlines(1:10000,
         cbind(pi_hat, pi_lower, pi_upper)[o,],
         col = "black",
         lty = 1)
#
lines(l_lower$x,
      l_lower$y,
      col = "blue")

lines(l_upper$x,
      l_upper$y,
      col = "blue")

cbind(pi_lower, bart_pred$prob.test.mean, pi_upper)[o,]

#-------------------------------------------------------------------------------







