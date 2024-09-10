# Load required packages
library(tidyverse) # for data manipulation
library(tidytext) # for text mining

#Load the Dataset
imdb <- read.csv("IMDB_dataset.csv")
View(imdb)
#attach(imdb)

# Convert labels to a factor
imdb$sentiment <- as.factor(imdb$sentiment)

# Add an identifier column using row number
imdb <- imdb %>%
  mutate(id = row_number())

#---------------------------Preprocessing steps-----------------------------
# Preprocessing steps
preprocessed_text <- imdb %>%
  # Convert to lowercase
  mutate(review = str_to_lower(review)) %>%
  # Remove numbers
  mutate(review = str_replace_all(review, "[0-9]", "")) %>%
  # Remove punctuation
  mutate(review = str_replace_all(review, "[[:punct:]]|<|>|/", " ")) %>%
  # Remove extra white spaces
  mutate(review = str_squish(review))

# Tokenize the text
tidy_text <- preprocessed_text %>%
  unnest_tokens(word, review)

# Remove stopwords
tidy_text <- tidy_text %>%
  anti_join(stop_words, by = "word")

# Define custom stopwords
custom_stopwords <- c(stop_words$word, "br", "ll", "t", "s", "d", "mr", "ve")

# Convert to a data frame
custom_stopwords_df <- data_frame(word = custom_stopwords)

# Remove custom stopwords
tidy_text <- tidy_text %>%
  anti_join(custom_stopwords_df, by = "word")

# Load sentiment lexicon (Bing lexicon)
bing_sentiments <- get_sentiments("bing")

# Join with the sentiment lexicon
text_sentiment <- tidy_text %>%
  inner_join(bing_sentiments, by = "word")

# Count the number of positive, negative, and neutral words
sentiment_count <- text_sentiment %>%
  group_by(id, sentiment.y) %>%
  summarize(count = n()) %>%
  spread(sentiment.y, count, fill = 0) %>%
  ungroup()

# Add a column for neutral words (total words - positive - negative)
total_words <- tidy_text %>%
  group_by(id) %>%
  summarize(total = n())

sentiment_count <- sentiment_count %>%
  left_join(total_words, by = "id") %>%
  mutate(neutral = total - positive - negative) %>% 
  select(-total)

# Ensure all original IDs are present
all_ids <- imdb %>%
  select(id)

sentiment_count <- all_ids %>%
  left_join(sentiment_count, by = "id")

## Check for and count missing values in the dataset 
colSums(is.na(sentiment_count)) #Count the number of missing values per column
# Likely occurred because some reviews did not have any words after preprocessing (such as stopwords removal, tokenization,
# and joining with the sentiment lexicon). This can happen if a review is very short, consists entirely of stopwords, 
# or contains only words that are not present in the sentiment lexicon.

# Add the original sentiment labels to the sentiment_count dataframe
sentiment_count$sentiment <- imdb$sentiment
#length(imdb$sentiment_score) #confirm it corresponds with the number of reviews

# Remove rows with NA values (for any reviews that had no words)
sentiment_count <- na.omit(sentiment_count)

# Confirm if NA's have been removed
colSums(is.na(sentiment_count))

# Print the first few rows of the result to check
print(head(sentiment_count))

# Add a column of sentiment scores (no. of positive words - no. of negative words)
sentiment_count <- sentiment_count %>%
                     mutate(overall = positive - negative)

sentiment_count_2 <- sentiment_count %>%
  left_join(total_words, by = "id") %>%
  mutate(positive_prop = positive / total,
         negative_prop = negative / total,
         neutral_prop = neutral / total,
         sentiment_prop = overall / total)

# boxplot
boxplot(overall ~ sentiment,
        data = sentiment_count)
#-------------------------------------------------------------------------------
# Sampling 

# Collecting two different samples from sentiment_count_2
#
#
# How many reviews?
#
n_row <- nrow(sentiment_count_2)
#
# Index of every review
#
indices <- 1:n_row
indices
#
# How many to take (percentage)
#
p <- 0.1 # Proportion under your control
#
no <- floor(p * n_row) # returns the highest number that is less than or equal to a number set as a parameter.
no
#
# Define the indices of the training set
#
training_set <- sample(1:n_row,
                       size = no,
                       replace = FALSE)

training_set
#
# Now get the indices of the potential validation set that are *different*
#
validation_available <- (1:n_row)[-training_set]
#
# Now sample these to get the indices of the validation set
#
validation_set <- sample(validation_available,
                         size = no,
                         replace = FALSE)
  
# Get the training data
#
sentiment_count_small <- sentiment_count_2[training_set,] 

# confirm the dimension of the training set
dim(sentiment_count_small)

# Get the validation data
sentiment_count_validation <- sentiment_count_2[validation_set,] 

# confirm the dimension of the validation set
dim(sentiment_count_validation)

#-------------------------------------------------------------------------------
# Checking for the best model fit using the AIC value
# Cross-validating through a combination of variables

m <- glm(sentiment ~ log(positive + 0.01) + log(neutral + 0.01) + log(negative + 0.01), # AIC Value:  5630.7
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~ positive  + neutral + negative , # AIC Value:  5578.1
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~  overall, # AIC Value:  5800.3
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~ positive_prop  + neutral_prop + negative_prop , # AIC Value:  5344.4
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~  sentiment_prop, # AIC Value:  5342.8
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~ positive_prop  + neutral_prop + negative_prop + total, # AIC Value:  5336.7
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~ total, # AIC Value: 6917.7
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~ positive_prop  + neutral_prop + negative_prop + total + sentiment_prop, # AIC Value:  5336.7
         data = sentiment_count_small,
         family = binomial)

summary(m)

m <- glm(sentiment ~ positive_prop  + neutral_prop + negative_prop + sentiment_prop, # AIC Value:  5344.4
         data = sentiment_count_small,
         family = binomial)

summary(m)

#-------------------------------------------------------------------------------
# Machine Learning

# Load libraries required for machine learning
library(caret) # for LR and other ML purposes
library(randomForest) # for RF
library(e1071) # for SVM
library(MASS) # for LDA
library(nnet) # for NN
library(rpart) # for DT

# Split the data into training and test sets.
set.seed(50)
trainIndex <- createDataPartition(sentiment_count_validation$sentiment, p = 0.8, list = FALSE)
trainData <- sentiment_count_validation[trainIndex, ]
testData <- sentiment_count_validation[-trainIndex, ]

# Separate features and labels
train_x <- trainData[, -5] # A subset of the trainData dataframe which includes all rows and all columns (features) except the class column.
train_y <- trainData$sentiment # A subset of the trainData which iincludes only the target variable (sentiment).
test_x <- testData[, -5] # A subset of the testData dataframe which includes all rows and all columns (features) except the class column.
test_y <- testData$sentiment # A subset of the testData which iincludes the target variable (sentiment).

# NOTE: Remember to use the training and validation sets ("sentiment_count_small" and "sentiment_count_validation") 
# respectively in Line 235 in the createDataPartition() function and train all models including BART

#----------Train different machine learning models using the training and validation set-------

# RF
system.time(rf_model <- train(sentiment ~ positive_prop  + neutral_prop + negative_prop + total,
                              data = trainData,
                              method = "rf"))
# LR
system.time(logistic_model <- train(sentiment ~ positive_prop  + neutral_prop + negative_prop + total,
                        data = trainData,
                        method = "glm"))
# SVM
system.time(svm_model <- train(sentiment ~ positive_prop  + neutral_prop + negative_prop + total,
                   data = trainData,
                   method = "svmLinear"))
# LDA
system.time(lda_model <- train(sentiment ~ positive_prop  + neutral_prop + negative_prop + total,
                   data = trainData,
                   method = "lda"))
# DT
system.time(tree_model <- train(sentiment ~ positive_prop  + neutral_prop + negative_prop + total,
                  data = trainData,
                  method = "rpart"))
# NN
system.time(nn_model <- train(sentiment ~ positive_prop  + neutral_prop + negative_prop + total,
                  data = trainData,
                  method = "nnet"))

# Evaluate the models
logistic_predictions <- predict(logistic_model, test_x)
rf_predictions <- predict(rf_model, test_x)
svm_predictions <- predict(svm_model, test_x)
lda_predictions <- predict(lda_model, test_x)
tree_predictions <- predict(tree_model, test_x)
nn_predictions <- predict(nn_model, test_x)

# Print the confusion matrices
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

#---------------------------------BART Model-----------------------------------#

#1.-----------------Applied to Training and Validation samples.-----------------

# Train the BART model using parallel computing
library(BART)
library(parallel) # For parallel computing

# To know the number of threads (cores) that my computer can support and, the BART package can use
detectCores()

# For binary classification, the outcome y.train is a vector containing zeros(0) and ones(1)
trainData <- trainData %>% 
  mutate(sentiment = ifelse(sentiment == "positive", 1, 0))

testData <- testData %>% 
  mutate(sentiment = ifelse(sentiment == "positive", 1, 0))

test_y <- testData$sentiment
train_y <- trainData$sentiment

# Train the model 
system.time(bart_model <- mc.pbart(x.train = train_x, y.train = train_y,
                                   x.test = test_x, ndpost = 1000,
                                   mc.cores = 4, seed = 99))

# Predict on the test set
bart_pred <- predict(bart_model, newdata = test_x)
str(bart_pred)

# Check the summary of the probability values on the test set after predicting
summary(bart_pred$prob.test.mean)

# Convert probabilities to binary class labels
bart_pred_class <- ifelse(bart_pred$prob.test.mean > 0.5, 1, 0)

# Confusion matrix
conf_matrix <- table(bart_pred_class, test_y)
conf_matrix

# Calculate accuracy
accuracy <- mean(bart_pred_class == test_y)
accuracy

# NOTE: Ensure the apply the instructions on Line 245-246!!!

#-------------------------------------------------------------------------------
# Visualisations

# Creating a dataframe
method <- c("L.R",
            "R.F",
            "S.V.M",
            "L.D.A",
            "D.T",
            "N.N",
            "B.A.R.T")

accuracy_training <- c(74.92, 71.92, 75.03, 74.92, 73.32, 74.92, 74.52)

accuracy_validation <- c(76.43, 73.12, 76.43, 76.43, 73.22, 76.13, 75.13)

results <- data.frame(method, 
                      accuracy_training,
                      accuracy_validation)

results_2 <- results %>%
  pivot_longer(cols = 2:3,
               names_to = "Phase",
               values_to = "Accuracy") %>%
  mutate(Method = factor(method,
                         levels = c("L.R",
                                    "R.F",
                                    "S.V.M",
                                    "L.D.A",
                                    "D.T",
                                    "N.N",
                                    "B.A.R.T")),
         Phase = factor(Phase,
                        levels = c("accuracy_training", "accuracy_validation"),
                        labels = c("Training Set", "Validation Set")))

# Figure 1 in the dissertation
ggplot(results_2, 
       aes(x = Method,
           y = Accuracy)) +
  geom_point() + 
  geom_line(aes(group = Phase)) +
  labs(x = "Model",
       y = "Accuracy (%)") +
  facet_grid(. ~ Phase) + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90,
                                   hjust = 1,
                                   vjust = 0.5,
                                   size = 12),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 12,
                                  face = "bold"),
        strip.text = element_text(size = 12))

# Figure 2 in the dissertation
ggplot(results_2, 
       aes(x = Phase,
           y = Accuracy)) +
  geom_point() + 
  geom_line(aes(group = Method)) +
  labs(x = "Phase",
       y = "Accuracy (%)") +
  facet_grid(. ~ Method,
             labeller = label_wrap_gen(width=10)) + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90,
                                   hjust = 1,
                                   vjust = 0.5,
                                   size = 12),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 12,
                                  face = "bold"),
        strip.text = element_text(size = 12))





