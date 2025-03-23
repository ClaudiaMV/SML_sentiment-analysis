# ------------------------- Load Libraries ---------------------------------------------------
library(dplyr)         # General data manipulation
library(janitor)       # Cleaning column names
library(MASS)          # AIC/BIC calculations
library(gridExtra)     # Arranging multiple plots
library(fmsb)          # Radar chart visualization
library(RColorBrewer)  # Color palettes
library(pacman)  

# Load necessary libraries using pacman
pacman::p_load(party, caret, knitr, kableExtra, ggplot2, tidyr, reshape2, ROSE)

# -------------------------- Load and Clean Data -----------------------------------
sent <- read.csv("SentimentAnalysis_control.csv") %>% as.data.frame()
blob <- read.csv("TextBlob_control.csv") %>% as.data.frame()
vader <- read.csv("vader_control.csv") %>% as.data.frame()
emotions <- read.csv("tokens_emotion.csv") %>% as.data.frame()

# Clean column names
sent <- clean_names(sent)
blob <- clean_names(blob)
vader <- clean_names(vader)
emotions <- clean_names(emotions)

# Add sentiment scores from other datasets
sent$polarity_blob <- blob$polarity
sent$compound_vader <- vader$compound
sent$emotions <- emotions$emotions - 4  # Transform for polarity

# Ensure 'word_count' exists
if (!"word_count" %in% colnames(sent)) {
  stop("Error: `word_count` column is missing from the dataset.")
}

# Normalize sentiment scores
sent <- sent %>%
  mutate(sentiment_gi = sentiment_gi / word_count,
         sentiment_he = sentiment_he / word_count,    
         sentiment_qdap = sentiment_qdap / word_count,  
         sentiment_blob = polarity_blob / word_count,
         sentiment_vader = compound_vader / word_count) %>% 
  mutate(sentiment_total = rowSums(across(c(sentiment_gi, sentiment_he, sentiment_qdap, sentiment_blob, sentiment_vader)), na.rm = TRUE))

# Select only relevant columns
sent <- sent[, c("sentiment_gi", "sentiment_he", "sentiment_qdap", "sentiment_blob", "sentiment_vader", "emotions", "sentiment_total")]

# Convert 'emotions' to ordered factor
sent$emotions <- factor(sent$emotions, ordered = TRUE)

# -------------------------- Compute Class Weights Before Oversampling -------------------
class_counts_original <- table(sent$emotions)  # Original class distribution
inverse_weights <- 1 / class_counts_original  # Compute inverse frequencies
log_weights <- log(1 + inverse_weights)  # Apply log smoothing

# Scale weights so that they sum to the total number of observations
scaled_weights <- (log_weights / sum(log_weights)) * length(sent$emotions)

# Assign weights to each class in the original data
original_weights <- setNames(scaled_weights, levels(sent$emotions))

# -------------------------- Perform Oversampling -----------------------------------
target_freq <- 78  # Target frequency per class

sent_corrected <- sent %>%
  group_by(emotions) %>%
  sample_n(size = target_freq, replace = TRUE) %>%
  ungroup()

# Ensure oversampling was successful
if (nrow(sent_corrected) != target_freq * length(unique(sent_corrected$emotions))) {
  stop("Error: Oversampling failed, dataset size does not match expected count.")
}

# Update the dataset
sent <- sent_corrected

# -------------------------- Adjust Weights After Oversampling -------------------
class_counts_new <- table(sent$emotions)  # New class distribution

# Assign original weights but scale them for new distribution
adjusted_weights <- original_weights[as.character(sent$emotions)]

# Normalize weights to reflect the new target frequency of 78
adjusted_weights <- adjusted_weights * (target_freq / mean(adjusted_weights))

# Ensure weights are integers for ctree()
adjusted_weights <- as.integer(round(adjusted_weights))

# Ensure weights are positive
adjusted_weights[is.na(adjusted_weights)] <- 1
adjusted_weights <- pmax(adjusted_weights, 1)

# -------------------------- Train-Test Split -----------------------------------
set.seed(123)  # For reproducibility
emotions <- sent$emotions

# Perform train-test split
train_indices <- createDataPartition(emotions, p = 0.8, list = FALSE)
test_indices <- setdiff(seq_len(nrow(sent)), train_indices)

# Create train and test datasets
train_data <- sent[train_indices, ]
test_data <- sent[test_indices, ]

# Subset weights correctly to match new dataset
train_weights <- adjusted_weights[train_indices]
test_weights <- adjusted_weights[test_indices]

# Ensure no missing or zero weights
train_weights[is.na(train_weights)] <- 1
test_weights[is.na(test_weights)] <- 1
train_weights <- pmax(train_weights, 1)
test_weights <- pmax(test_weights, 1)

# Debugging: Check if sizes match
cat("Train Data Size:", nrow(train_data), " | Train Weights Size:", length(train_weights), "\n")
cat("Test Data Size:", nrow(test_data), " | Test Weights Size:", length(test_weights), "\n")

# Stop execution if a mismatch occurs
if (length(train_weights) != nrow(train_data) | length(test_weights) != nrow(test_data)) {
  stop("🚨 Error: Mismatch between train_weights/test_weights and train_data/test_data size")
}

# -------------------------- Train and Evaluate Models -----------------------------------
train_ctree_model <- function(predictor, predictor_name) {
  cat("\nTraining model for", predictor_name, "using CTree with Cost-Sensitive Learning...\n")
  
  # Subset training and testing data
  train_set <- data.frame(emotions = train_data$emotions, predictor = train_data[[predictor]])
  test_set <- data.frame(emotions = test_data$emotions, predictor = test_data[[predictor]])
  
  # Train CTree model with integer-adjusted weights
  ctree_model <- ctree(emotions ~ predictor, data = train_set, weights = train_weights)
  pred <- predict(ctree_model, newdata = test_set, type = "response")
  
  # Ensure factor levels match
  pred <- factor(pred, levels = levels(test_set$emotions), ordered = TRUE)
  
  # Compute accuracy
  accuracy <- mean(pred == test_set$emotions, na.rm = TRUE)
  cat("Accuracy for", predictor_name, ":", accuracy, "\n")
  
  return(list(model = ctree_model, predictions = pred, accuracy = accuracy, test_set = test_set))
}

# Train models
models <- list(
  GI = train_ctree_model("sentiment_gi", "GI"),
  HE = train_ctree_model("sentiment_he", "HE"),
  QDAP = train_ctree_model("sentiment_qdap", "QDAP"),
  BLOB = train_ctree_model("sentiment_blob", "Blob"),
  VADER = train_ctree_model("sentiment_vader", "VADER"),
  TOTAL = train_ctree_model("sentiment_total", "Total")
)

# -------------------------- Generate Accuracy Table -----------------------------------
accuracy_results <- data.frame(
  Predictor = names(models),
  Accuracy = sapply(models, function(x) x$accuracy)
)
print(accuracy_results)

# -------------------------- Compute Confusion Matrices for Each Model -----------------------------------

# Function to compute and print confusion matrix
compute_conf_matrix <- function(pred, actual, predictor_name) {
  cat("\nConfusion matrix for", predictor_name, "predictor:\n")
  conf_matrix <- confusionMatrix(pred, actual)
  print(conf_matrix)
  return(conf_matrix)
}

# Compute confusion matrices for each model
conf_matrices <- list(
  GI = compute_conf_matrix(models$GI$predictions, models$GI$test_set$emotions, "GI"),
  HE = compute_conf_matrix(models$HE$predictions, models$HE$test_set$emotions, "HE"),
  QDAP = compute_conf_matrix(models$QDAP$predictions, models$QDAP$test_set$emotions, "QDAP"),
  BLOB = compute_conf_matrix(models$BLOB$predictions, models$BLOB$test_set$emotions, "Blob"),
  VADER = compute_conf_matrix(models$VADER$predictions, models$VADER$test_set$emotions, "VADER"),
  TOTAL = compute_conf_matrix(models$TOTAL$predictions, models$TOTAL$test_set$emotions, "Total")
)

# -------------------------- Compute Precision, Recall, F1-Score, Specificity, FPR, FNR -----------------------------------

# Function to calculate metrics from a confusion matrix
calculate_metrics <- function(conf_matrix) {
  cm <- conf_matrix$table  # Extract confusion matrix table
  
  # Calculate precision, recall, and F1-score
  precision <- diag(cm) / colSums(cm)  # TP / (TP + FP)
  recall <- diag(cm) / rowSums(cm)     # TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)  # F1-Score
  
  # Calculate specificity, FPR (False Positive Rate), FNR (False Negative Rate)
  TN <- sapply(1:nrow(cm), function(i) sum(cm[-i, -i]))
  FP <- colSums(cm) - diag(cm)
  FN <- rowSums(cm) - diag(cm)
  specificity <- TN / (TN + FP)  
  fpr <- FP / (FP + TN)  
  fnr <- FN / (FN + diag(cm))  
  
  # Compute accuracy
  accuracy <- sum(diag(cm)) / sum(cm)  
  
  # Handle NaN cases (caused by division by zero)
  precision[is.na(precision)] <- 0
  recall[is.na(recall)] <- 0
  f1_score[is.na(f1_score)] <- 0
  specificity[is.na(specificity)] <- 0
  fpr[is.na(fpr)] <- 0
  fnr[is.na(fnr)] <- 0
  
  # Return as a data frame
  return(data.frame(
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score,
    Specificity = specificity,
    FPR = fpr,
    FNR = fnr,
    Accuracy = accuracy
  ))
}

# Compute metrics for each model
metrics_list <- list(
  GI = calculate_metrics(conf_matrices$GI),
  HE = calculate_metrics(conf_matrices$HE),
  QDAP = calculate_metrics(conf_matrices$QDAP),
  BLOB = calculate_metrics(conf_matrices$BLOB),
  VADER = calculate_metrics(conf_matrices$VADER),
  TOTAL = calculate_metrics(conf_matrices$TOTAL)
)

# Convert to a data frame for easier viewing
metrics_df <- do.call(rbind, metrics_list)
metrics_df$Model <- rownames(metrics_df)

# Print the computed precision-recall metrics
print(metrics_df)

# -------------------------- Plot Confusion Matrices as Heatmaps -----------------------------------

# Function to plot a confusion matrix as a heatmap
plot_confusion_matrix <- function(conf_matrix, title) {
  conf_mat <- as.data.frame(conf_matrix$table)
  colnames(conf_mat) <- c("Prediction", "Reference", "Freq")
  
  ggplot(conf_mat, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black", size = 5) +  # Add frequency labels
    scale_fill_gradient(low = "blue", high = "red") +  # Color scale
    labs(title = title, x = "Actual Class", y = "Predicted Class") +
    theme_minimal(base_size = 14)
}

# Generate confusion matrix plots for each model
plot_gi <- plot_confusion_matrix(conf_matrices$GI, "GI Predictor")
plot_he <- plot_confusion_matrix(conf_matrices$HE, "HE Predictor")
plot_qdap <- plot_confusion_matrix(conf_matrices$QDAP, "QDAP Predictor")
plot_blob <- plot_confusion_matrix(conf_matrices$BLOB, "Blob Predictor")
plot_vader <- plot_confusion_matrix(conf_matrices$VADER, "VADER Predictor")


# Arrange and display all confusion matrix plots
grid.arrange(plot_gi, plot_he, plot_qdap, plot_blob, plot_vader,
             ncol = 3, nrow = 2)





