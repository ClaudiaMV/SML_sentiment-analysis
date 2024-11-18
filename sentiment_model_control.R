# ------------------------- Libraries ---------------------------------------------------
# Load required libraries for data manipulation, cleaning, and model evaluation.
library(dplyr)         # For general data manipulation
library(janitor)       # For cleaning column names
library(MASS)          # For AIC/BIC
library(gridExtra)
library(fmsb)  # For radar chart
library(RColorBrewer)  # For color palettes


library(pacman) 
pacman::p_load(party, caret, knitr, dplyr, kableExtra, ggplot2, tidyr, reshape2, ROSE)
# -------------------------- Data Loading and Cleaning -----------------------------------
# Load the datasets
sent <- as.data.frame(read.csv("SentimentAnalysis_control.csv"))  # Load the sentiment control data
blob <- as.data.frame(read.csv("TextBlob_control.csv"))  # Load data with additional sentiment scores
emotions <- as.data.frame(read.csv("tokens_emotion.csv"))

# Clean the column names to ensure uniformity (e.g., converting spaces to underscores)
sent <- clean_names(sent)
blob <- clean_names(blob)
emotions <- clean_names(emotions)

# Add sentiment_blob column from the 'blob' dataset to 'sent'
sent$polarity_blob <- blob$polarity
sent$emotions <- emotions$emotions - 4 #transformed for polarity

# Create normalized scores for the sentiment variables (sentiment_gi, sentiment_he, sentiment_qdap, sentiment_blob)
sent <- sent %>%
  mutate(sentiment_gi = sentiment_gi / word_count,
         sentiment_he = sentiment_he / word_count,    
         sentiment_qdap = sentiment_qdap / word_count,  
         sentiment_blob = polarity_blob / word_count) %>% 
  mutate(sentiment_total = rowSums(across(c(sentiment_gi, sentiment_he, sentiment_qdap, sentiment_blob)), na.rm = TRUE))


sent <- sent[, c("sentiment_gi", "sentiment_he", "sentiment_qdap", "sentiment_blob", "emotions", "sentiment_total")] #Create a data set with only the relevant information

# -------------------------- Create Variables -----------------------------------
# Make sure 'emotions' is an ordered factor, and predictors are continuous
sentiment_gi <- sent$sentiment_gi
sentiment_he <- sent$sentiment_he 
sentiment_qdap <- sent$sentiment_qdap
sentiment_blob <- sent$sentiment_blob
sentiment_total <- sent$sentiment_total
summary(sent[,2:(ncol(sent)-1)])

# dependent variable
emotions <- factor(sent$emotions, ordered = TRUE)  

kable(table(sent$emotions),
      col.names = c("emotions", "Frequency"), align = 'l')  %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "left")

# Data Sampling -----------------------------------------------------------


#### -------------------------- Cost-Sensitive Learning: Apply Class Weights -------------------
## Calculate class weights inversely proportional to class frequencies
class_weights <- table(sent$emotions)  # Class frequencies
inverse_weights <- 1 / class_weights   # Inverse of class frequencies

# Logarithmic scaling to smooth extreme weights and avoid over-penalizing certain classes
log_weights <- log(1 + inverse_weights)  # Apply log scaling

# Proportional scaling: Scale the weights to match the number of observations
total_obs <- length(sent$emotions)  # Total number of observations
scaled_weights <- round(log_weights / sum(log_weights) * total_obs)  # Normalize and scale

# Assign the scaled weights to each observation based on their class
weights <- sapply(sent$emotions, function(x) scaled_weights[as.character(x)])

# Ensure weights are positive integers (replace 0 or negative weights with 1)
weights[weights <= 0] <- 1
weights <- as.integer(weights)  # Ensure weights are integers


#### -------------------------- Oversampling -------------------
# Set target frequency for each emotions level
target_freq <- 78

# Oversample the dataset manually
sent_corrected <- sent %>%
  group_by(emotions) %>%
  sample_n(size = target_freq, replace = TRUE) %>%
  ungroup()

# Display the table showing the distribution of 'Class' after oversampling
kable(table(sent_corrected$emotions),
      col.names = c("emotions", "Frequency"), align = 'l') %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "left")

sent <- sent_corrected

# -------------------------- Data Splitting -------------------------------------
# Split the data into training (80%) and testing (20%) sets for model evaluation
set.seed(123)  # Set seed for reproducibility
train_indices <- createDataPartition(emotions, p = 0.8, list = FALSE)

# Subset the data into training and test sets
train_data <- sent[train_indices, ]
test_data <- sent[-train_indices, ]

# Check the size of the training and test sets
cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# -------------------------- Train and Evaluate Each Predictor Model in a Series -----------------------------

# Prepare data subsets for each predictor
train_gi <- data.frame(emotions = emotions[train_indices], gi = sentiment_gi[train_indices])
test_gi <- data.frame(emotions = emotions[-train_indices], gi = sentiment_gi[-train_indices])

train_he <- data.frame(emotions = emotions[train_indices], he = sentiment_he[train_indices])
test_he <- data.frame(emotions = emotions[-train_indices], he = sentiment_he[-train_indices])

train_qdap <- data.frame(emotions = emotions[train_indices], qdap = sentiment_qdap[train_indices])
test_qdap <- data.frame(emotions = emotions[-train_indices], qdap = sentiment_qdap[-train_indices])

train_blob <- data.frame(emotions = emotions[train_indices], blob = sentiment_blob[train_indices])
test_blob <- data.frame(emotions = emotions[-train_indices], blob = sentiment_blob[-train_indices])

# Train and evaluate model for 'gi' using ctree with cost-sensitive learning
cat("\nTraining model for 'gi' predictor using ctree with Cost-Sensitive Learning...\n")
ctree_gi <- ctree(emotions ~ gi, data = train_gi, weights = weights[train_indices])  # Pass weights here
pred_gi <- predict(ctree_gi, newdata = test_gi, type = "response")
pred_gi <- factor(pred_gi, levels = levels(test_gi$emotions), ordered = TRUE)
accuracy_gi <- mean(pred_gi == test_gi$emotions)
cat("Accuracy for 'gi' predictor:", accuracy_gi, "\n")

# Train and evaluate model for 'he' using ctree with cost-sensitive learning
cat("\nTraining model for 'he' predictor using ctree with Cost-Sensitive Learning...\n")
ctree_he <- ctree(emotions ~ he, data = train_he, weights = weights[train_indices])  # Pass weights here
pred_he <- predict(ctree_he, newdata = test_he, type = "response")
pred_he <- factor(pred_he, levels = levels(test_he$emotions), ordered = TRUE)
accuracy_he <- mean(pred_he == test_he$emotions)
cat("Accuracy for 'he' predictor:", accuracy_he, "\n")

# Train and evaluate model for 'qdap' using ctree with cost-sensitive learning
cat("\nTraining model for 'qdap' predictor using ctree with Cost-Sensitive Learning...\n")
ctree_qdap <- ctree(emotions ~ qdap, data = train_qdap, weights = weights[train_indices])  # Pass weights here
pred_qdap <- predict(ctree_qdap, newdata = test_qdap, type = "response")
pred_qdap <- factor(pred_qdap, levels = levels(test_qdap$emotions), ordered = TRUE)
accuracy_qdap <- mean(pred_qdap == test_qdap$emotions)
cat("Accuracy for 'qdap' predictor:", accuracy_qdap, "\n")

# Train and evaluate model for 'blob' using ctree with cost-sensitive learning
cat("\nTraining model for 'blob' predictor using ctree with Cost-Sensitive Learning...\n")
ctree_blob <- ctree(emotions ~ blob, data = train_blob, weights = weights[train_indices])  # Pass weights here
pred_blob <- predict(ctree_blob, newdata = test_blob, type = "response")
pred_blob <- factor(pred_blob, levels = levels(test_blob$emotions), ordered = TRUE)
accuracy_blob <- mean(pred_blob == test_blob$emotions)
cat("Accuracy for 'blob' predictor:", accuracy_blob, "\n")


# Prepare data subsets for 'sentiment_total'
train_total <- data.frame(emotions = emotions[train_indices], sentiment_total = sentiment_total[train_indices])
test_total <- data.frame(emotions = emotions[-train_indices], sentiment_total = sentiment_total[-train_indices])

# Train and evaluate model for 'sentiment_total' using ctree with cost-sensitive learning
cat("\nTraining model for 'sentiment_total' predictor using ctree with Cost-Sensitive Learning...\n")
ctree_total <- ctree(emotions ~ sentiment_total, data = train_total, weights = weights[train_indices])  # Pass weights here
pred_total <- predict(ctree_total, newdata = test_total, type = "response")
pred_total <- factor(pred_total, levels = levels(test_total$emotions), ordered = TRUE)
accuracy_total <- mean(pred_total == test_total$emotions)
cat("Accuracy for 'sentiment_total' predictor:", accuracy_total, "\n")


#Print the accuracy results for each predictor
results <- data.frame(
  Predictor = c("gi", "he", "qdap", "blob", "sentiment_total"),
  Accuracy = c(accuracy_gi, accuracy_he, accuracy_qdap, accuracy_blob, accuracy_total)
)

print(results)


# -------------------------- Confusion Matrix for Each Model -------------------------------------

# Confusion matrix for 'gi' predictor
cat("\nConfusion matrix for 'gi' predictor:\n")
conf_matrix_gi <- confusionMatrix(pred_gi, test_gi$emotions)
print(conf_matrix_gi)

# Confusion matrix for 'he' predictor
cat("\nConfusion matrix for 'he' predictor:\n")
conf_matrix_he <- confusionMatrix(pred_he, test_he$emotions)
print(conf_matrix_he)

# Confusion matrix for 'qdap' predictor
cat("\nConfusion matrix for 'qdap' predictor:\n")
conf_matrix_qdap <- confusionMatrix(pred_qdap, test_qdap$emotions)
print(conf_matrix_qdap)

# Confusion matrix for 'blob' predictor
cat("\nConfusion matrix for 'blob' predictor:\n")
conf_matrix_blob <- confusionMatrix(pred_blob, test_blob$emotions)
print(conf_matrix_blob)

# Confusion matrix for 'total' predictor
cat("\nConfusion matrix for 'sentiment_total' predictor:\n")
conf_matrix_total <- confusionMatrix(pred_total, test_total$emotions)
print(conf_matrix_total)


## Confusion Matrix Plot ---------------------------------------------------
#frequency refers to the number of occurrences of each combination of predicted and actual classes. 
#It represents how often a particular pair of predicted and actual outcomes occurs in your test dataset.

# Function to plot confusion matrix as a heatmap
plot_confusion_matrix <- function(conf_matrix, title) {
  conf_mat <- as.data.frame(conf_matrix$table)
  colnames(conf_mat) <- c("Prediction", "Reference", "Freq")
  
  ggplot(conf_mat, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black") +
    scale_fill_gradient(low = "#eff6fc", high = "#ff9900") +
    labs(title = title, x = "Actual Class", y = "Predicted Class") +
    theme_minimal()
}

# Plot confusion matrix for gi pred ictor
plot_gi <- plot_confusion_matrix(conf_matrix_gi, "GI")

# Plot confusion matrix for he predictor
plot_he <- plot_confusion_matrix(conf_matrix_he, "HE")

# Plot confusion matrix for qdap predictor
plot_qdap <- plot_confusion_matrix(conf_matrix_qdap, "QDAP")

# Plot confusion matrix for blob predictor
plot_blob <- plot_confusion_matrix(conf_matrix_blob, "Blob")

# Plot confusion matrix for sentiment_total predictor
plot_total <- plot_confusion_matrix(conf_matrix_total, "Total")


# Arrange all four plots into a grid
grid.arrange(plot_gi, plot_he, plot_qdap, plot_blob, plot_total, ncol = 2, nrow = 3)


# -------------------------- Precision, Recall, and F1-Score Calculation -------------------------------------

# Extended function to calculate more metrics from a confusion matrix
calculate_metrics <- function(conf_matrix) {
  # Extract the confusion matrix table
  cm <- conf_matrix$table
  
  # Calculate precision, recall, and F1-score for each class
  precision <- diag(cm) / colSums(cm)  # TP / (TP + FP)
  recall <- diag(cm) / rowSums(cm)     # TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)  # F1-Score
  
  # Calculate support (the number of true instances for each class)
  support <- rowSums(cm)               # TP + FN
  
  # Calculate accuracy
  total <- sum(cm)
  correct <- sum(diag(cm))
  accuracy <- correct / total          # (TP + TN) / (TP + TN + FP + FN)
  
  # Calculate specificity for each class
  TN <- sapply(1:nrow(cm), function(i) sum(cm[-i, -i]))
  FP <- colSums(cm) - diag(cm)
  FN <- rowSums(cm) - diag(cm)
  specificity <- TN / (TN + FP)        # TN / (TN + FP)
  
  # Calculate False Positive Rate (FPR)
  fpr <- FP / (FP + TN)                # FP / (FP + TN)
  
  # Calculate False Negative Rate (FNR)
  fnr <- FN / (FN + diag(cm))          # FN / (FN + TP)
  
  # Handle any NaN results (caused by division by zero)
  precision[is.na(precision)] <- 0
  recall[is.na(recall)] <- 0
  f1_score[is.na(f1_score)] <- 0
  specificity[is.na(specificity)] <- 0
  fpr[is.na(fpr)] <- 0
  fnr[is.na(fnr)] <- 0
  
  # Return all metrics in a data frame
  return(data.frame(
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score,
    Specificity = specificity,
    FPR = fpr,
    FNR = fnr,
    Support = support,
    Accuracy = rep(accuracy, length(precision))  # Accuracy is the same for all classes
  ))
}


# Confusion matrix for 'gi' predictor
cat("\nConfusion matrix and precision metrics for 'gi' predictor:\n")
conf_matrix_gi <- confusionMatrix(pred_gi, test_gi$emotions)
print(conf_matrix_gi)
metrics_gi <- calculate_metrics(conf_matrix_gi)
print(metrics_gi)

# Confusion matrix for 'he' predictor
cat("\nConfusion matrix and precision metrics for 'he' predictor:\n")
conf_matrix_he <- confusionMatrix(pred_he, test_he$emotions)
print(conf_matrix_he)
metrics_he <- calculate_metrics(conf_matrix_he)
print(metrics_he)

# Confusion matrix for 'qdap' predictor
cat("\nConfusion matrix and precision metrics for 'qdap' predictor:\n")
conf_matrix_qdap <- confusionMatrix(pred_qdap, test_qdap$emotions)
print(conf_matrix_qdap)
metrics_qdap <- calculate_metrics(conf_matrix_qdap)
print(metrics_qdap)

# Confusion matrix for 'blob' predictor
cat("\nConfusion matrix and precision metrics for 'blob' predictor:\n")
conf_matrix_blob <- confusionMatrix(pred_blob, test_blob$emotions)
print(conf_matrix_blob)
metrics_blob <- calculate_metrics(conf_matrix_blob)
print(metrics_blob)

# Confusion matrix for 'sentiment_total' predictor
cat("\nConfusion matrix and precision metrics for 'sentiment_total' predictor:\n")
conf_matrix_total <- confusionMatrix(pred_total, test_total$emotions)
print(conf_matrix_total)
metrics_total <- calculate_metrics(conf_matrix_total)
print(metrics_total)



# -------------------------- Combine Precision Metrics -------------------------------------

# Combine precision, recall, and F1 scores for all models
combined_metrics <- list(
  gi = metrics_gi,
  he = metrics_he,
  qdap = metrics_qdap,
  blob = metrics_blob,
  sentiment_total = metrics_total
)

# Convert the list to a data frame for easier comparison
combined_metrics_df <- do.call(rbind, lapply(combined_metrics, function(x) round(x, 2)))
combined_metrics_df$Model <- rep(c("gi", "he", "qdap", "blob", "sentiment_total"), each = nrow(metrics_gi))

# Print combined metrics
print(combined_metrics_df)


# Calculate macro-averages for precision, recall, and F1-score, specificity, FPR, FNR, support, and accuracy
macro_precision <- sapply(combined_metrics, function(x) mean(x$Precision))
macro_recall <- sapply(combined_metrics, function(x) mean(x$Recall))
macro_f1_scores <- sapply(combined_metrics, function(x) mean(x$F1_Score))
macro_specificity <- sapply(combined_metrics, function(x) mean(x$Specificity))
macro_fpr <- sapply(combined_metrics, function(x) mean(x$FPR))
macro_fnr <- sapply(combined_metrics, function(x) mean(x$FNR))
macro_accuracy <- sapply(combined_metrics, function(x) mean(x$Accuracy))



# Final plot --------------------------------------------------------------
display.brewer.all()

# Combine the precision, recall, F1 scores, and accuracy for the radar chart
radar_data <- data.frame(
  Model = c("gi", "he", "qdap", "blob", "sentiment_total"),
  Precision = c(0.1809524, 0.1047619, 0.1251701, 0.1428571, 0.1095238),  # Macro averaged precision values from each model
  Recall = c(0.05891669, 0.03654485, 0.07818533, 0.03576783, 0.08809524),     # Macro averaged recall values from each model
  F1_Score = c(0.05724508, 0.05418719, 0.08641359, 0.05324042 , 0.09230769),   # Macro averaged F1-Score values from each model
  Accuracy = c(0.1154, 0.2115, 0.2308, 0.1346, 0.2115), # Accuracy values for each model
  Specificity = c(0.8655186, 0.8303663, 0.8591750, 0.8534112, 0.8685030),
  FPR = c(0.1344814, 0.1696337, 0.1408250, 0.1465888, 0.1314970),
  FNR = c(0.2267976, 0.3920266, 0.7789575, 0.2499465, 0.9119048)
)

# Normalize data: Radar charts require a normalized range (typically 0 to 1)
radar_data_scaled <- as.data.frame(lapply(radar_data[,-1], function(x) (x - min(x)) / (max(x) - min(x))))
radar_data_scaled$Model <- radar_data$Model

# Add an extra row to represent the max and min of the scales
radar_data_scaled <- rbind(rep(1, ncol(radar_data_scaled) - 1), rep(0, ncol(radar_data_scaled) - 1), radar_data_scaled[,-ncol(radar_data_scaled)])

# Define the chart colors using RColorBrewer 
colors_border <- brewer.pal(5, "Set1") 
colors_fill <- adjustcolor(colors_border, alpha.f=0.2)

# Plot the radar chart
radarchart(radar_data_scaled, axistype = 1,
           # Customize the polygon and lines
           pcol = colors_border, pfcol = colors_fill, plwd = 2, plty = 1,
           
           # Customize the grid
           cglcol = "grey", cglty = 1, axislabcol = "darkgrey", caxislabels = seq(0, 1, 0.2), cglwd = 0.8,
           
           # Customize labels
           vlcex = 0.8
)

# Define custom labels for the legend
custom_labels <- c("GI", "HE", "QDAP", "Blob", "Total")

# Add legend with custom labels
legend(x = 0.7, y = 1.3, legend = custom_labels, bty = "n", pch = 20, col = colors_border, text.col = "black", cex = 0.9, pt.cex = 1.5)


