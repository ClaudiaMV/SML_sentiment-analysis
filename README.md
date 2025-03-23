[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Evaluating Sentiment Dictionary Predictors for Emotional Classification Using Conditional Inference Trees

## Overview  
This R script evaluates the predictive performance of five sentiment dictionaries—**GI**, **HE**, **QDAP**, **TextBlob**, and **VADER**—in classifying self-rated emotional valence within autobiographical narratives. The script performs data preprocessing, stratified oversampling, cost-sensitive learning via conditional inference trees (CTree), and evaluates model performance using various metrics and visualizations.

---

## 📁 Contents  

### 1. Library Loading  
Loads necessary packages using `pacman` and `library()`, including:
- Data manipulation: `dplyr`, `tidyr`, `reshape2`, `janitor`
- Modelling: `party`, `caret`, `MASS`, `ROSE`
- Visualization: `ggplot2`, `gridExtra`, `fmsb`, `RColorBrewer`
- Reporting: `knitr`, `kableExtra`

### 2. Data Loading & Cleaning  
- Imports four CSV files:  
  - `SentimentAnalysis_control.csv`  
  - `TextBlob_control.csv`  
  - `vader_control.csv`  
  - `tokens_emotion.csv`  
- Cleans column names and merges sentiment scores.
- Normalizes all sentiment scores by word count.
- Creates a new `sentiment_total` score by summing across all normalized scores.
- Converts the target variable (`emotions`) into an ordered factor.

### 3. Class Weighting (Before Oversampling)  
- Calculates log-scaled inverse class frequencies.
- Scales weights to match the number of observations.

### 4. Oversampling  
- Applies stratified oversampling to ensure each class contains 78 samples.
- Verifies dataset size after oversampling.

### 5. Adjusted Weights (After Oversampling)  
- Reassigns original class weights to the oversampled dataset.
- Converts them into positive integers for use with `ctree()`.

### 6. Train-Test Split  
- Splits the data into 80% training and 20% testing using `createDataPartition()`.
- Aligns sample weights with respective splits.
- Validates integrity of weight-to-data alignment.

### 7. Model Training  
- Trains separate CTree models for each sentiment predictor:
  - `sentiment_gi`
  - `sentiment_he`
  - `sentiment_qdap`
  - `sentiment_blob`
  - `sentiment_vader`
- Computes and prints classification accuracy.

### 8. Confusion Matrices  
- Generates and prints confusion matrices using `caret::confusionMatrix()`.
- Plots each confusion matrix as a heatmap using `ggplot2`.

### 9. Performance Metrics  
- Calculates for each model:
  - Precision
  - Recall
  - F1-Score
  - Specificity
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
  - Accuracy
- Outputs metrics as a combined data frame.

### 10. Visualization  
- Displays all decision trees (CTree structures).
- Creates a radar chart to compare dictionary performance across all metrics.

---

## 📦 Required Files  
Ensure the following CSV files are available in your working directory:

- `SentimentAnalysis_control.csv`
- `TextBlob_control.csv`
- `vader_control.csv`
- `tokens_emotion.csv`

---

## 🚀 How to Run  

1. Open R or RStudio.
2. Install required libraries using `install.packages()` or `pacman::p_load(...)`.
3. Set the working directory to the folder containing your CSVs.
4. Source the script or run it step-by-step.
5. Outputs will include:
   - Accuracy metrics
   - Confusion matrix tables and heatmaps
   - Visual decision trees
   - Radar chart comparing performance

---

## 📊 Outputs  

- **Accuracy Table**: Overall accuracy per model  
- **Confusion Matrices**: Printed and visual heatmaps  
- **Metric Table**: Precision, Recall, F1, etc.  
- **Radar Chart**: Visual comparison across all models  
- **CTree Plots**: Node structures per predictor  

---

## 📝 Notes  

- Emotion ratings (`emotions`) are transformed to a balanced 5-class scale and treated as an ordered factor.
- Oversampling is used to mitigate class imbalance, followed by reweighting to reflect the original distribution.
- Radar chart values are illustrative and based on hardcoded metrics; replace with updated results if needed.

---

## 🧠 Author  
Claudia Morales Valiente  
Psychology Department, University of Western Ontario

---

