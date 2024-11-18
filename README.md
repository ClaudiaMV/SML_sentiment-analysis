# ctree_SML_sentiment
Conditional inference trees to compare polarity predictability of several sentiment dictionaries. The sentiment analysis is not performed in this code.

## Overview

This script performs sentiment analysis using various predictors and evaluates their performance in predicting ordered emotion classes. The key steps include data loading and preprocessing, oversampling to balance class frequencies, applying cost-sensitive learning, training conditional inference tree models (`ctree`), and evaluating model performance with accuracy, precision, recall, F1-scores, specificity, false positive/negative rates, and confusion matrices. Additionally, a radar chart is generated for visual comparison of model metrics.

---

## Key Features

1. **Libraries Used**:
   - Data Manipulation: `dplyr`, `janitor`
   - Visualization: `ggplot2`, `fmsb`, `gridExtra`, `RColorBrewer`
   - Modelling: `party`, `caret`, `ROSE`
   - Formatting: `knitr`, `kableExtra`

2. **Data Preprocessing**:
   - Datasets loaded: `SentimentAnalysis_control.csv`, `TextBlob_control.csv`, and `tokens_emotion.csv`.
   - Normalized sentiment scores (`gi`, `he`, `qdap`, `blob`) based on word counts.
   - Added a composite sentiment score (`sentiment_total`).
   - Ensured uniform column naming using `janitor`.

3. **Class Balancing**:
   - Class weights computed inversely proportional to class frequencies for cost-sensitive learning.
   - Oversampling performed to ensure uniform class distributions.

4. **Data Splitting**:
   - Dataset split into 80% training and 20% testing sets.

5. **Model Training**:
   - Conditional inference trees (`ctree`) trained for each predictor (`gi`, `he`, `qdap`, `blob`, `sentiment_total`) with cost-sensitive weights.

6. **Evaluation Metrics**:
   - Accuracy, precision, recall, F1-scores, specificity, false positive rate (FPR), false negative rate (FNR), and support calculated for each predictor.
   - Confusion matrices generated and visualized as heatmaps.

7. **Visualization**:
   - Radar chart illustrating model performance across precision, recall, F1-scores, and accuracy.

---

## Script Workflow

### 1. **Setup**
   - Required libraries are loaded.
   - Datasets (`sent`, `blob`, `emotions`) are loaded and cleaned.

### 2. **Data Preprocessing**
   - Sentiment scores normalized.
   - New variables created for predictors and the dependent variable (`emotions`).

### 3. **Balancing and Sampling**
   - Weights assigned based on class frequencies for cost-sensitive learning.
   - Oversampling performed to create balanced classes.

### 4. **Train-Test Splitting**
   - Training and test datasets created using an 80-20 split.

### 5. **Model Training and Evaluation**
   - Separate models trained for each predictor using `ctree`.
   - Predictions compared with actual values in the test set.
   - Confusion matrices generated and evaluated for each model.

### 6. **Metric Calculation**
   - Precision, recall, F1-scores, specificity, FPR, and FNR calculated for all predictors.
   - Combined metrics presented in a tabular format.

### 7. **Visualization**
   - Heatmaps plotted for confusion matrices.
   - Radar chart created to compare model metrics.

---

## Files Required

- **Input Datasets**:
  - `SentimentAnalysis_control.csv` database with the data from the sentiment analysis
  - `TextBlob_control.csv` database with the data from the sentiment analysis
  - `tokens_emotion.csv` database with the data for training and testing

- **Output**:
  - Combined metrics table
  - Confusion matrix visualizations
  - Radar chart comparing model performance

---

## How to Run

1. Place the input CSV files in the working directory.
2. Install the required R packages.
3. Run the script in R or RStudio.
4. Review the printed metrics, confusion matrices, and visualizations.

---

## Results

- Model performance is summarized in a radar chart and confusion matrix heatmaps.
- Accuracy, precision, recall, and F1-scores provide insight into the strengths and weaknesses of each predictor.

---

## Dependencies

Ensure the following R packages are installed:
- `dplyr`, `janitor`, `party`, `caret`, `MASS`, `ggplot2`, `reshape2`, `fmsb`, `gridExtra`, `RColorBrewer`, `ROSE`, `knitr`, `kableExtra`.

---

## Author

Claudia Morales Valiente  
Cognitive Developmental and Brain Science Program  
University of Western Ontario
cmorale7@uwo.ca
