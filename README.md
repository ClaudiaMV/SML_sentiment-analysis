[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
README
Title:

Evaluating Sentiment Dictionary Predictors for Emotional Classification Using Conditional Inference Trees
Overview

This R script evaluates the predictive validity of five sentiment dictionaries (GI, HE, QDAP, TextBlob, VADER) in classifying self-rated emotional valence in a dataset of autobiographical texts. The workflow includes preprocessing, oversampling, weighting, model training using conditional inference trees (CTree), and evaluation via classification metrics and visualization.
Contents

The script is structured into the following sections:

    Library Loading
    Loads all required packages for data manipulation, modelling, and visualization (e.g., dplyr, caret, party, fmsb, etc.).

    Data Loading and Cleaning

        Imports four datasets: SentimentAnalysis_control.csv, TextBlob_control.csv, vader_control.csv, and tokens_emotion.csv.

        Standardizes column names and merges sentiment scores from different tools into a unified dataframe sent.

        Sentiment scores are normalized by word count, and a combined score (sentiment_total) is computed.

    Class Weight Computation

        Calculates log-scaled inverse class frequencies to handle class imbalance before oversampling.

    Oversampling

        Applies stratified oversampling to balance the class distribution (target frequency: 78 per class).

        Ensures the integrity of the dataset after oversampling.

    Adjusted Weight Calculation

        Computes adjusted weights for each sample to reflect original class importance post-oversampling.

        These weights are used in cost-sensitive learning with CTree.

    Train-Test Split

        Splits the dataset into 80% training and 20% test sets.

        Aligns sample weights accordingly and performs integrity checks.

    Model Training

        Trains CTree models separately for each sentiment predictor (GI, HE, QDAP, Blob, VADER, Total).

        Evaluates each model’s classification accuracy on the test set.

    Confusion Matrices

        Generates and prints confusion matrices for each model.

        Visualizes them as heatmaps using ggplot2.

    Performance Metrics

        Computes precision, recall, F1-score, specificity, false positive rate (FPR), false negative rate (FNR), and accuracy from the confusion matrices.

    Visualization

        Plots all confusion matrices as heatmaps.

        Visualizes each CTree model structure.

        Generates a radar chart to compare performance metrics across models.

Files Required

Make sure the following files are in your working directory:

    SentimentAnalysis_control.csv

    TextBlob_control.csv

    vader_control.csv

    tokens_emotion.csv

How to Run

    Open R or RStudio.

    Ensure all required packages are installed.

    Set the working directory to the folder containing the input .csv files.

    Run the script line by line or as a whole.

    The output will include:

        Printed accuracy and metric tables.

        Visual plots of confusion matrices and decision trees.

        A radar chart comparing dictionary performance.

Outputs

    Accuracy summary table for each dictionary-based model.

    Confusion matrices and classification reports.

    Radar chart summarizing key evaluation metrics.

    Visual representation of each decision tree used.

Notes

    The emotional valence variable (emotions) is treated as an ordered factor.

    Oversampling is performed with replacement to ensure class balance.

    Integer weights are used for compatibility with ctree() from the party package.

    Radar chart metrics are manually defined for illustration and can be updated with calculated values.
