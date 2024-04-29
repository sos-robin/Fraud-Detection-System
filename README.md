Fraud Detection System using Machine Learning Model
Introduction

The Fraud Detection System aims to safeguard financial systems and protect customers from fraudulent activities in banking. In today's technologically advanced world, financial fraud has become more sophisticated, posing challenges for banks and financial institutions. Leveraging machine learning techniques can effectively identify and prevent fraudulent transactions by analyzing large volumes of data and detecting anomalous patterns and behavior.
Dataset Overview

    Source: Transactions made by European cardholders in September 2013
    Content: Contains transactions over two days, with 492 frauds out of 284,807 transactions
    Features: Principal components obtained with PCA transformation, 'Time', and 'Amount'
    Target Variable: 'Class' indicating fraud (1) or normal (0) transactions
    Imbalance: Highly imbalanced dataset with frauds accounting for 0.172% of all transactions

Data Preprocessing

    Imported data using Pandas, checked data information, and handled missing values
    Scaled 'Amount' column using StandardScaler
    Dropped 'Time' column and removed duplicate values
    Explored distribution of target variable ('Class')

Model Selection and Evaluation

    Explored Logistic Regression, Decision Tree Classifier, and Random Forest Classifier
    Oversampled using SMOTE to handle class imbalance
    Split dataset into training and testing sets
    Evaluated models using accuracy, precision, recall, and F1-score
    Chose Random Forest Classifier as the best model based on performance metrics

Model Training and Evaluation

    Trained Random Forest Classifier on the entire dataset after oversampling
    Evaluated model using accuracy, precision, recall, and F1-score
    Checked confusion matrix and AUC-ROC score for model performance

Simulation and Model Implementation

    Saved the trained model using joblib
    Implemented a GUI using tkinter for user input and prediction

Appendices

    Included additional information such as references and dataset source

References

    Kaggle: Credit Card Fraud Detection Dataset
    Scikit-learn Documentation
    XGBoost Documentation
    Various research papers and books on fraud detection and machine learning
