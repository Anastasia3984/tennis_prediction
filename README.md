# Tennis match prediction

An ML project focused on predicting the outcomes of professional tennis matches using historical data from 2006 to 2024

## Overview

This project applies supervised machine learning models to forecast the winner of ATP tennis matches based on match statistics, player seedings, surface types, and more. The primary goal is to explore sports analytics using structured datasets and classification techniques.

## Dataset

- Source: [Jeff Sackmann’s ATP tennis data](https://github.com/JeffSackmann/tennis_atp)
- Files used: match results from 2006 to 2024
- Preprocessing included:
  - Filtering incomplete matches
  - Handling categorical features like surface and player seed
  - Feature engineering for player stats and historical trends

## Technologies used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Random Forest, SVM, Logistic Regression
- Seaborn, Matplotlib
- Google Colab


## ML models and approach

Several models were trained and evaluated:
- **XGBoost Classifier**
- **Random Forest**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Stacking Ensemble (Logistic Regression + SVM + XGBoost)**

Data preprocessing steps:
- Standard scaling for numeric features
- One-hot encoding for categorical variables
- Train/test split
- Cross-validation and performance comparison

## Results

- The best-performing model was **XGBoost**, integrated into a stacking ensemble.
- Achieved an accuracy of **around 76%** on the test set.
- Model performance metrics:
  - `Accuracy`: 76%
  - `ROC AUC`: 0.82
  - Strong performance on predicting top-seeded match outcomes
- Confusion matrix and classification report showed that the model performs significantly better than random guessing.

## Files

- `teniss_prediction.ipynb` – main notebook with data preprocessing, model training, evaluation, and results

## Future improvements

- Include more player-specific performance stats (e.g., serve success rate)
- Integrate Elo rating or ATP ranking trends
- Expand feature set with recent match history
- Deploy the model as a simple web app or API
