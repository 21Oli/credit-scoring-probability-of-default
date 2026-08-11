# Credit Scoring: Probability of Default Prediction

## Overview

This project develops a **Probability of Default (PD) prediction model** using machine learning techniques for credit scoring. The objective is to identify customers who are likely to default on their credit card payments in the following month while reducing model complexity through a sequential feature selection pipeline.

Starting with **23 predictor variables**, the project applies correlation filtering, Information Value (IV) analysis, and SHAP feature importance to obtain a compact set of **15 highly informative features**. The selected features are then used to train and compare six machine learning algorithms.

---

## Objectives

* Predict customer default probability.
* Reduce redundant and less informative features.
* Evaluate the effect of feature selection on LightGBM performance.
* Compare multiple machine learning algorithms using the final selected features.
* Identify the best-performing model for credit scoring.

---

## Dataset

* **Dataset:** Default of Credit Card Clients
* **Samples:** 30,000 customers
* **Target Variable:** Default payment next month
* **Original Features:** 23
* **Final Features:** 15

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Data inspection
* Missing value analysis
* Data types
* Target distribution
* Correlation analysis

### 2. Data Preparation

* 80% Training Set
* 20% Testing Set
* Stratified train-test split
* Random state = 42

Feature selection was performed **only on the training data** to prevent data leakage.

### 3. Feature Selection

#### Step A — Correlation Filtering

* Pearson Correlation Matrix
* Threshold: **|r| ≥ 0.85**
* Compared correlated feature pairs using individual LightGBM AUC
* Removed the less predictive feature

**Result**

* 23 → 19 Features

#### Step B — Information Value (IV)

* Computed Information Value for every remaining feature
* Removed features with **IV ≤ 0**

**Result**

* No features removed
* Remaining Features: **19**

#### Step C — SHAP Feature Selection

* Trained a LightGBM model
* Computed SHAP values
* Ranked features by Mean Absolute SHAP Value
* Selected the Top 15 features

**Final Feature Set:** 15 Features

---

## LightGBM Performance During Feature Selection

| Stage             | Features | Train AUC | Test AUC |
| ----------------- | -------: | --------: | -------: |
| Original          |       23 |    0.9047 |   0.7726 |
| After Correlation |       19 |    0.9014 |   0.7705 |
| After IV          |       19 |    0.9014 |   0.7705 |
| After SHAP        |       15 |    0.8964 |   0.7693 |

Feature selection reduced the number of predictors from **23 to 15** while decreasing the Test AUC by only **0.0033**, demonstrating that most predictive information was preserved.

---

## Models Compared

* Logistic Regression
* LightGBM
* CatBoost
* XGBoost
* AdaBoost
* Random Forest

Hyperparameter tuning was performed using **GridSearchCV** with **5-fold Cross-Validation** and **ROC-AUC** as the optimization metric.

---

## Final Results

| Rank | Model               | Train AUC |   Test AUC |
| ---- | ------------------- | --------: | ---------: |
| 🥇   | CatBoost            |    0.8171 | **0.7713** |
| 🥈   | XGBoost             |    0.8162 |     0.7704 |
| 🥉   | LightGBM            |    0.8165 |     0.7697 |
| 4    | Random Forest       |    0.8509 |     0.7683 |
| 5    | AdaBoost            |    0.7835 |     0.7638 |
| 6    | Logistic Regression |    0.7241 |     0.7155 |

CatBoost achieved the highest Test AUC, although the performance differences among CatBoost, XGBoost, and LightGBM were very small.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost
* SHAP
* ScorecardPy

---

## Key Skills Demonstrated

* Credit Risk Modeling
* Probability of Default Prediction
* Feature Engineering
* Feature Selection
* Correlation Analysis
* Information Value (IV)
* SHAP Explainability
* Hyperparameter Tuning
* GridSearchCV
* ROC-AUC Evaluation
* Model Comparison
* Explainable AI (XAI)

---

## Future Improvements

* Bayesian Hyperparameter Optimization using Optuna
* Probability Calibration
* Scorecard Development
* Weight of Evidence (WOE) Transformation
* Class Imbalance Techniques (SMOTE and Cost-sensitive Learning)
* Model Deployment with Streamlit or FastAPI

---

## Author

**Oli Bakala**

Aspiring Data Scientist | Machine Learning Engineer | AI Engineer


