# Data Mining Fundamentals Project 1: Credit Limit Prediction

This repository contains the documentation and code for the first project of the Data Mining Fundamentals course. The primary goal of this project is to design and implement a regression model to predict the financial credit limit (`Credit_Limit`) of bank customers based on their demographic features and financial behavior.

## Dataset Structure
The data used in this project is sourced from the `CreditPrediction.csv` file. The target variable in this problem is `Credit_Limit`.
- The descriptive variables include demographic features (e.g., age, gender, marital status, education level, and income category) as well as bank interaction indicators (e.g., card category, months on book, transaction counts, and transaction amounts).

## Data Preprocessing & Cleaning
To build a reliable and scientifically sound model, the following steps were applied to the raw data:
- **Preventing Data Leakage:** The `Avg_Utilization_Ratio` column was removed due to its hidden mathematical correlation with the target variable, ensuring the model is evaluated under real-world conditions. The `CLIENTNUM` column was also dropped as it holds no predictive value.
- **Missing Value Imputation:** To preserve the data distribution, missing values in numerical columns were filled using the **Median**, while categorical missing values were imputed using the **Mode**.
- **Outlier Handling:** Outliers in the input features were capped and controlled using the Interquartile Range (IQR) method to reduce their destructive impact on the training process. The target variable (`Credit_Limit`) was left untouched to preserve the essence of the problem.

## Feature Engineering
To extract hidden behavioral patterns of the customers, new features were engineered from the existing data:
- **`Avg_Transaction_Amount`:** Created by dividing the total transaction amount by the transaction count, this feature represents the average value of each transaction (the customer's spending pattern).
- **`Transactions_per_Month`:** Calculated using the total transaction count and the months of activity, this quantifies the customer's average monthly transactions, providing a more accurate measure of their continuous engagement with the bank.

## Categorical Encoding
Given the different nature of categorical data, distinct encoding strategies were adopted:
- **Ordinal Encoding:** Features such as `Education_Level`, `Income_Category`, and `Card_Category` possess an inherent order and value (e.g., a Platinum card is higher than a Blue card). By using Ordinal Encoding, the concept of "order and magnitude" is preserved, helping tree-based models find more logical splits.
- **One-Hot Encoding:** Applied to nominal variables lacking any inherent priority (like `Gender` and `Marital_Status`) to prevent the algorithm from incorrectly assuming an ordinal relationship between unrelated categories.

## Pre-Modeling Preparation
- **Data Splitting:** The dataset was divided into Training (80%) and Testing (20%) sets using a standard split.
- **Feature Scaling:** Numerical feature values were standardized using `StandardScaler` so that scale differences do not cause computational errors in models like Linear Regression.

## Modeling & Final Evaluation
In this project, three different algorithms were developed and compared:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Best Model & Numerical Results
Based on the evaluations, the **Random Forest** algorithm achieved the best performance:
- **R² Score:** 0.5985
- **RMSE:** 5862.27
- **Performance Analysis:** The evaluation demonstrated that the linear model failed to capture the complex patterns in the data. By aggregating multiple decision trees, the Random Forest model achieved a better balance and delivered realistic, scientifically robust performance, even after removing data-leaking features.

### Feature Importance Analysis
Extracting feature importance from the Random Forest model revealed that `Income_Category` and `Card_Category` played the most significant roles in accurately predicting the customers' credit limits.