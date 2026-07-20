<div align="center">

# Credit Limit Prediction with Regression Models

### A data mining project for predicting bank customers' credit limits

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-success)

</div>

---

## Overview

This project develops and compares multiple regression models to predict the credit limit assigned to a bank customer.

The dataset contains demographic information, account characteristics, card details, transaction behavior, and customer activity records. The target variable is:

```text
Credit_Limit
```

The project covers the main stages of a standard data mining workflow:

- Data loading and cleaning
- Duplicate removal
- Missing-value treatment
- Feature engineering
- Categorical encoding
- Outlier treatment
- Correlation analysis
- Feature scaling
- Regression model training
- Performance evaluation
- Feature importance analysis
- Visual comparison of actual and predicted values

The final comparison shows that the **Random Forest Regressor** provides the best performance among the evaluated models.

---

## Project Objective

The main objective is to estimate a customer's credit limit using the available banking and demographic features.

This is a supervised machine learning problem because:

- The dataset contains known credit-limit values.
- `Credit_Limit` is a continuous numerical target.
- Regression algorithms are therefore used.

---

## Dataset

The original dataset contains:

| Item | Value |
|---|---:|
| Rows | 10,167 |
| Columns | 20 |
| Fully duplicated rows | 35 |
| Duplicated customer identifiers | 40 |
| Target column | `Credit_Limit` |

After removing duplicated customers and unnecessary columns, the cleaned dataset contains:

| Item | Value |
|---|---:|
| Rows | 10,127 |
| Retained original columns | 17 |
| Training samples | 8,101 |
| Testing samples | 2,026 |
| Final model features | 18 |

### Main feature groups

The dataset includes:

- Customer age and gender
- Number of dependents
- Education level
- Marital status
- Income category
- Card category
- Length of relationship with the bank
- Number of banking products
- Inactive months
- Number of customer contacts
- Revolving balance
- Transaction amount and count
- Changes in transaction behavior

---

## Data Cleaning

The notebook performs several initial cleaning operations.

### Empty column removal

The column below contains no useful values and is removed:

```text
Unnamed: 19
```

### Duplicate removal

Two duplicate checks are applied:

1. Fully duplicated rows are removed.
2. Repeated `CLIENTNUM` values are removed while keeping the first record.

This reduces the possibility of having the same customer in both the training and testing sets.

### Identifier removal

`CLIENTNUM` is removed after duplicate checking because it is only a unique identifier and does not represent a meaningful predictive feature.

### Target leakage prevention

`Avg_Utilization_Ratio` is removed because it has a direct mathematical relationship with the target variable:

```text
Avg_Utilization_Ratio ≈ Total_Revolving_Bal / Credit_Limit
```

Keeping this feature could allow the model to indirectly access information derived from the target.

---

## Missing Values

Missing values are handled separately for numerical and categorical columns.

### Numerical columns

Missing numerical values are filled using the median calculated from the training data.

The median is used because it is less sensitive to extreme values than the mean.

### Categorical columns

Missing categorical values are filled using the most frequent category, also called the mode.

The replacement values are learned only from the training set and then applied to the testing set to reduce data leakage.

The original cleaned data contains missing values in columns such as:

- `Gender`
- `Marital_Status`
- `Card_Category`
- `Months_on_book`
- `Total_Relationship_Count`

---

## Feature Engineering

Four additional features are created from the original banking variables.

### Average transaction amount

```text
Avg_Transaction_Amount =
Total_Trans_Amt / Total_Trans_Ct
```

This feature represents the average amount spent per transaction.

### Transactions per month

```text
Transactions_per_Month =
Total_Trans_Ct / Months_on_book
```

This measures the customer's average monthly transaction frequency.

### Transaction amount per month

```text
Amount_per_Month =
Total_Trans_Amt / Months_on_book
```

This represents the customer's average monthly transaction volume.

### Inactive rate

```text
Inactive_Rate =
Months_Inactive_12_mon / 12
```

This feature represents the proportion of inactive months during the last year.

Zero denominators are replaced with missing values before division, and infinite values are converted to missing values for safe processing.

---

## Categorical Encoding

Machine learning algorithms require numerical input, so categorical variables are converted into numerical form.

### Ordinal encoding

Ordered categories are mapped to increasing numerical values.

The following columns use ordinal encoding:

- `Education_Level`
- `Income_Category`
- `Card_Category`

For example, card categories are represented in this order:

```text
Blue < Silver < Gold < Platinum
```

### One-hot encoding

Categorical variables without a natural order are transformed using one-hot encoding.

Examples include:

- `Gender`
- `Marital_Status`

The training and testing DataFrames are aligned after encoding to guarantee identical columns and column order.

---

## Outlier Treatment

The notebook supports three outlier-processing methods through the `OUTLIER_METHOD` variable.

The currently selected method is:

```python
OUTLIER_METHOD = "IQR"
```

### 1. Interquartile Range

The IQR method uses the first and third quartiles:

```text
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

Values outside these limits are clipped to the calculated boundaries.

This is the active method because it is robust for skewed numerical data.

### 2. Z-Score

The Z-Score option creates boundaries using three standard deviations from the mean:

```text
Lower bound = Mean - 3 × Standard Deviation
Upper bound = Mean + 3 × Standard Deviation
```

Values beyond these limits are clipped.

### 3. Isolation Forest

Isolation Forest is an anomaly-detection algorithm that identifies observations that are easier to isolate from the rest of the dataset.

In this implementation:

- The expected outlier proportion is set to 5%.
- Detected anomalous numerical values are replaced with the training median.

All outlier boundaries or models are learned from the training data and then applied to the testing data.

---

## Correlation Analysis

A correlation matrix is calculated for the independent training features.

The absolute correlation value is used so that both strong positive and strong negative relationships can be detected.

Features with an absolute correlation greater than `0.85` are removed to reduce redundant information.

The following features are removed by the current notebook:

```text
Total_Trans_Ct
Avg_Transaction_Amount
Amount_per_Month
Inactive_Rate
```

A heatmap is also generated to visually inspect the relationships between features.

---

## Feature Scaling

The notebook supports two scaling methods:

- `StandardScaler`
- `MinMaxScaler`

The current configuration uses:

```python
SCALING_METHOD = "Standard"
```

`StandardScaler` transforms each feature so that it has approximately:

- Mean equal to 0
- Standard deviation equal to 1

The scaler is fitted on the training data and only transformed on the testing data.

Scaling is especially important for distance-based algorithms such as K-Nearest Neighbors.

---

## Regression Models

Four regression algorithms are trained and compared.

### Linear Regression

Linear Regression estimates a linear relationship between the input features and the target.

It is simple, fast, and useful as a baseline model.

### Decision Tree Regressor

Decision Tree creates a tree of decision rules and can model nonlinear relationships.

It is easy to interpret but may overfit the training data.

### Random Forest Regressor

Random Forest combines predictions from multiple decision trees.

The notebook uses:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Combining many trees generally improves stability and reduces overfitting compared with one decision tree.

### K-Nearest Neighbors Regressor

KNN predicts a value using the nearest training samples.

The notebook uses:

```python
KNeighborsRegressor(n_neighbors=5)
```

Because KNN relies on distances, feature scaling is important for this algorithm.

---

## Evaluation Metrics

The models are evaluated using four regression metrics.

### Mean Absolute Error

MAE is the average absolute difference between actual and predicted values.

Lower values are better.

### Mean Squared Error

MSE is the average squared prediction error.

It gives more weight to large errors.

### Root Mean Squared Error

RMSE is the square root of MSE and has the same unit as the target variable.

Lower values are better.

### R² Score

R² measures the proportion of variation in `Credit_Limit` explained by the model.

Higher values are better, and a value closer to 1 indicates stronger performance.

---

## Model Results

The current notebook produces the following test-set results:

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| **Random Forest** | **0.5910** | **5,916.16** | **3,891.83** |
| Linear Regression | 0.4336 | 6,962.23 | 4,949.88 |
| K-Nearest Neighbors | 0.4028 | 7,149.06 | 4,748.14 |
| Decision Tree | 0.1516 | 8,521.44 | 5,188.68 |

### Selected model

Random Forest is selected as the best model because it achieves:

- The highest R² score
- The lowest RMSE
- The lowest MAE

The model explains approximately **59.1%** of the variation in customer credit limits on the current test split.

---

## Feature Importance

The Random Forest model is used to estimate the importance of the input features.

The five most important features in the current run are:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Income_Category` | 0.3374 |
| 2 | `Card_Category` | 0.1831 |
| 3 | `Total_Trans_Amt` | 0.0823 |
| 4 | `Total_Amt_Chng_Q4_Q1` | 0.0569 |
| 5 | `Total_Ct_Chng_Q4_Q1` | 0.0535 |

The results suggest that income category and card category have the strongest influence on the predicted credit limit.

---

## Visualizations

The notebook generates several visual outputs.

### Correlation heatmap

Displays the relationships between the independent features.

### Feature importance chart

Shows the five most influential variables identified by Random Forest.

### Actual versus predicted plots

For every regression model, actual credit limits are plotted against predicted values.

The red dashed line represents an ideal prediction where:

```text
Predicted value = Actual value
```

Points closer to this line indicate more accurate predictions.

### Error connection plots

For the first 50 test samples:

- Actual values are displayed as red crosses.
- Predicted values are displayed as blue circles.
- Dashed vertical lines show the prediction error for each sample.

These plots provide a direct visual comparison of model behavior.

---

## Project Structure

```text
credit-limit-prediction/
│
├── CreditPrediction.csv
├── Untitled-1(1).ipynb
└── README.md
```

For a cleaner repository, the notebook may optionally be renamed to:

```text
credit_limit_prediction.ipynb
```

---

## Requirements

Recommended environment:

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab

Required Python packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd credit-limit-prediction
```

### 2. Install the dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Untitled-1(1).ipynb
```

### 5. Run all cells

Make sure `CreditPrediction.csv` is located in the same directory as the notebook.

Then use:

```text
Kernel → Restart & Run All
```

---

## Reproducibility

The project uses:

```python
random_state=42
```

for train-test splitting, Decision Tree, Random Forest, and Isolation Forest.

This makes the results more reproducible across repeated executions using the same library versions and dataset.

---

## Current Limitations

- Model performance is based on one train-test split.
- Hyperparameter tuning is not included.
- Correlated features are removed using a fixed threshold.
- The target variable has a wide and right-skewed distribution.
- Random Forest feature importance may be affected by feature type and variability.
- The project does not currently save the trained model to disk.
- The current results may change with different random seeds or package versions.

---

## Possible Future Improvements

Potential improvements include:

- Cross-validation for more stable evaluation
- Hyperparameter tuning for Random Forest
- Testing Gradient Boosting or Extra Trees
- Comparing different outlier methods
- Evaluating the effect of removing correlated features
- Applying a logarithmic transformation to the target
- Adding residual-distribution plots
- Saving the final model and scaler with `joblib`
- Building a simple interface for entering customer information

---

## Academic Context

This repository was developed as an individual project for a Fundamentals of Data Mining course.

Its main purpose is to demonstrate a complete regression workflow, including preprocessing, feature engineering, model comparison, evaluation, and interpretation.

---

<div align="center">

Developed for educational and academic purposes.

</div>
