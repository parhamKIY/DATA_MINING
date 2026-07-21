# Credit Limit Prediction Using Regression Algorithms

A complete machine-learning regression project for predicting a bank customer's **credit card limit** from demographic information, account activity, and transaction behavior.

The project covers the full data-mining workflow: data cleaning, missing-value imputation, feature engineering, categorical encoding, outlier treatment, correlation analysis, feature scaling, model training, evaluation, visualization, and feature-importance analysis.

---

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Missing-Value Handling](#missing-value-handling)
- [Feature Engineering](#feature-engineering)
- [Outlier Detection and Treatment](#outlier-detection-and-treatment)
- [Feature Encoding](#feature-encoding)
- [Correlation Analysis](#correlation-analysis)
- [Feature Scaling](#feature-scaling)
- [Regression Models](#regression-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Challenges and Future Improvements](#challenges-and-future-improvements)

---

## Project Objective

The goal is to predict the target variable:

```text
Credit_Limit
```

The prediction is based on customer characteristics such as:

- Age and gender
- Education and marital status
- Income category
- Card category
- Length of relationship with the bank
- Number of products held
- Account inactivity and contact frequency
- Revolving balance
- Transaction amount and count
- Changes in transaction behavior

Several regression algorithms are trained and compared to identify the most suitable model for this dataset.

---

## Dataset

The notebook expects the following CSV file in the same directory:

```text
CreditPrediction.csv
```

### Main Columns

| Group | Features |
|---|---|
| Customer information | `CLIENTNUM`, `Customer_Age`, `Gender`, `Dependent_count` |
| Demographic information | `Education_Level`, `Marital_Status`, `Income_Category` |
| Banking product | `Card_Category` |
| Relationship information | `Months_on_book`, `Total_Relationship_Count` |
| Customer activity | `Months_Inactive_12_mon`, `Contacts_Count_12_mon` |
| Financial behavior | `Total_Revolving_Bal`, `Total_Trans_Amt`, `Total_Trans_Ct` |
| Behavioral changes | `Total_Amt_Chng_Q4_Q1`, `Total_Ct_Chng_Q4_Q1` |
| Target | `Credit_Limit` |

### Removed Columns

The following columns are removed during preprocessing:

- `Unnamed: 19`: empty or unnecessary column
- `CLIENTNUM`: unique customer identifier with no useful predictive meaning
- `Avg_Utilization_Ratio`: removed to prevent target leakage because it has a direct mathematical relationship with `Credit_Limit`

Duplicate rows and repeated customer identifiers are also removed.

---

## Project Workflow

```text
Load Dataset
     ↓
Initial Data Cleaning
     ↓
Train/Test Split
     ↓
Missing-Value Imputation
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Outlier Treatment
     ↓
Correlation Analysis
     ↓
Feature Scaling
     ↓
Train Regression Models
     ↓
Evaluate and Compare Models
     ↓
Feature Importance and Visual Analysis
```

The dataset is split before learning preprocessing parameters. This prevents information from the test set from leaking into the training process.

---

## Data Preprocessing

The preprocessing stage includes:

- Removing empty columns
- Removing duplicate records
- Removing duplicate customer IDs
- Dropping identifier columns
- Removing leakage-prone features
- Replacing infinite values with missing values
- Splitting data into training and testing sets
- Filling missing values
- Encoding categorical variables
- Treating outliers
- Removing highly correlated features
- Scaling numerical inputs

The train/test split uses:

```python
test_size=0.2
random_state=42
```

This means 80% of the data is used for training and 20% for testing.

---

## Missing-Value Handling

Three missing-value handling methods are available.

### 1. Median and Mode Imputation

For numerical features, missing values are replaced with the median calculated from the training set.

For categorical features, missing values are replaced with the most frequent category, or mode.

**Advantages**

- Simple and fast
- Easy to interpret
- Resistant to numerical outliers
- Suitable as a baseline

**Limitations**

- Does not use relationships between features
- May reduce the natural variation of the data

### 2. KNN Imputer

`KNNImputer` estimates missing numerical values using the most similar customers in the training data.

The implementation uses five neighbors. With distance-based weighting, closer neighbors have a greater effect on the estimated value.

**Advantages**

- Uses similarity between customers
- Considers relationships among numerical variables
- Can create more realistic values than median filling

**Limitations**

- More computationally expensive
- Sensitive to feature scales
- May perform poorly when many values are missing

Categorical missing values are still filled using the mode.

### 3. MICE

MICE stands for **Multiple Imputation by Chained Equations**.

The project uses a MICE-style iterative approach through Scikit-learn's `IterativeImputer`. Each numerical feature containing missing values is temporarily predicted from the other numerical features, and the process is repeated for several iterations.

**Advantages**

- Uses relationships among multiple features
- Can preserve correlations between variables
- Produces more informed estimates

**Limitations**

- Slower and more complex
- Depends on the estimator and iteration count
- Cannot directly process categorical columns in this implementation

Categorical missing values are handled separately using the mode.

### Data-Leakage Prevention

For every imputation method:

1. The imputer is fitted only on `X_train`.
2. The fitted imputer transforms `X_train`.
3. The same fitted imputer transforms `X_test`.

No replacement values are learned from the test set.

---

## Feature Engineering

Four behavioral features are created from the original columns.

### Average Transaction Amount

```text
Avg_Transaction_Amount = Total_Trans_Amt / Total_Trans_Ct
```

Represents the average amount spent per transaction.

### Transactions per Month

```text
Transactions_per_Month = Total_Trans_Ct / Months_on_book
```

Represents the customer's average transaction frequency.

### Transaction Amount per Month

```text
Amount_per_Month = Total_Trans_Amt / Months_on_book
```

Represents average monthly transaction volume.

### Inactive Rate

```text
Inactive_Rate = Months_Inactive_12_mon / 12
```

Represents the proportion of inactive months during the last year.

Division by zero is handled by temporarily replacing zero denominators with missing values. Infinite outputs are then converted to missing values and processed by the selected imputation method.

---

## Outlier Detection and Treatment

Three configurable methods are included.

### 1. Interquartile Range

The IQR method calculates the first and third quartiles:

```text
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside the bounds are clipped to the nearest boundary.

This method is robust for skewed financial data.

### 2. Z-Score

The Z-score approach uses three standard deviations from the training mean:

```text
Lower Bound = Mean - 3 × Standard Deviation
Upper Bound = Mean + 3 × Standard Deviation
```

Values outside this range are clipped.

This approach works best when feature distributions are approximately normal.

### 3. Isolation Forest

Isolation Forest is an unsupervised anomaly-detection algorithm.

It isolates unusual observations through random partitions. Samples detected as anomalies are treated using median values from the training set.

The configured contamination value is:

```python
contamination=0.05
```

This represents an expected outlier proportion of approximately 5%.

### Important Detail

Outlier statistics and models are learned from the training set. The resulting rules are then applied to the test set to reduce data leakage.

The target variable is not included in outlier treatment.

---

## Feature Encoding

### Ordinal Encoding

Ordered categorical features are manually mapped to numerical values.

#### Education Level

```text
Unknown < Uneducated < High School < College
< Graduate < Post-Graduate < Doctorate
```

#### Income Category

```text
Unknown < Less than $40K < $40K-$60K
< $60K-$80K < $80K-$120K < $120K+
```

#### Card Category

```text
Blue < Silver < Gold < Platinum
```

### One-Hot Encoding

Nominal variables without a meaningful order, such as gender and marital status, are converted using one-hot encoding.

```python
pd.get_dummies(..., drop_first=True)
```

Training and testing columns are aligned after encoding to guarantee identical feature structures.

---

## Correlation Analysis

A correlation matrix is calculated from the processed training features and visualized using a heatmap.

Features with an absolute pairwise correlation greater than `0.85` are identified as highly correlated and removed.

In the saved notebook run, the following features were removed:

- `Total_Trans_Ct`
- `Avg_Transaction_Amount`
- `Amount_per_Month`
- `Inactive_Rate`

This step reduces redundant information and may improve model stability.

---

## Feature Scaling

Two scaling methods are supported.

### StandardScaler

Transforms each feature to approximately zero mean and unit variance.

```python
SCALING_METHOD = "Standard"
```

### MinMaxScaler

Transforms features into a fixed range, usually from 0 to 1.

```python
SCALING_METHOD = "MinMax"
```

The scaler is fitted only on the training data and then applied to the test data.

Scaling is especially important for:

- Linear Regression
- K-Nearest Neighbors

Tree-based models generally do not require scaling, but a shared scaled feature matrix is used to provide a consistent comparison.

---

## Regression Models

Four regression algorithms are compared.

### Linear Regression

A simple and interpretable baseline model that assumes a linear relationship between predictors and the target.

### Decision Tree Regressor

A nonlinear tree-based model that divides data using decision rules.

### Random Forest Regressor

An ensemble of multiple decision trees. Predictions are calculated by averaging tree outputs.

Configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

### K-Nearest Neighbors Regressor

A distance-based model that predicts the target from nearby training samples.

Configuration:

```python
KNeighborsRegressor(n_neighbors=5)
```

---

## Evaluation Metrics

The models are evaluated using four regression metrics.

### Mean Absolute Error

```text
MAE = average absolute prediction error
```

Lower values are better.

### Mean Squared Error

```text
MSE = average squared prediction error
```

Larger errors receive a stronger penalty.

### Root Mean Squared Error

```text
RMSE = square root of MSE
```

RMSE is expressed in the same unit as `Credit_Limit`. Lower values are better.

### Coefficient of Determination

```text
R² = proportion of target variance explained by the model
```

Values closer to 1 indicate better explanatory performance.

---

## Results

The following results were recorded in the saved notebook output using the default preprocessing configuration:

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.4336 | 6962.23 | 4949.88 |
| Decision Tree | 0.1516 | 8521.44 | 5188.68 |
| **Random Forest** | **0.5910** | **5916.16** | **3891.83** |
| K-Nearest Neighbors | 0.4028 | 7149.06 | 4748.14 |

### Best Model

Random Forest achieved:

- The highest R² score
- The lowest RMSE
- The lowest MAE

Therefore, Random Forest was selected as the strongest model in the recorded run.

> Results may change when a different missing-value, outlier, or scaling method is selected.

---

## Feature Importance

Random Forest feature importance is used to identify the variables with the greatest influence on credit-limit predictions.

The five most important features in the saved run were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Income_Category` | 0.3374 |
| 2 | `Card_Category` | 0.1831 |
| 3 | `Total_Trans_Amt` | 0.0823 |
| 4 | `Total_Amt_Chng_Q4_Q1` | 0.0569 |
| 5 | `Total_Ct_Chng_Q4_Q1` | 0.0535 |

Income category and card category had the strongest influence on the Random Forest predictions.

Feature importance represents model behavior and should not automatically be interpreted as a causal relationship.

---

## Visualizations

The notebook includes:

- Correlation heatmap
- Top-five Random Forest feature-importance chart
- Actual-versus-predicted scatter plots
- Ideal prediction reference line
- Actual and predicted values for a subset of test samples
- Error-connection lines between actual and predicted values
- Comparative plots for all trained models

These visualizations help reveal prediction quality, large errors, model bias, and important variables.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Main Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## How to Run

1. Place `CreditPrediction.csv` in the project directory.
2. Open the notebook:

```bash
jupyter notebook
```

3. Select the project notebook.
4. Choose the desired preprocessing methods.
5. Run all cells from top to bottom.
6. Review the metrics, plots, and feature importance.

The code assumes that the CSV file is accessible using:

```python
pd.read_csv("CreditPrediction.csv")
```

---

## Configuration

### Missing-Value Method

Select one of the following options:

```python
MISSING_VALUE_METHOD = "Median_Mode"
```

```python
MISSING_VALUE_METHOD = "KNN"
```

```python
MISSING_VALUE_METHOD = "MICE"
```

### Outlier Method

```python
OUTLIER_METHOD = "IQR"
```

```python
OUTLIER_METHOD = "Z-Score"
```

```python
OUTLIER_METHOD = "IsolationForest"
```

### Scaling Method

```python
SCALING_METHOD = "Standard"
```

or:

```python
SCALING_METHOD = "MinMax"
```

To compare preprocessing strategies fairly, change one setting at a time and rerun all cells from the preprocessing stage onward.

---

## Project Structure

```text
credit-limit-prediction/
│
├── CreditPrediction.csv
├── Untitled-1.ipynb
├── README.md
└── requirements.txt        # optional
```

For a cleaner GitHub repository, the notebook can be renamed to:

```text
credit_limit_prediction.ipynb
```

---

## Challenges and Future Improvements

### Current Challenges

- Missing values occur in both numerical and categorical columns.
- Financial features may contain skewed distributions and extreme values.
- KNN-based methods are sensitive to feature scales.
- Some engineered features are strongly correlated with original features.
- Default model parameters may not provide optimal performance.
- Model performance can change across different preprocessing combinations.

### Suggested Improvements

- Compare all missing-value methods in a structured experiment.
- Record model performance for every preprocessing configuration.
- Use cross-validation instead of relying on one train/test split.
- Tune model hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.
- Add regularized models such as Ridge, Lasso, and Elastic Net.
- Test gradient-boosting algorithms such as HistGradientBoosting, XGBoost, LightGBM, or CatBoost.
- Use a Scikit-learn `Pipeline` and `ColumnTransformer` to organize preprocessing.
- Evaluate target transformations if `Credit_Limit` is highly skewed.
- Add residual-distribution and heteroscedasticity analysis.
- Save the final fitted pipeline with `joblib`.
- Add automated tests and a reproducible `requirements.txt`.

---

## Reproducibility

Random states are fixed where supported:

```python
random_state=42
```

This improves reproducibility for:

- Train/test splitting
- Random Forest
- Decision Tree
- Isolation Forest
- MICE-style iterative imputation

Minor differences may still appear across library versions or computing environments.

---

## Conclusion

This project demonstrates an end-to-end regression workflow for predicting customer credit limits.

It compares multiple strategies for:

- Missing-value imputation
- Outlier treatment
- Feature scaling
- Regression modeling

Among the evaluated models in the saved run, **Random Forest** produced the best overall performance. The project also shows that income category, card category, and transaction behavior are among the most influential predictors of customer credit limits.

The modular preprocessing options make the notebook suitable for further experimentation and improvement.

---

## Disclaimer

This project is intended for educational purposes.

Credit-limit decisions in real banking systems require additional validation, fairness testing, explainability, regulatory compliance, security controls, and human oversight.
