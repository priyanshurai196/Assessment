# Colon Cancer Analysis and Classification

This repository provides a complete workflow for analyzing, preprocessing, and building machine learning models to classify colon cancer data. It utilizes Python libraries such as `pandas`, `matplotlib`, `seaborn`, `sklearn`, and `shap` for data manipulation, visualization, model building, and feature interpretation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Data Loading and Exploration](#data-loading-and-exploration)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Train-Test Split](#train-test-split)
7. [Model Training](#model-training)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Model Evaluation](#model-evaluation)
10. [Feature Importance](#feature-importance)
11. [SHAP Analysis](#shap-analysis)

## Project Overview

This project performs an analysis and classification on colon cancer datasets. It includes the following steps:
- Loading and examining two datasets.
- Merging the datasets based on the `Type of Colon Cancer` feature.
- Encoding categorical data and scaling numerical features.
- Engineering new features for assessing risk and severity.
- Training and evaluating Logistic Regression and Random Forest models.
- Interpreting feature importance using SHAP (SHapley Additive exPlanations).

## Requirements

The project requires the following Python libraries, which can be installed using pip:
```bash
pip install pandas matplotlib seaborn scikit-learn shap
Data Loading and Exploration
Load and inspect the datasets using pandas to get an initial understanding of the data structure:

python
Copy code
import pandas as pd

# Load datasets
dataset_1 = pd.read_csv('dataset_1_colon_cancer.csv')
dataset_2 = pd.read_csv('dataset_2_colon_cancer.csv')

# Display initial rows and basic info
print(dataset_1.head())
print(dataset_1.info())
print(dataset_2.head())
print(dataset_2.info())
Data Preprocessing
Merge the two datasets on Type of Colon Cancer, encode categorical variables, and scale numerical ones:

python
Copy code
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Merge datasets
merged_data = pd.merge(dataset_1, dataset_2, on='Type of Colon Cancer', how='inner')

# Encode categorical features
label_encoders = {}
for col in merged_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    merged_data[col] = le.fit_transform(merged_data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns.difference(['Type of Colon Cancer'])
merged_data[numeric_cols] = scaler.fit_transform(merged_data[numeric_cols])
Feature Engineering
Create new features for assessing risk and severity of colon cancer:

python
Copy code
# Risk Indicator
merged_data['Risk Indicator'] = merged_data['Family History'] * merged_data['Smoking History']

# Severity Index
merged_data['Severity Index'] = merged_data['Tumor Grade'] + merged_data['Lymph Node Involvement'] + merged_data['Bowel Obstruction']
Train-Test Split
Split the data into training and test sets:

python
Copy code
from sklearn.model_selection import train_test_split

X = merged_data.drop(columns=['Type of Colon Cancer'])
y = merged_data['Type of Colon Cancer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Model Training
Train Logistic Regression and Random Forest models on the training data:

python
Copy code
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
Hyperparameter Tuning
Use Grid Search to find optimal parameters for both models:

python
Copy code
from sklearn.model_selection import GridSearchCV

# Define parameter grids
lr_param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}

# Grid Search for Logistic Regression
lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid_search.fit(X_train, y_train)

# Grid Search for Random Forest
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Best models
best_lr_model = lr_grid_search.best_estimator_
best_rf_model = rf_grid_search.best_estimator_
Model Evaluation
Evaluate model performance using metrics such as accuracy, precision, recall, and F1 score:

python
Copy code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logistic Regression Evaluation
lr_preds = best_lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_preds)
lr_precision = precision_score(y_test, lr_preds, average='weighted')
lr_recall = recall_score(y_test, lr_preds, average='weighted')
lr_f1 = f1_score(y_test, lr_preds, average='weighted')

# Random Forest Evaluation
rf_preds = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_precision = precision_score(y_test, rf_preds, average='weighted')
rf_recall = recall_score(y_test, rf_preds, average='weighted')
rf_f1 = f1_score(y_test, rf_preds, average='weighted')

# Display results
print(f"Logistic Regression - Accuracy: {lr_accuracy}, Precision: {lr_precision}, Recall: {lr_recall}, F1 Score: {lr_f1}")
print(f"Random Forest - Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}, F1 Score: {rf_f1}")
Feature Importance
Analyze feature importance using the Random Forest model:

python
Copy code
import pandas as pd

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", feature_importances)
SHAP Analysis
Use SHAP (SHapley Additive exPlanations) to interpret the Random Forest model:

python
Copy code
import shap

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)