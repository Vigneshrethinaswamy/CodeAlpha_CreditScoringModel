# Credit Scoring Model

## Overview

This project predicts an individual's creditworthiness using historical financial data. The objective is to identify whether a customer is likely to default on a loan using machine learning classification techniques.

This project was developed as part of the CodeAlpha Machine Learning Internship.

---

## Dataset

Dataset: Give Me Some Credit Dataset

Features include:

* Revolving Utilization of Unsecured Lines
* Age
* Debt Ratio
* Monthly Income
* Number of Open Credit Lines and Loans
* Number of Times 30-59 Days Past Due
* Number of Times 60-89 Days Past Due
* Number of Times 90 Days Late
* Number of Real Estate Loans
* Number of Dependents

Target Variable:

* SeriousDlqin2yrs

  * 0 = No Default
  * 1 = Default

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Missing Value Imputation
4. Exploratory Data Analysis
5. Train-Test Split
6. Logistic Regression Training
7. Random Forest Training
8. Model Evaluation
9. Best Model Selection
10. Model Saving

---

## Models Used

### Logistic Regression

Used as a baseline classification model.

### Random Forest Classifier

Used to capture complex non-linear relationships and improve prediction performance.

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 77.63%   | 18.17%    | 66.98% | 28.58%   | 80.21%  |
| Random Forest       | 80.58%   | 22.23%    | 76.26% | 34.42%   | 86.78%  |

Best Model: Random Forest

---

## Generated Visualizations

* Class Distribution
* Correlation Heatmap
* Feature Distributions
* Boxplots
* Confusion Matrices
* ROC Curve
* Feature Importance Plot

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

---

## Project Structure

CodeAlpha_CreditScoringModel/

├── data/

├── models/

├── notebooks/

├── plots/

├── src/

│ ├── data_preprocessing.py

│ ├── eda.py

│ ├── train.py

│ └── evaluate.py

├── main.py

├── requirements.txt

└── README.md

---

## Author

Vignesh Rethinaswamy
Machine Learning Intern – CodeAlpha
