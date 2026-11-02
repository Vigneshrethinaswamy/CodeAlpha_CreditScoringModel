# Credit Scoring Model

## Overview

This project predicts an individual's creditworthiness using historical financial data. The objective is to identify whether a customer is likely to default on a loan using machine learning classification techniques.


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

1. Load and clean the Give Me Some Credit data
2. Separate features and target
3. Stratified 80/20 train/test split
4. Fit median imputation values on the training split only; apply them to train and test
5. Run exploratory data analysis
6. Train Logistic Regression and Random Forest pipelines
7. Evaluate with threshold-sensitive and ranking metrics
8. Select and save the model

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

## Recorded results

The table below reflects the metrics recorded by the earlier project run. The preprocessing code has since been corrected so missing-value medians are fitted on the training split only. These metrics have **not yet been regenerated after that correction**, so treat them as historical results rather than the verified output of the current code.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 77.63%   | 18.17%    | 66.98% | 28.58%   | 80.21%  |
| Random Forest       | 80.58%   | 22.23%    | 76.26% | 34.42%   | 86.78%  |

Best Model in the recorded run: Random Forest

**Important interpretation:** The target is imbalanced. The Random Forest's 80.58% accuracy coexists with only 22.23% precision and 34.42% F1 for the positive/default class (as recorded in the report). Accuracy alone is therefore not a sufficient measure of default detection. Review the confusion matrix, precision-recall trade-off and decision threshold before using predictions.

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

Credit-Risk-Prediction/

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

## Limitations

- This is an educational classification project, not a credit underwriting system.
- Dataset provenance and redistribution terms should be checked before republishing the raw CSV.
- The saved historical metrics should be rerun against the current leakage-reduced preprocessing pipeline before being presented as current performance.

## Author

Vignesh R
