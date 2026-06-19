# PaySim Fraud Detection

This project focuses on fraud detection using the PaySim synthetic financial transaction dataset.

The main goal of this project was to compare multiple machine learning models on a highly imbalanced fraud detection problem and understand how threshold tuning affects false positives, false negatives, precision, recall, and F1-score.

## Project Overview

Fraud detection is a binary classification problem:

* Positive class: Fraud transaction
* Negative class: Normal transaction

Because the dataset is highly imbalanced, accuracy alone is not a useful metric. Instead, this project focuses on metrics that are more meaningful for fraud detection:

* Precision
* Recall
* F1-score
* False Positives
* False Negatives
* Threshold tuning

## Dataset

This project uses the PaySim synthetic financial transaction dataset.

The original dataset file and the prepared parquet file are not included in this repository because of file size.

The local data files are expected to be stored in the `data/` folder:

```text
data/
├── paysim.parquet
└── original_dataset.xlsx
```

## Project Structure

```text
paysim-fraud-detection/
│
├── data/
├── outputs/
│   ├── logistic_regression/
│   ├── decision_tree/
│   └── random_forest/
│
├── prepare_data.py
├── fraud_eda.py
├── logistic_regression_model.py
├── decision_tree_model.py
├── random_forest_model.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Workflow

The project follows this workflow:

1. Prepare the dataset
2. Perform exploratory data analysis
3. Train a Logistic Regression baseline model
4. Train Decision Tree models
5. Train Random Forest models
6. Tune classification thresholds
7. Compare models using fraud-class metrics
8. Analyze false positives and false negatives
9. Select final model options based on business priorities

## Models Tested

The following models were tested:

* Logistic Regression
* Controlled Logistic Regression
* Raw Decision Tree
* Controlled Decision Tree
* Controlled Random Forest
* Stronger Random Forest

## Key Evaluation Idea

In fraud detection, different types of errors have different business costs.

A false negative means a fraudulent transaction was missed.

A false positive means a normal transaction was incorrectly flagged as fraud.

Because of this, the final model should not be selected only by F1-score. The decision also depends on the business priority:

* If missing fraud is very expensive, recall and false negatives are more important.
* If manual review workload must be minimized, false positives are more important.
* If a balanced model is needed, precision, recall, F1-score, FP, and FN should be considered together.

## Model Results Summary

Logistic Regression performed as a useful baseline, but it missed too many fraudulent transactions.

The raw Decision Tree achieved strong results but showed signs of overfitting.

The controlled Decision Tree was more stable and more reliable than the raw tree.

Random Forest provided the best overall model family in this project.

## Final Model Interpretation

The stronger Random Forest model was selected as the strongest overall model family.

Different thresholds can be used depending on the business goal.

### Fraud-focused option

```text
Model: stronger_forest_model
Threshold: 0.3
False Positives: 76
False Negatives: 390
Precision: 0.943
Recall: 0.763
F1-score: 0.843
```

This option catches more fraudulent transactions, but creates more false positives.

### Balanced option

```text
Model: stronger_forest_model
Threshold: 0.5
False Positives: 33
False Negatives: 407
Precision: 0.974
Recall: 0.752
F1-score: 0.849
```

This option provides a balanced trade-off between fraud detection and false positive control.

### Workload-friendly option

```text
Model: stronger_forest_model
Threshold: 0.6
False Positives: 10
False Negatives: 417
Precision: 0.992
Recall: 0.746
F1-score: 0.852
```

This option minimizes false positives and reduces manual review workload.

## Outputs

The project generates CSV files and plots for each model family.

```text
outputs/
├── logistic_regression/
├── decision_tree/
└── random_forest/
```

The outputs include:

* Model performance CSV files
* Threshold comparison CSV files
* Candidate model CSV files
* Precision / Recall / F1 plots
* False Positive / False Negative plots

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the scripts in this order:

```bash
python prepare_data.py
python fraud_eda.py
python logistic_regression_model.py
python decision_tree_model.py
python random_forest_model.py
```

## Requirements

Main libraries used:

* pandas
* scikit-learn
* matplotlib
* pyarrow
* openpyxl

## What I Learned

In this project, I learned:

* How to work with a large financial transaction dataset
* Why accuracy is not enough for imbalanced classification
* How to evaluate fraud detection models using precision, recall, F1-score, FP, and FN
* How threshold tuning changes model behavior
* How to compare different machine learning models
* How overfitting appears in decision trees
* How to organize model outputs for a portfolio project
* How model selection depends on business priorities, not only metric scores

## Project Status

This project is currently completed as a model comparison and analysis project.

A possible next step is to build a simple Streamlit app where a user can upload transaction data and receive fraud risk predictions.
