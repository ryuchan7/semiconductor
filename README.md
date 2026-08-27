# Semiconductor Process Intelligence

An AI-based semiconductor manufacturing analytics system for yield failure prediction, process anomaly detection, and explainable AI.

## Features

- Semiconductor yield failure prediction using Random Forest
- Process anomaly detection using Isolation Forest
- SHAP explainability
- CSV batch analysis
- Production batch monitoring
- Priority review queue
- Detailed sample investigation
- Analysis result export

## Dataset

UCI SECOM Semiconductor Manufacturing Dataset.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib

## Current Model

Random Forest is used as the primary yield-risk prediction model.

Decision threshold:

0.10

Test results:

- Accuracy: 82.5%
- FAIL Precision: 19.6%
- FAIL Recall: 52.4%
- FAIL F1-score: 28.6%
- ROC-AUC: 76.9%

## Anomaly Detection

Isolation Forest is used as a secondary process anomaly detector.

Anomaly threshold:

0.408069

## Disclaimer

This project is an educational and research prototype based on anonymized semiconductor manufacturing data.