# End-to-End Credit Card Fraud Detection Using Machine Learning  
### Model Comparison, Threshold Optimization, and Business Impact Analysis

## Project Overview

Credit card fraud is one of the most serious challenges in the financial industry. Fraud detection is difficult because fraudulent transactions represent only a very small portion of the total transactions, resulting in a highly **imbalanced dataset**.

This project aims to build an **end-to-end machine learning pipeline** to detect fraudulent credit card transactions. Multiple machine learning algorithms are compared to identify the best-performing model.

In addition to model comparison, this project also performs **threshold optimization** and **business impact analysis** to ensure the selected model not only performs well technically but also provides meaningful business value in reducing potential financial losses caused by fraud.

The result of this project is expected to help financial institutions detect fraudulent transactions earlier and minimize financial losses.

---

# Main Results

The best performing model in this project is **XGBoost Tuned**, with the following performance :

- **PR-AUC : 0.874**
- **Fraud Detection Rate : 78.6%**
- **False Positive Rate : Very Low**

These results indicate that the model is capable of detecting the majority of fraudulent transactions in a highly imbalanced dataset while maintaining a very low number of false alarms.

---

# Project Information

**Author** : Neo Saffana Farhalik  
**Role** : Data Analyst  
**Date** : February 2026  
**Environment** : Python, Google Colab  
**GitHub** : https://github.com/neosaffana

---

# Project Objectives

The main objectives of this project are :

- Build a machine learning model to detect potentially fraudulent credit card transactions
- Compare the performance of several machine learning algorithms
- Evaluate models using metrics suitable for **imbalanced datasets**
- Optimize the **classification threshold** to improve fraud detection performance
- Analyze the **business impact** of the resulting model

---

# Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** published by the Machine Learning Group ULB and available on Kaggle.

The dataset contains credit card transactions made by European cardholders in **September 2013**.

Dataset Source :  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# Dataset Characteristics

- **Total transactions** : 284,807  
- **Fraud transactions** : 492  
- **Fraud proportion** : 0.172%  
- Dataset is **highly imbalanced**

This imbalance makes fraud detection particularly challenging because the model must identify a very small number of fraud cases among a large number of normal transactions.

---

# Dataset Features

The dataset contains **31 features** :

- **Time** → time elapsed since the first transaction
- **Amount** → transaction amount
- **V1 – V28** → PCA-transformed features to protect sensitive information
- **Class** → target variable  
  - 0 = Normal transaction  
  - 1 = Fraudulent transaction

Because the dataset is highly imbalanced, model evaluation focuses on metrics such as **Precision, Recall, F1 Score, ROC-AUC, and PR-AUC**.

---

# Machine Learning Models Used

Several machine learning algorithms were implemented and compared in this project :

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Each model was trained using the training dataset and evaluated on the testing dataset to assess its performance in detecting fraudulent transactions.

---

# Evaluation Metrics

Since the dataset is highly imbalanced, **accuracy alone is not a reliable metric**.

The following evaluation metrics were used :

### Precision
Measures how many predicted fraud transactions are actually fraud.

### Recall
Measures how many fraud transactions are successfully detected by the model.

### F1 Score
The harmonic mean of precision and recall.

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
Measures the model's ability to distinguish between fraud and non-fraud transactions.

### PR-AUC (Precision-Recall Area Under Curve)
A crucial metric for **imbalanced datasets** because it focuses on the model’s performance in detecting fraud cases.

---

# Project Pipeline

The workflow of this project includes the following stages :

1. Import Library  
2. Load Dataset  
3. Data Understanding  
4. Exploratory Data Analysis (EDA)  
5. Data Preprocessing  
6. Model Training  
7. Model Comparison  
8. Threshold Optimization  
9. Business Impact Analysis  
10. Final Model and Business Recommendation

---

# Final Model and Business Recommendation

Based on the experimental results and model evaluation, several machine learning algorithms were compared to detect fraudulent credit card transactions, including :

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The evaluation results show that **XGBoost Tuned** provides the best performance with :

- **ROC-AUC : 0.978**
- **PR-AUC : 0.874**

Because the dataset is highly imbalanced, **PR-AUC is considered the most relevant metric** for evaluating fraud detection performance.

---

## Threshold Optimization

Threshold optimization was performed to determine the most optimal threshold value that balances precision and recall.

The optimal threshold selected is:

**Threshold = 0.7**

With this threshold, the model produces :

- **Fraud Detected** : 77 transactions
- **Missed Fraud** : 21 transactions
- **False Alarm** : 3 transactions

The model is able to detect approximately **78.6% of fraudulent transactions** while maintaining a very low number of false alarms.

---

# Business Impact

The results show that the model has strong potential to support banking and fintech systems in detecting fraudulent transactions more effectively.

By detecting fraud earlier and minimizing false alarms, financial institutions can significantly reduce potential financial losses while maintaining a better customer experience.

---

## How to Run the Project

1. Clone this repository

git clone https://github.com/neosaffana/credit-card-fraud-detection.git

2. Download the dataset

Download the dataset from Kaggle https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

3. Place the dataset file

Place the file `creditcard.csv` in the project folder.

4. Install required libraries

pip install -r requirements.txt

5. Open the notebook

credit_card_fraud_detection.ipynb

6. Run all cells to reproduce the analysis.

---

## Project Structure

credit-card-fraud-detection  
│  
├── credit_card_fraud_detection.ipynb  
├── README.md  
├── requirements.txt  
└── .gitignore  

---

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
LightGBM  
Matplotlib  
Seaborn  
Google Colab  

---

Note: The dataset is not included in this repository due to file size limitations.
Please download the dataset from Kaggle and place the file `creditcard.csv`
in the project directory before running the notebook.
