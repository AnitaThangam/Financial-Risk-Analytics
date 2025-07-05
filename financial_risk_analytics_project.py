# Financial Risk Analytics - Loan Default Prediction Project
# Complete End-to-End Project for Resume/Portfolio

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv('loan_data.csv')
df.head()

# -------------------------------
# Step 2: Data Cleaning & Preparation
# -------------------------------
print(df.isnull().sum())  # Check for missing values
df.fillna(method='ffill', inplace=True)
df['Credit_History'] = df['Credit_History'].astype('category')

df.describe()

# -------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -------------------------------
sns.countplot(data=df, x='Loan_Status')
plt.title('Loan Approval vs Default Status')
plt.show()

sns.boxplot(data=df, x='Loan_Status', y='ApplicantIncome')
plt.title('Income Distribution by Loan Status')
plt.show()

sns.boxplot(data=df, x='Loan_Status', y='LoanAmount')
plt.title('Loan Amount Distribution by Loan Status')
plt.show()

# -------------------------------
# Step 4: Predictive Modeling
# -------------------------------
features = df[['ApplicantIncome', 'LoanAmount', 'Credit_History']]
target = df['Loan_Status'].map({'Y': 0, 'N': 1})  # 1 = Default

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# Step 5: Model Evaluation
# -------------------------------
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# -------------------------------
# Step 6: Feature Importance
# -------------------------------
importances = model.feature_importances_
feature_names = features.columns

sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance for Loan Default Prediction')
plt.show()

# -------------------------------
# Step 7: Export Clean Data for Power BI
# -------------------------------
df['Loan_Status_Numeric'] = target
df.to_csv('clean_loan_data_for_powerbi.csv', index=False)

print("Data prepared for Power BI. Use 'clean_loan_data_for_powerbi.csv' for dashboard creation.")
