import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load and preprocess data
data = pd.read_csv('Dataset/hypothyroid (1).csv')

# Replace missing values
data.replace('?', np.nan, inplace=True)
data['sex'] = data['sex'].map({'F': 0, 'M': 1})

# Define feature names
FEATURE_NAMES = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'T3', 'TT4', 'T4U', 'FTI',
    'referral_source_STMW', 'referral_source_SVHC', 'referral_source_SVHD', 'referral_source_SVI',
    'referral_source_other'
]

# Convert binary columns to 0/1
binary_columns = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych'
]

for col in binary_columns:
    data[col] = data[col].map({'t': 1, 'f': 0})

# Handle measured columns
measured_columns = ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']
for col in measured_columns:
    data[col.replace('_measured', '')] = data[col.replace('_measured', '')].astype(float)

# One-hot encode referral source
referral_dummies = pd.get_dummies(data['referral_source'], prefix='referral_source')
data = pd.concat([data.drop('referral_source', axis=1), referral_dummies], axis=1)

# Drop measured columns and TBG
data = data.drop(measured_columns + ['TBG'], axis=1)

# Convert age to float
data['age'] = data['age'].astype(float)

# Prepare features and target
X = data[FEATURE_NAMES]
y = data['Class'].map({'negative': 0, 'compensated_hypothyroid': 1, 'primary_hypothyroid': 1})

# Drop rows with missing values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train XGBoost model
model = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic'
)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open('model/xg_Model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

print("Model and scaler saved successfully!")

# Test prediction
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Print feature importance
importance = model.feature_importances_
for name, score in sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score}") 