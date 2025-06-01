from flask import Flask, request, render_template
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define feature names to match the model's expectations
FEATURE_NAMES = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'T3', 'TT4', 'T4U', 'FTI',
    'referral_source_STMW', 'referral_source_SVHC', 'referral_source_SVHD', 'referral_source_SVI',
    'referral_source_other'
]

# Load the trained model and scaler
try:
    model = pickle.load(open('model/xg_Model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from the form
            features = []
            
            # Basic Information
            features.append(float(request.form['age']))
            features.append(1 if request.form['sex'] == '1' else 0)  # Convert sex to binary
            
            # Medical History
            features.append(1 if request.form['on_thyroxine'] == '1' else 0)
            features.append(1 if request.form['query_on_thyroxine'] == '1' else 0)
            features.append(1 if request.form['on_antithyroid_medication'] == '1' else 0)
            features.append(1 if request.form['sick'] == '1' else 0)
            features.append(1 if request.form['pregnant'] == '1' else 0)
            features.append(1 if request.form['thyroid_surgery'] == '1' else 0)
            features.append(1 if request.form['I131_treatment'] == '1' else 0)
            features.append(1 if request.form['query_hypothyroid'] == '1' else 0)
            features.append(1 if request.form['query_hyperthyroid'] == '1' else 0)
            features.append(1 if request.form['lithium'] == '1' else 0)
            features.append(1 if request.form['goitre'] == '1' else 0)
            features.append(1 if request.form['tumor'] == '1' else 0)
            features.append(1 if request.form['hypopituitary'] == '1' else 0)
            features.append(1 if request.form['psych'] == '1' else 0)
            
            # Test Results
            features.append(float(request.form['T3']))
            features.append(float(request.form['TT4']))
            features.append(float(request.form['T4U']))
            features.append(float(request.form['FTI']))
            
            # Referral Source (One-hot encoded)
            referral_source = request.form['referral_source']
            features.append(1 if referral_source == 'STMW' else 0)
            features.append(1 if referral_source == 'SVHC' else 0)
            features.append(1 if referral_source == 'SVHD' else 0)
            features.append(1 if referral_source == 'SVI' else 0)
            features.append(1 if referral_source == 'other' else 0)
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Create a DataFrame with feature names
            features_df = pd.DataFrame(features_array, columns=FEATURE_NAMES)
            
            # Print input features for debugging
            print("Input features:", features)
            print("Feature length:", len(features))
            print("Feature names:", FEATURE_NAMES)
            
            # Scale the features
            features_scaled = scaler.transform(features_df)
            
            # Make prediction
            probabilities = model.predict_proba(features_scaled)
            
            # Get the probability of positive class (thyroid disease)
            positive_prob = probabilities[0][1]  # Assuming binary classification with positive class at index 1
            probability_percent = round(positive_prob * 100, 2)
            
            # Determine the prediction text
            if positive_prob >= 0.5:
                prediction_text = f'Positive (Thyroid Disease) with {probability_percent}% confidence'
            else:
                prediction_text = f'Negative (No Thyroid Disease) with {(100 - probability_percent)}% confidence'
            
            return render_template('index.html', prediction_text=prediction_text)
            
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error making prediction: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
