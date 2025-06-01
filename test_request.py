import requests
from bs4 import BeautifulSoup
import re

# Data for a severe hypothyroid case with values in the training data range
data = {
    'age': '42',
    'sex': '0',  # Female
    'on_thyroxine': '1',  # Currently on thyroxine
    'query_on_thyroxine': '1',
    'on_antithyroid_medication': '0',
    'sick': '1',
    'pregnant': '0',
    'thyroid_surgery': '0',
    'I131_treatment': '0',
    'query_hypothyroid': '1',
    'query_hyperthyroid': '0',
    'lithium': '0',
    'goitre': '1',
    'tumor': '0',
    'hypopituitary': '0',
    'psych': '0',
    'T3': '0.5',     # Very low T3
    'TT4': '15.0',   # Very low TT4
    'T4U': '0.6',    # Low T4U
    'FTI': '15.0',   # Very low FTI
    'referral_source': 'SVI'
}

# Send POST request to the endpoint
response = requests.post('http://localhost:5000/predict', data=data)

# Parse the HTML response
soup = BeautifulSoup(response.text, 'html.parser')

# Find any text containing "Positive" or "Negative"
prediction_pattern = re.compile(r'(Positive|Negative).*confidence', re.IGNORECASE)
for element in soup.find_all(text=prediction_pattern):
    print("Prediction:", element.strip())
    break
else:
    print("Raw response:", response.text) 