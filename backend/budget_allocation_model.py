import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import re

# Load dataset
file_path = "dataset.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Preprocess data
label_encoders = {}

# Encode categorical features
for column in ['Goals', 'Audience Category', 'Budget Constraints']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Function to extract the dollar amount from a string (for 'Allocated Budget Distribution')
def extract_amount(text):
    match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    if match:
        return float(match.group(1).replace(',', ''))  # Remove commas and convert to float
    return 0.0  # Default value if no match is found

# Apply the function to the target column
target_column = 'Allocated Budget Distribution'
data[target_column] = data[target_column].apply(extract_amount)

# Feature columns and target column
feature_columns = ['Budget Constraints', 'Goals', 'Audience Category']
X = data[feature_columns]
y = data[target_column]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump({'model': model, 'encoders': label_encoders}, file)

print("Model and encoders saved to model.pkl")