import pandas as pd
import re

# Load your CSV
data = pd.read_csv('dataset.csv')

# Step 1: Extract numerical values from the 'Allocated Budget Distribution'
def extract_budget(budget_str):
    # Use regex to find all numeric values in the text
    budget_values = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', budget_str)
    # Convert the values to integers
    return [int(value.replace(',', '')) for value in budget_values]

# Apply the function to the 'Allocated Budget Distribution' column
data[['Influencer Partnerships Budget', 'Targeted Ads Budget', 'Content Creation Budget']] = pd.DataFrame(data['Allocated Budget Distribution'].apply(extract_budget).to_list(), index=data.index)

# Step 2: One-Hot Encoding for 'Goals' and 'Audience Category'
data_encoded = pd.get_dummies(data, columns=['Goals', 'Audience Category'])

# Step 3: Scale numerical columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_encoded[['Budget Constraints', 'Influencer Partnerships Budget', 'Targeted Ads Budget', 'Content Creation Budget']] = scaler.fit_transform(
    data_encoded[['Budget Constraints', 'Influencer Partnerships Budget', 'Targeted Ads Budget', 'Content Creation Budget']]
)

# Step 4: Save the cleaned data to a new CSV
data_encoded.to_csv('encoded_data.csv', index=False)
