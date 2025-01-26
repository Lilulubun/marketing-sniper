import pandas as pd
import re
import random

# Load the CSV file
file_path = './extracted_marketing_data.csv'  # Replace with your file path
output_file_path = './processed_marketing_data.csv'  # Output file path

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Function to extract numeric value from Budget Constraints
def extract_number(value):
    if pd.isna(value):
        return None
    numbers = re.findall(r'\d+', str(value))
    return int(numbers[0]) if numbers else None

# Apply the function to the Budget Constraints column
df['Budget Constraints'] = df['Budget Constraints'].apply(extract_number)

# Remove rows where Budget Constraints is None
df = df.dropna(subset=['Budget Constraints'])

# List of possible goals to randomize
possible_goals = ["Sales", "Brand Awareness", "Website Traffic", "Lead Generation", "Engagement"]

# Function to categorize the Goals field
def categorize_goals(value):
    if pd.isna(value) or value == '':
        # Randomly select a goal if the field is empty
        return random.choice(possible_goals)
    value = value.lower()
    if 'increase sales' in value or 'boost sales' in value:
        return 'Sales'
    elif 'brand awareness' in value:
        return 'Brand Awareness'
    elif 'traffic' in value:
        return 'Website Traffic'
    elif 'lead' in value:
        return 'Lead Generation'
    elif 'engagement' in value:
        return 'Engagement'
    else:
        return value  # Return the original value if it doesn't match any category

# Function to check and move goals from Target Audience to Goals
def move_goals_from_audience(row):
    audience = str(row['Target Audience']).lower()
    if 'increase sales' in audience or 'boost sales' in audience:
        return 'Sales'
    elif 'brand awareness' in audience:
        return 'Brand Awareness'
    elif 'traffic' in audience:
        return 'Website Traffic'
    elif 'lead' in audience:
        return 'Lead Generation'
    elif 'engagement' in audience:
        return 'Engagement'
    return row['Goals']

# Apply the function to move goals from Target Audience to Goals
df['Goals'] = df.apply(move_goals_from_audience, axis=1)

# Apply the categorization function to the Goals column
df['Goals'] = df['Goals'].apply(categorize_goals)

df = df.dropna()

# Save the updated DataFrame back to a CSV file
df.to_csv(output_file_path, index=False)

print(f"Updated file saved to {output_file_path}")