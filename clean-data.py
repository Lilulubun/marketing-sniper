import pandas as pd
import re

# Load the CSV file
input_file = "output.csv"
output_file = "cleaned_data.csv"

# Read the CSV into a DataFrame
data = pd.read_csv(input_file)

# Define columns that should only contain numbers
numeric_columns = ["Total Budget", "Ads Budget", "Influencer Budget", "Content Budget"]

# Function to clean numeric fields
def clean_numeric(value):
    if pd.notnull(value):
        # Remove non-numeric characters except periods
        return re.sub(r"[^0-9.]", "", str(value))
    return value

# Apply cleaning to numeric columns
for column in numeric_columns:
    data[column] = data[column].apply(clean_numeric)

# Save the cleaned DataFrame to a new CSV file
data.to_csv(output_file, index=False)

print(f"Cleaned data has been saved to {output_file}")
