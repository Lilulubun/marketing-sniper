import pandas as pd

# Load the CSV file
df = pd.read_csv('processed_marketing_data.csv')

# Define the categorization mapping
def categorize_audience(audience):
    if "Gen Z" in audience or "18-24" in audience or "18-25" in audience or "18-30" in audience:
        return "Gen Z"
    elif "Millennial" in audience or "25-45" in audience or "22-35" in audience or "25-35" in audience or "25-40" in audience:
        return "Millennial"
    elif "Business owners" in audience or "IT professionals" in audience or "IT Decision-Makers" in audience:
        return "Business Owner"
    else:
        return "Boomer"  # Fallback for any other group

# Apply the categorization to the 'Target Audience' column
df['Audience Category'] = df['Target Audience'].apply(categorize_audience)

# Drop the original 'Target Audience' column
df = df.drop(columns=['Target Audience'])


# Save the updated DataFrame to a new CSV file (optional)
df.to_csv('categorized_marketing_data.csv', index=False)

# Display the updated DataFrame
print(df[['Target Audience', 'Audience Category']].head())