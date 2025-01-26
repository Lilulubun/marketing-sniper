import pandas as pd

# Load CSV
file_path = "cleaned_data.csv"  # Update with your file path
output_path = "scaled_budget_data.csv"
data = pd.read_csv(file_path)

# Scale budgets
for index, row in data.iterrows():
    ads_budget = row['Ads Budget']
    influencer_budget = row['Influencer Budget']
    content_budget = row['Content Budget']
    total_budget = row['Total Budget']

    # Calculate the sum of the three fields
    total_allocation = ads_budget + influencer_budget + content_budget

    # Scale each field proportionally if total allocation is not zero
    if total_allocation != 0:
        scale_factor = total_budget / total_allocation
        data.loc[index, 'Ads Budget'] = ads_budget * scale_factor
        data.loc[index, 'Influencer Budget'] = influencer_budget * scale_factor
        data.loc[index, 'Content Budget'] = content_budget * scale_factor

# Save the updated CSV
data.to_csv(output_path, index=False)
print(f"Scaled budgets have been saved to {output_path}")
