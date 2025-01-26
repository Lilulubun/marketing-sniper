import json
import csv
import random

# Load the JSON data
with open('marketing_social_media_dataset_v2.json', 'r') as file:
    data = json.load(file)

# Define categories for target audience and goal
target_audience_categories = ['Gen Z', 'Millennial', 'Business Company', 'Public']
goal_categories = ['Brand Awareness', 'Sales', 'Increase Traffic']

# Function to categorize target audience
def categorize_audience(audience):
    audience = audience.lower()
    if 'gen z' in audience or '18-24' in audience or '18-25' in audience:
        return 'Gen Z'
    elif 'millennial' in audience or '25-35' in audience or '25-45' in audience:
        return 'Millennial'
    elif 'business' in audience or 'b2b' in audience or 'it professionals' in audience or 'business owners' in audience:
        return 'Business Company'
    elif 'public' in audience or 'general public' in audience or 'all ages' in audience:
        return 'Public'
    else:
        # Inject a random value if no match is found
        return random.choice(target_audience_categories)

# Function to categorize goal
def categorize_goal(goal):
    goal = goal.lower()
    if 'brand awareness' in goal or 'increase followers' in goal or 'increase recognition' in goal:
        return 'Brand Awareness'
    elif 'sales' in goal or 'increase sales' in goal or 'drive sales' in goal or 'boost revenue' in goal:
        return 'Sales'
    elif 'traffic' in goal or 'increase website traffic' in goal or 'drive traffic' in goal or 'boost visits' in goal:
        return 'Increase Traffic'
    else:
        # Inject a random value if no match is found
        return random.choice(goal_categories)

# Prepare CSV file
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Total Budget', 'Target Audience', 'Goal', 'Ads Budget', 'Influencer Budget', 'Content Budget', 'Response Text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for entry in data:
        # Extract total budget from constraints
        constraints = entry['input']
        total_budget = None
        if 'budget' in constraints.lower() and '$' in constraints:
            total_budget = constraints.split('$')[-1].split()[0]
            if not total_budget:
                total_budget = random.choice([1000, 1500, 2000, 3000, 5000])
        
        # Extract target audience and categorize
        target_audience = entry['input'].split('Target Audience: ')[-1].split('\n')[0]
        target_audience_category = categorize_audience(target_audience)
        
        # Extract goal and categorize
        goal = entry['input'].split('Goals: ')[-1].split('\n')[0]
        goal_category = categorize_goal(goal)
        
        # Extract budgets from response
        response = entry['response']
        ads_budget = None
        influencer_budget = None
        content_budget = None
        
        if '$' in response:
            parts = response.split('$')
            for part in parts[1:]:
                amount = part.split()[0]
                if not amount:
                    amount = random.choice([1000, 1500, 2000, 3000, 5000])
                if 'ads' in part.lower():
                    ads_budget = amount
                elif 'influencer' in part.lower():
                    influencer_budget = amount
                elif 'content' in part.lower():
                    content_budget = amount
        
        # Write to CSV
        writer.writerow({
            'Total Budget': total_budget if total_budget else random.choice([1000, 1500, 2000, 3000, 5000]),
            'Target Audience': target_audience_category,
            'Goal': goal_category,
            'Ads Budget': ads_budget if ads_budget else random.choice([1000, 1500, 2000, 3000, 5000]),
            'Influencer Budget': influencer_budget if influencer_budget else random.choice([1000, 1500, 2000, 3000, 5000]),
            'Content Budget': content_budget if content_budget else random.choice([1000, 1500, 2000, 3000, 5000]),
            'Response Text': response
        })
