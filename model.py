import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset
data = pd.read_csv("scaled_budget_data.csv")

# Preprocess the data
# Extract input features and target variables
X = data[['Total Budget', 'Goal', 'Target Audience']]
y = data[['Ads Budget', 'Influencer Budget', 'Content Budget']]

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(X[['Goal', 'Target Audience']])

# Combine encoded features with numeric ones
X_preprocessed = pd.concat(
    [
        pd.DataFrame(X[['Total Budget']].values, columns=['Total Budget']),
        pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Goal', 'Target Audience']))
    ],
    axis=1
)

# Ensure column names are strings
X_preprocessed.columns = X_preprocessed.columns.astype(str)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# Define a function to predict budget allocation for a new input
def predict_budget(total_budget, goal, target_audience):
    # Prepare the input
    input_data = pd.DataFrame([[total_budget, goal, target_audience]], columns=['Total Budget', 'Goal', 'Target Audience'])
    encoded_input = encoder.transform(input_data[['Goal', 'Target Audience']])
    
    input_preprocessed = pd.concat(
        [
            pd.DataFrame(input_data[['Total Budget']].values, columns=['Total Budget']),
            pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['Goal', 'Target Audience']))
        ],
        axis=1
    )
    
    # Ensure feature names match the model's training data
    input_preprocessed.columns = input_preprocessed.columns.astype(str)
    input_preprocessed = input_preprocessed.reindex(columns=X_preprocessed.columns, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(input_preprocessed)
    return prediction

# Example usage
new_prediction = predict_budget(4000.0, "Increase Traffic", "Public")
print("Predicted budget allocation (Ads, Influencer, Content):", new_prediction)
