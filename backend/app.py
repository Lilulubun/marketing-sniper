import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np  # Ensure numpy is imported

# Set up logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Load the pre-trained model and encoder
def load_model():
    try:
        model = joblib.load('budget_allocation_model.pkl')
        encoder = joblib.load('encoder.pkl')
        app.logger.debug('Model and encoder loaded successfully')
        return model, encoder
    except Exception as e:
        app.logger.error(f"Error loading model or encoder: {e}")
        return None, None

model, encoder = load_model()

def predict_budget_allocation(total_budget, goal, target_audience):
    if model is None or encoder is None:
        app.logger.error("Model or encoder not loaded properly, cannot make predictions")
        return None

    # Prepare the input data
    input_data = pd.DataFrame([[total_budget, goal, target_audience]], columns=['Total Budget', 'Goal', 'Target Audience'])
    
    try:
        # Apply one-hot encoding to categorical features
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
        input_preprocessed = input_preprocessed.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Make a prediction
        allocation = model.predict(input_preprocessed)
        app.logger.debug(f"Predicted allocation: {allocation}")

        # Flatten the allocation if it's a 2D array
        if isinstance(allocation, np.ndarray) and allocation.ndim > 1:
            allocation = allocation.flatten().tolist()
        elif isinstance(allocation, (list, np.ndarray)):
            allocation = allocation.tolist()

        return allocation

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return None

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    if not data:
        app.logger.error("No data received in request")
        return jsonify({"error": "Invalid input data"}), 400

    try:
        total_budget = float(data['total_budget'])
        goal = data['goal']
        target_audience = data['target_audience']

    except KeyError as e:
        app.logger.error(f"Missing parameter in request: {e}")
        return jsonify({"error": f"Missing parameter: {e}"}), 400
    except ValueError as e:
        app.logger.error(f"Invalid data type: {e}")
        return jsonify({"error": "Invalid data type"}), 400

    # Make a prediction
    recommended_allocation = predict_budget_allocation(total_budget, goal, target_audience)

    if recommended_allocation is None:
        return jsonify({"error": "Failed to make prediction. Ensure the goal and target audience match the training data."}), 500

    # Map allocation amounts to budget names
    budget_names = ["Ads Budget", "Influencer Budget", "Content Budget"]
    allocation_with_names = [
        {"name": name, "value": value}
        for name, value in zip(budget_names, recommended_allocation)
    ]

    return jsonify({'recommended_allocation': allocation_with_names})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)