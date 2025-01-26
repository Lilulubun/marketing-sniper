# Marketing-Sniper
A web-based ad spend allocation tool for optimizing marketing budgets.

## Getting Started
### Backend
1. Navigate to the `backend/` directory.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   python app.py
   ```

### Frontend
1. Navigate to the `frontend/` directory.
2. Start the development server:
   ```bash
   npm start
   ```

## Features
- Input budget and platforms.
- Calculate budget allocation.
- Simple, responsive design with Tailwind CSS.

## Tech Stack
- **Frontend**: React, Tailwind CSS
- **Backend**: Flask

## App Overview
Marketing-Sniper is a web-based ad spend allocation tool designed to optimize marketing budgets. Users can input their total budget, select goals, and target audiences, and the app provides data-driven recommendations for budget distribution across various categories.

## Machine Learning Approach
The Marketing-Sniper app employs a machine learning model trained on historical budget allocation data. The model uses the following input features:
- **Total Budget**: The overall budget available for allocation.
- **Goal**: The marketing objective (e.g., brand awareness, lead generation).
- **Target Audience**: The demographic or segment the marketing efforts are aimed at.

The model processes this input data to predict optimal budget allocations across various categories, such as Ads Budget, Influencer Budget, and Content Budget. By leveraging this data-driven approach, the app provides users with informed recommendations for their marketing spend.

## App Flow
1. Users input their total budget, goal, and target audience through the frontend interface.
2. The frontend sends this data to the backend via a POST request to the `/recommend` endpoint.
3. The backend processes the input, prepares it for the model, and makes predictions using the loaded machine learning model.
4. The predicted budget allocations are returned to the frontend, where they are visualized in a bar chart.

## Running the App Locally
### Backend
1. Navigate to the `backend/` directory.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   python app.py
   ```

### Frontend
1. Navigate to the `frontend/` directory.
2. Start the development server:
   ```bash
   npm start
