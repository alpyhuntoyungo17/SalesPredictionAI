import joblib
import pandas as pd
from flask import Blueprint, request, jsonify
import os

# Load the trained model
model_path = os.path.join('models', 'sales_prediction_model.joblib')
model = joblib.load(model_path)

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json

        # Convert data to DataFrame
        df = pd.DataFrame(data, index=[0])

        # Ensure all required features are present
        required_features = ['day_of_week', 'month', 'category', 'store_id', 'weather_condition',
                             'avg_price', 'is_weekend', 'is_holiday', 'promotion_active', 'temperature']
        
        for feature in required_features:
            if feature not in df.columns:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400

        # Make prediction
        prediction = model.predict(df)

        # Return the prediction
        return jsonify({'prediction': prediction[0]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@predict_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Get data from request
        data = request.json

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Ensure all required features are present
        required_features = ['day_of_week', 'month', 'category', 'store_id', 'weather_condition',
                             'avg_price', 'is_weekend', 'is_holiday', 'promotion_active', 'temperature']
        
        for feature in required_features:
            if feature not in df.columns:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400

        # Make predictions
        predictions = model.predict(df)

        # Return the predictions
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
