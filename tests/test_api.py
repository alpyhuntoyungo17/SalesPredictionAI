import unittest
import json
from flask import Flask
from backend.server import app
from backend.routes.predict import predict_bp

class TestPredictionAPI(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.register_blueprint(predict_bp)
        self.client = self.app.test_client()

    def test_predict_endpoint(self):
        # Test data
        test_data = {
            "day_of_week": "Monday",
            "month": 1,
            "category": "Electronics",
            "store_id": "S001",
            "weather_condition": "Sunny",
            "avg_price": 999.99,
            "is_weekend": 0,
            "is_holiday": 0,
            "promotion_active": 1,
            "temperature": 25.5
        }

        # Send POST request to /predict endpoint
        response = self.client.post('/predict', 
                                    data=json.dumps(test_data),
                                    content_type='application/json')

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains a prediction
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIsInstance(data['prediction'], (int, float))

    def test_predict_missing_feature(self):
        # Test data with missing feature
        test_data = {
            "day_of_week": "Monday",
            "month": 1,
            "category": "Electronics",
            "store_id": "S001",
            "weather_condition": "Sunny",
            "avg_price": 999.99,
            "is_weekend": 0,
            "is_holiday": 0,
            "promotion_active": 1
            # Missing 'temperature'
        }

        # Send POST request to /predict endpoint
        response = self.client.post('/predict', 
                                    data=json.dumps(test_data),
                                    content_type='application/json')

        # Check if the response indicates an error
        self.assertEqual(response.status_code, 400)

        # Check if the response contains an error message
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Missing required feature', data['error'])

    def test_batch_predict_endpoint(self):
        # Test data for batch prediction
        test_data = [
            {
                "day_of_week": "Monday",
                "month": 1,
                "category": "Electronics",
                "store_id": "S001",
                "weather_condition": "Sunny",
                "avg_price": 999.99,
                "is_weekend": 0,
                "is_holiday": 0,
                "promotion_active": 1,
                "temperature": 25.5
            },
            {
                "day_of_week": "Tuesday",
                "month": 2,
                "category": "Clothing",
                "store_id": "S002",
                "weather_condition": "Rainy",
                "avg_price": 59.99,
                "is_weekend": 0,
                "is_holiday": 0,
                "promotion_active": 0,
                "temperature": 18.0
            }
        ]

        # Send POST request to /batch_predict endpoint
        response = self.client.post('/batch_predict', 
                                    data=json.dumps(test_data),
                                    content_type='application/json')

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains predictions
        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)
        self.assertEqual(len(data['predictions']), 2)

if __name__ == '__main__':
    unittest.main()
