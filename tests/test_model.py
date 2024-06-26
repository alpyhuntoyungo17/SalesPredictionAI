import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import SalesPredictionModel

class TestSalesPredictionModel(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        self.data = pd.DataFrame({
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'month': [1, 2, 3, 4, 5],
            'category': ['Electronics', 'Clothing', 'Food', 'Electronics', 'Clothing'],
            'store_id': ['S001', 'S002', 'S003', 'S001', 'S002'],
            'weather_condition': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy'],
            'avg_price': [100.0, 50.0, 25.0, 120.0, 60.0],
            'is_weekend': [0, 0, 0, 0, 0],
            'is_holiday': [0, 0, 1, 0, 0],
            'promotion_active': [1, 0, 1, 0, 1],
            'temperature': [25.0, 18.0, 22.0, 27.0, 20.0],
            'total_sales': [1000, 500, 750, 1200, 600]
        })
        
        # Split the data into features (X) and target (y)
        self.X = self.data.drop('total_sales', axis=1)
        self.y = self.data['total_sales']
        
        # Initialize the model
        self.model = SalesPredictionModel()

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.preprocessor)

    def test_build_model(self):
        self.model.build_model()
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.preprocessor)

    def test_train_and_predict(self):
        # Build and train the model
        self.model.build_model()
        self.model.train(self.X, self.y)

        # Make predictions
        predictions = self.model.predict(self.X)

        # Check if predictions have the correct shape
        self.assertEqual(len(predictions), len(self.y))

        # Check if predictions are within a reasonable range
        self.assertTrue(all(0 <= pred <= max(self.y) * 1.5 for pred in predictions))

    def test_feature_importance(self):
        # Build and train the model
        self.model.build_model()
        self.model.train(self.X, self.y)

        # Get feature importance
        importance = self.model.get_feature_importance()

        # Check if all features are present in the importance dict
        self.assertEqual(set(importance.keys()), set(self.X.columns))

        # Check if importance values sum up to 1 (or close to 1 due to floating-point arithmetic)
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)

    def test_model_performance(self):
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Build and train the model
        self.model.build_model()
        self.model.train(X_train, y_train)

        # Make predictions on test set
        predictions = self.model.predict(X_test)

        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(predictions - y_test))

        # Check if MAE is within a reasonable range (this threshold might need adjustment)
        self.assertLess(mae, 500)  # Assuming sales are in the range of thousands

    def test_invalid_input(self):
        # Build the model
        self.model.build_model()

        # Test with missing feature
        invalid_data = self.X.drop('temperature', axis=1)
        with self.assertRaises(ValueError):
            self.model.predict(invalid_data)

        # Test with invalid data type
        invalid_data = self.X.copy()
        invalid_data['month'] = invalid_data['month'].astype(str)
        with self.assertRaises(ValueError):
            self.model.predict(invalid_data)

if __name__ == '__main__':
    unittest.main()
