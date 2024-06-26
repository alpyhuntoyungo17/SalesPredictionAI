import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

class SalesPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def load_data(self, file_path):
        """
        Load the sales data from a CSV file.

        :param file_path: str, path to the CSV file containing sales data
        :return: pd.DataFrame, sales data
        """
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        """
        Preprocess the sales data.

        :param data: pd.DataFrame, raw sales data
        :return: pd.DataFrame, preprocessed sales data
        """
        # Fill missing values with the median value of each column
        data = data.fillna(data.median())

        # Convert categorical columns to numerical
        data = pd.get_dummies(data, drop_first=True)

        return data

    def train(self, data, target_column):
        """
        Train the model on the sales data.

        :param data: pd.DataFrame, sales data
        :param target_column: str, the name of the target column
        :return: None
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f'Mean Absolute Error: {mae}')

    def predict(self, data):
        """
        Make predictions on new data.

        :param data: pd.DataFrame, new sales data
        :return: np.ndarray, predicted sales
        """
        predictions = self.model.predict(data)
        return predictions

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        :param file_path: str, path to save the model file
        :return: None
        """
        joblib.dump(self.model, file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        """
        Load a trained model from a file.

        :param file_path: str, path to the model file
        :return: None
        """
        self.model = joblib.load(file_path)
        print(f'Model loaded from {file_path}')

