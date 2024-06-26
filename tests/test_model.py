import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import load

def load_data(file_path):
    # Load test data from CSV or any other format
    data = pd.read_csv(file_path)
    return data

def test_prediction_accuracy(model_file, test_data_file):
    # Load trained model
    model = load(model_file)

    # Load test data
    test_data = load_data(test_data_file)

    # Separate features and target variable
    X_test = test_data.drop('Sales', axis=1)
    y_test = test_data['Sales']

    # Predict using the loaded model
    y_pred = model.predict(X_test)

    # Calculate and print RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Optionally, add more evaluation metrics as needed

if __name__ == "__main__":
    # Path to the trained model file (.pkl, .joblib, etc.)
    model_file = 'path/to/trained_model.pkl'

    # Path to the test data file (.csv, .xlsx, etc.)
    test_data_file = 'path/to/test_data.csv'

    # Perform model testing
    test_prediction_accuracy(model_file, test_data_file)

