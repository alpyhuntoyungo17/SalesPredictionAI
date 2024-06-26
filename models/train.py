import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from model import SalesPredictionModel

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create target variable (assuming we're predicting total_sales)
    y = df['total_sales']
    
    # Select features for X
    features = ['day_of_week', 'month', 'category', 'store_id', 'weather_condition',
                'avg_price', 'is_weekend', 'is_holiday', 'promotion_active', 'temperature']
    X = df[features]
    
    return X, y

def train_model():
    # Load data
    data_path = os.path.join('data', 'processed', 'processed_data.csv')
    df = load_data(data_path)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and build the model
    model = SalesPredictionModel()
    model.build_model()
    
    # Train the model
    model.train(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Save the model
    model_path = os.path.join('models', 'sales_prediction_model.joblib')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_model()
