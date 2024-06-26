import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from model import SalesPredictionModel

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data."""
    df['date'] = pd.to_datetime(df['date'])
    
    # Assuming we're predicting total_sales
    y = df['total_sales']
    
    features = ['day_of_week', 'month', 'category', 'store_id', 'weather_condition',
                'avg_price', 'is_weekend', 'is_holiday', 'promotion_active', 'temperature']
    X = df[features]
    
    return X, y

def evaluate_model():
    # Load the trained model
    model_path = os.path.join('models', 'sales_prediction_model.joblib')
    model = joblib.load(model_path)
    
    # Load test data
    test_data_path = os.path.join('data', 'processed', 'test_data.csv')
    df_test = load_data(test_data_path)
    
    # Preprocess test data
    X_test, y_test = preprocess_data(df_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'actual_vs_predicted.png'))
    plt.close()
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    features, importance = zip(*sorted_importance)
    plt.figure(figsize=(10, 6))
    plt.bar(features, importance)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'feature_importance.png'))
    plt.close()
    
    print("\nFeature Importance:")
    for feature, imp in sorted_importance:
        print(f"{feature}: {imp:.4f}")

if __name__ == "__main__":
    evaluate_model()
