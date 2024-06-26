import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import SalesPredictionModel

def evaluate(model, data, target_column):
    """
    Evaluasi kinerja model pada data baru.

    :param model: SalesPredictionModel, model yang sudah dilatih
    :param data: pd.DataFrame, data untuk evaluasi
    :param target_column: str, nama kolom target
    :return: dict, hasil evaluasi
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    predictions = model.predict(X)
    
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    results = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }
    
    return results

def main(data_path, target_column, model_path):
    # Inisialisasi model
    model = SalesPredictionModel()
    
    # Memuat model
    model.load_model(model_path)
    
    # Memuat data
    data = model.load_data(data_path)
    
    # Praproses data
    data = model.preprocess_data(data)
    
    # Evaluasi model
    results = evaluate(model, data, target_column)
    
    # Menampilkan hasil evaluasi
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Sales Prediction Model')
    parser.add_argument('--data', type=str, required=True, help='Path to the evaluation data CSV file')
    parser.add_argument('--target', type=str, required=True, help='Name of the target column in the dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    
    args = parser.parse_args()
    
    main(args.data, args.target, args.model)

