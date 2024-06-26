import argparse
import pandas as pd
from model import SalesPredictionModel

def main(data_path, target_column, model_path):
    # Inisialisasi model
    model = SalesPredictionModel()
    
    # Memuat data
    data = model.load_data(data_path)
    
    # Praproses data
    data = model.preprocess_data(data)
    
    # Melatih model
    model.train(data, target_column)
    
    # Menyimpan model
    model.save_model(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sales Prediction Model')
    parser.add_argument('--data', type=str, required=True, help='Path to the sales data CSV file')
    parser.add_argument('--target', type=str, required=True, help='Name of the target column in the dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to save the trained model')
    
    args = parser.parse_args()
    
    main(args.data, args.target, args.model)

