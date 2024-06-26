import argparse
import pandas as pd
from model import SalesPredictionModel

def main(model_path, data_path, output_path):
    # Inisialisasi model
    model = SalesPredictionModel()
    
    # Memuat model
    model.load_model(model_path)
    
    # Memuat data baru
    data = model.load_data(data_path)
    
    # Praproses data
    data = model.preprocess_data(data)
    
    # Membuat prediksi
    predictions = model.predict(data)
    
    # Menyimpan prediksi ke file CSV
    output = pd.DataFrame(predictions, columns=['Predicted_Sales'])
    output.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Sales Predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the new data CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the predictions CSV file')
    
    args = parser.parse_args()
    
    main(args.model, args.data, args.output)

