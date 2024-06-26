from flask import Flask, request, jsonify
import pandas as pd
from model import SalesPredictionModel

app = Flask(__name__)

# Inisialisasi model
model = SalesPredictionModel()
model.load_model('path/to/trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan data dari permintaan POST
        data = request.get_json()
        
        # Mengonversi data menjadi DataFrame
        df = pd.DataFrame(data)
        
        # Praproses data
        df = model.preprocess_data(df)
        
        # Membuat prediksi
        predictions = model.predict(df)
        
        # Mengonversi hasil prediksi menjadi list
        predictions_list = predictions.tolist()
        
        # Mengembalikan hasil prediksi dalam format JSON
        return jsonify({'predictions': predictions_list})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

