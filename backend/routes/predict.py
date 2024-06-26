def predict_sales(data):
    # Di sini, Anda dapat menulis logika untuk melakukan prediksi berdasarkan data yang diterima
    # Misalnya, untuk contoh sederhana, kita akan mengembalikan data yang sama sebagai prediksi
    predicted_revenue = data['units'] * 10  # Contoh prediksi sederhana: harga per unit adalah $10
    prediction = {
        'product': data['product'],
        'units': data['units'],
        'revenue': predicted_revenue
    }
    return prediction
