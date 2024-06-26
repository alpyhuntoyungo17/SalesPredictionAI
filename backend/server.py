from flask import Flask, request, jsonify
from routes.predict import predict_sales

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = predict_sales(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
