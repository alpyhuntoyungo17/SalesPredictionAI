import requests
import json

def test_predict_sales_api():
    url = 'http://localhost:3000/predict'  # Ganti dengan URL API yang sesuai
    headers = {'Content-Type': 'application/json'}
    data = [
        {"Store": 1, "ProductA": 1, "ProductB": 0, "Promotion": 1, "Holiday": 0, "Weather_Sunny": 1, "Weather_Rainy": 0, "Weather_Cloudy": 0},
        {"Store": 2, "ProductA": 0, "ProductB": 1, "Promotion": 0, "Holiday": 1, "Weather_Sunny": 0, "Weather_Rainy": 1, "Weather_Cloudy": 0}
    ]
    
    # Mengirim permintaan POST ke API
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Memastikan respons status sukses (200 OK)
    assert response.status_code == 200, f"Failed: Expected status 200 but got {response.status_code}"
    
    # Memastikan data JSON yang valid
    try:
        json_response = response.json()
        assert 'predictions' in json_response, "Failed: 'predictions' key not found in response"
        predictions = json_response['predictions']
        assert len(predictions) == len(data), f"Failed: Expected {len(data)} predictions but got {len(predictions)}"
        print("API Test Passed Successfully!")
    except json.JSONDecodeError:
        assert False, "Failed: Response is not valid JSON"

if __name__ == "__main__":
    test_predict_sales_api()

