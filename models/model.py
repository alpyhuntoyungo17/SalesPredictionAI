import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(training_data):
    X = training_data.drop(columns=['target'])
    y = training_data['target']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, input_data):
    return model.predict(input_data)
