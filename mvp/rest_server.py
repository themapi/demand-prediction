import pandas as pd
from flask import Flask, jsonify, request

from mvp import data_processing, regression_model

app = Flask(__name__)

model_path = None
n_lag_days = -1


@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    data = pd.DataFrame(json)

    x, _ = data_processing.process_data(data, n_lag_days)

    model = regression_model.RandomForestForecastModel.load(model_path)

    y_hat = model.forecast(x)

    return jsonify(y_hat)
