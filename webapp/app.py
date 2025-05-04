import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'log_reg_acc_6549')
model = joblib.load(MODEL_PATH)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Preprocessing configuration
cat_cols = ['Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
num_cols = ['Flight', 'Time', 'Length', 'Airline_DelayRate', 'Route_AvgDelay']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", BinaryEncoder(), cat_cols)
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON
        data = request.get_json()
        input_data = {
            'Airline': data['Airline'],
            'AirportFrom': data['AirportFrom'],
            'AirportTo': data['AirportTo'],
            'Route': f"{data['AirportFrom']}-{data['AirportTo']}",
            'DayOfWeek': int(data['DayOfWeek']),
            'Flight': float(data['FlightNumber']),
            'Time': float(data['Time']),
            'Length': float(data['Length']),
            'Airline_DelayRate': 0.5,  # Default value, should be calculated dynamically
            'Route_AvgDelay': 0.5  # Default value, should be calculated dynamically
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict_proba(df)[0]
        delay_prob = prediction[1] * 100  # Probability of delay

        return jsonify({
            'delay_probability': round(delay_prob, 2),
            'is_delayed': 'Yes' if delay_prob > 50 else 'No'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
