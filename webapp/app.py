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
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model')
# model = joblib.load(MODEL_PATH)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_production_model.joblib')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_preprocessor.joblib')
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

df_airline_delay_rate = pd.read_csv(os.path.join(os.path.dirname(__file__), 'precalc_metrics', 'airline_delay_rate.csv'), index_col=None)
df_route_airline_delay_rate = pd.read_csv(os.path.join(os.path.dirname(__file__), 'precalc_metrics', 'route_airline_delay_rate.csv'), index_col=None)
df_route_delay_rate = pd.read_csv(os.path.join(os.path.dirname(__file__), 'precalc_metrics', 'route_delay_rate.csv'), index_col=None)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

cat_cols = ['Airline', 'AirportFrom', 'AirportTo', 'Route', 'DayOfWeek']
# num_cols = ['Flight', 'Time', 'Length', 'Airline_DelayRate', 'Route_AvgDelay']
num_cols = ['Time', 'Length', 'Airline_DelayRate', 'Route_AvgDelay']

# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), num_cols),
#     ("cat", BinaryEncoder(), cat_cols)
# ])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON or form
        if request.is_json:
            input_data = request.get_json()
        else:
            input_data = {
                'Airline': request.form['airline'],
                'AirportFrom': request.form['airport_from'],
                'AirportTo': request.form['airport_to'],
                'Route': f"{request.form['airport_from']}-{request.form['airport_to']}",
                'DayOfWeek': request.form['day_of_week'],
                # 'Flight': float(request.form['flight_number']),
                'Time': float(request.form['time']),
                'Length': float(request.form['length']),
                'Airline_DelayRate': 0.0,  
                'Route_AvgDelay': 0.0  
            }
        
        # input_data['Airline_DelayRate'] = float(df_airline_delay_rate[df_airline_delay_rate['Airline'] == input_data['Airline']]['delay_rate'])
        # print("--------------------")
        match = df_airline_delay_rate.loc[df_airline_delay_rate['Airline'] == input_data['Airline'], 'delay_rate']
        input_data['Airline_DelayRate'] = float(match.squeeze()) if not match.empty else 0.0
        # print(input_data['Airline_DelayRate'])

        match_route = df_route_delay_rate.loc[
            (df_route_delay_rate['AirportFrom'] == input_data['AirportFrom']) & 
            (df_route_delay_rate['AirportTo'] == input_data['AirportTo']),
            'delay_rate'
        ]
        input_data['Route_AvgDelay'] = float(match_route.squeeze()) if not match_route.empty else 0.0
        # print(input_data['Route_AvgDelay'])
        # print("--------------------")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        X_processed = preprocessor.transform(df)

        # Predict probabilities
        # Get probabilistic predictions
        prediction = model.predict_proba(X_processed)
        delay_prob = round(prediction[0][1] * 100, 2)
        no_delay_prob = round(prediction[0][0] * 100, 2)
        
        # Determine risk category
        if delay_prob <= 25:
            risk_category = 'Low Risk'
        elif delay_prob <= 50:
            risk_category = 'Moderate Risk'
        elif delay_prob <= 75:
            risk_category = 'High Risk'
        else:
            risk_category = 'Very High Risk'

        # -----percentage of flights that are delayed on {airline} with route {from} to {to}
        match_route_airline = df_route_airline_delay_rate.loc[
            (df_route_airline_delay_rate['Airline'] == input_data['Airline']) &
            (df_route_airline_delay_rate['AirportFrom'] == input_data['AirportFrom']) &
            (df_route_airline_delay_rate['AirportTo'] == input_data['AirportTo']),
            'delay_rate'
        ]
        route_airline_delay_rate = float(match_route_airline.squeeze()) if not match_route_airline.empty else None

        # delay rate of route according to airline + user selection
        route_mask = (
            (df_route_airline_delay_rate['AirportFrom'] == input_data['AirportFrom']) &
            (df_route_airline_delay_rate['AirportTo'] == input_data['AirportTo'])
        )
        route_airlines = df_route_airline_delay_rate[route_mask].copy()
        route_airlines = route_airlines.sort_values('delay_rate', ascending=False)
        user_airline_row = route_airlines[route_airlines['Airline'] == input_data['Airline']]
        user_airline_in_top10 = input_data['Airline'] in route_airlines.head(10)['Airline'].values
        if user_airline_in_top10:
            top10 = route_airlines.head(10)
        else:
            top10 = pd.concat([route_airlines.head(9), user_airline_row]) if not user_airline_row.empty else route_airlines.head(10)
        
        histogram_data = [
            {'airline': row['Airline'], 'delay_rate': row['delay_rate']}
            for _, row in top10.iterrows()
        ]

        # top delayed routes from user's selected airline
        airline_routes = df_route_airline_delay_rate[df_route_airline_delay_rate['Airline'] == input_data['Airline']].copy()
        top_delayed_routes = airline_routes.sort_values('delay_rate', ascending=False).head(10)
        route_histogram_data = [
            {
                'route': f"{row['AirportFrom']}-{row['AirportTo']}", 
                'delay_rate': row['delay_rate']
            }
            for _, row in top_delayed_routes.iterrows()
        ]


        return jsonify({
            'delay_probability': delay_prob,
            'risk_category': risk_category,
            'is_delayed': 'Yes' if delay_prob > 50 else 'No',
            'detailed_prediction': {
                'no_delay_prob': no_delay_prob,
                'delay_prob': delay_prob
            },
            'route_airline_delay_rate': route_airline_delay_rate,
            'histogram_data': histogram_data,
            'route_histogram_data': route_histogram_data
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
