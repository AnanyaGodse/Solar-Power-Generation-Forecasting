from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the models and scalers
with open('best_model1.pkl', 'rb') as file:
    best_model1 = pickle.load(file)

with open('best_model2.pkl', 'rb') as file:
    best_model2 = pickle.load(file)

with open('scaler1.pkl', 'rb') as file:
    scaler1 = pickle.load(file)

with open('scaler2.pkl', 'rb') as file:
    scaler2 = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    data = request.json
    if 'irradiation' not in data:
        return jsonify({'error': 'Invalid input. Must contain irradiation.'}), 400

    # Extract the features for prediction
    irradiation = data['irradiation']

    # Convert the features to a numpy array 
    features = np.array([irradiation]).reshape(1, -1)
    
    # Scaling the features
    scaled_features = scaler1.transform(features)

    # Predict
    prediction = best_model1.predict(scaled_features)
    
    # Converting prediction to float
    return jsonify({'prediction': float(prediction[0])})

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    data = request.json
    required_fields = ['irradiation', 'date', 'time']

    # Ensuring that the needed fields are present
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Invalid input. Must contain irradiation, date, and time.'}), 400

    # Extract the features for prediction
    irradiation = data['irradiation']
    date_str = data['date']  # Format: 'YYYY-MM-DD'
    time_str = data['time']  # Format: 'HH:MM'

    # Convert date and time strings to a datetime object
    date_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

    # Extract year, month, and day of the week
    year = date_time.year
    month = date_time.month
    day_of_week = date_time.weekday()  # Monday is 0 and Sunday is 6

    # Convert the features into a numpy array
    features = np.array([irradiation, year, month, day_of_week]).reshape(1, -1)

    # Scaling the features
    scaled_features = scaler2.transform(features)

    # Predict
    prediction = best_model2.predict(scaled_features)

    # Convert prediction to float
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


