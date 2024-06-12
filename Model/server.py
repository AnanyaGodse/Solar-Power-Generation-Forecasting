from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the models
model1_path = os.path.join(current_dir, 'model1.pkl')
model2_path = os.path.join(current_dir, 'model2.pkl')

model1 = pickle.load(open(model1_path, 'rb'))
model2 = pickle.load(open(model2_path, 'rb'))

# Define route for rendering the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Define prediction endpoint for Model 1
@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    # Get the JSON data from the request
    data = request.get_json()
    
    print("Received data for Model 1:", data)
    # Extract hour from the JSON data
    hour = int(data['hour'])
    
    # Predict for model 1
    prediction = model1.predict(np.array([[hour]]))[0]
    
    # Create response JSON
    response = {
        'prediction': prediction.tolist()
    }
    
    return jsonify(response)

# Define prediction endpoint for Model 2
@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    # Get the JSON data from the request
    data = request.get_json()
    
    print("Received data for Model 2:", data)
    # Extract features from the JSON data
    hour = int(data['hour'])
    dayOfWeek = int(data['dayOfWeek'])
    weekOfYear = int(data['weekOfYear'])
    
    # Predict for model 2
    prediction = model2.predict(np.array([[weekOfYear, dayOfWeek, hour]]))[0]
    
    # Create response JSON
    response = {
        'prediction': prediction.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
