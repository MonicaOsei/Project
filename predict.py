from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load your trained model
MODEL_PATH = 'best_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"{MODEL_PATH} not found. Make sure it's in the same folder as predict.py")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return jsonify({'message': 'Random Forest Prediction API is running'})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input, expected JSON with key "features"'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    
    # Check if feature count matches the model
    expected_features = best_model.n_features_in_
    if features.shape[1] != expected_features:
        return jsonify({
            'error': f'Expected {expected_features} features, got {features.shape[1]}'
        }), 400
    
    # Predict class
    prediction = int(best_model.predict(features)[0])
    # Predict probability
    prob = float(best_model.predict_proba(features)[0][1])
    
    return jsonify({
        'prediction': prediction,
        'probability': prob
    })

# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Render assigns PORT automatically
    app.run(host='0.0.0.0', port=port)
