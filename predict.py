from flask import Flask, request, jsonify
import pickle 
import numpy as np

#Load Random Forest Model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
    

app = Flask(__name__)

#Home route
@app.route('/')
def home():
    return {'message': 'Random Forest Prediction API is running'}

#Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    features = np.array(data['features']).reshape(1, -1)
    
    
    # Predict class
    prediction = int(best_model.predict(features)[0])
    
    #Predict probability
    prob = float(best_model.predict_proba(features)[0][1])
    
    return jsonify({
        'prediction' : prediction,
        'probability': prob
    })



#Run Flask app
if __name__ =='__main__':
    app.run(host='0.0.0.0', port=8000)
    
    
  