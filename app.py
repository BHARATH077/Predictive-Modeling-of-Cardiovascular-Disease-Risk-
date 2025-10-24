# =========================================
# Flask API for Cardiovascular Risk Prediction
# =========================================
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

# Load model
model = joblib.load("lightgbm_heart_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json(force=True)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        response = {
            "prediction": int(prediction),
            "risk_probability": float(probability)
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
