from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize Flask app
app_scaled = Flask(__name__)

# 💡 CRITICAL: Define the exact sequence of features your model was trained on
FEATURE_COLUMNS = ['age', 'income', 'credit_score']  # <-- CHANGE THESE to your actual training feature names

# Resolve paths safely relative to this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_regression_model_scaled.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

# Load the trained scaled model and scaler
try:
    model_scaled_api = joblib.load(MODEL_PATH)
    scaler_api = joblib.load(SCALER_PATH)
    print("Scaled model and scaler loaded successfully for API.")
except Exception as e:
    print(f"Error loading scaled model or scaler for API: {e}")
    model_scaled_api = None
    scaler_api = None 

@app_scaled.route('/predict_scaled', methods=['POST'])
def predict_scaled():
    if model_scaled_api is None or scaler_api is None:
        return jsonify({'error': 'Scaled model or scaler not loaded on the server.'}), 500

    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Ensure all required features are provided in the payload
        missing_features = [feat for feat in FEATURE_COLUMNS if feat not in data]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Convert input data to a DataFrame and strictly enforce column order
        input_df = pd.DataFrame([data])
        input_df = input_df[FEATURE_COLUMNS] 

        # Scale the input data
        input_scaled = scaler_api.transform(input_df)

        # Make prediction
        prediction = model_scaled_api.predict(input_scaled)
        prediction_proba = model_scaled_api.predict_proba(input_scaled)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Entry point for running the server
if __name__ == '__main__':
    app_scaled.run(host='0.0.0.0', port=5000, debug=True)
