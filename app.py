"""
AeroPredict - Airport Operations Flight Delay Prediction & Management System
AI Pioneers Internship - Week 4 Capstone Deployment (Flask REST API + Web Dashboard)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load serialized models and scaler
classifier = joblib.load(os.path.join(MODELS_DIR, "airport_delay_classifier.joblib"))
regressor = joblib.load(os.path.join(MODELS_DIR, "airport_delay_regressor.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as f:
    METADATA = json.load(f)

@app.route("/")
def home():
    """Renders interactive web dashboard."""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for API status."""
    return jsonify({
        "status": "online",
        "service": "AeroPredict Flight Delay Inference API",
        "version": METADATA.get("version", "1.0.0"),
        "models_loaded": {
            "classifier": "RandomForestClassifier",
            "regressor": "RandomForestRegressor",
            "scaler": "StandardScaler"
        }
    }), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    REST API endpoint for real-time flight delay inference.
    Accepts JSON payload:
    {
        "airline": "IndiGo",
        "aircraft_type": "Airbus A320neo",
        "weather_condition": "Thunderstorm",
        "distance_km": 1150.0,
        "flight_duration_min": 140.0,
        "passenger_count": 180,
        "baggage_handling_min": 32.0
    }
    """
    try:
        data = request.get_json(force=True)
        
        # Extract inputs with defaults
        airline = data.get("airline", "IndiGo")
        aircraft_type = data.get("aircraft_type", "Airbus A320neo")
        weather = data.get("weather_condition", "Clear")
        distance = float(data.get("distance_km", 1100.0))
        duration = float(data.get("flight_duration_min", 135.0))
        passengers = float(data.get("passenger_count", 150))
        baggage = float(data.get("baggage_handling_min", 25.0))
        speed = distance / (duration / 60.0)

        # Scale continuous features
        raw_continuous = np.array([[distance, duration, passengers, baggage, speed]])
        scaled_continuous = scaler.transform(raw_continuous)[0]

        # Construct full feature vector matching trained model schema
        features_dict = {
            'Distance_km_scaled': scaled_continuous[0],
            'Flight_Duration_min_scaled': scaled_continuous[1],
            'Passenger_Count_scaled': scaled_continuous[2],
            'Baggage_Handling_min_scaled': scaled_continuous[3],
            'Calculated_Speed_kmh_scaled': scaled_continuous[4],
            'Airline_Akasa Air': 1 if airline == 'Akasa Air' else 0,
            'Airline_IndiGo': 1 if airline == 'IndiGo' else 0,
            'Airline_SpiceJet': 1 if airline == 'SpiceJet' else 0,
            'Airline_Vistara': 1 if airline == 'Vistara' else 0,
            'Weather_Condition_Fog': 1 if weather == 'Fog' else 0,
            'Weather_Condition_Overcast': 1 if weather == 'Overcast' else 0,
            'Weather_Condition_Rain': 1 if weather == 'Rain' else 0,
            'Weather_Condition_Thunderstorm': 1 if weather == 'Thunderstorm' else 0,
            'Aircraft_Type_Airbus A320neo': 1 if aircraft_type == 'Airbus A320neo' else 0,
            'Aircraft_Type_Airbus A321neo': 1 if aircraft_type == 'Airbus A321neo' else 0,
            'Aircraft_Type_Boeing 737 MAX': 1 if aircraft_type == 'Boeing 737 MAX' else 0,
            'Aircraft_Type_Boeing 787 Dreamliner': 1 if aircraft_type == 'Boeing 787 Dreamliner' else 0,
        }

        feature_vector = pd.DataFrame([features_dict])

        # Model Inferences
        prob_delay = float(classifier.predict_proba(feature_vector)[0, 1])
        is_delayed = bool(prob_delay >= 0.5)
        est_delay_min = float(np.round(regressor.predict(feature_vector)[0], 1))

        # Risk Classification & Operational Recommendations
        if prob_delay >= 0.65 or est_delay_min >= 40:
            risk = "High"
            summary = "⚠️ High Probability of Severe Flight Delay (>30 mins)"
            recommendation = "Adverse conditions detected (Weather/Turnaround). Issue ground hold warning, notify passengers, and prioritize apron turnaround sequence."
        elif prob_delay >= 0.35 or est_delay_min >= 20:
            risk = "Medium"
            summary = "⏳ Moderate Risk of Minor Schedule Buffer Delay"
            recommendation = "Maintain regular monitoring. Ensure expedited baggage loading and alert ground handling staff to avoid compounding delay."
        else:
            risk = "Low"
            summary = "✅ On-Time Expected Flight Operation"
            recommendation = "Standard operational procedure. Flight is cleared for on-time boarding and pushback schedule."

        return jsonify({
            "status": "success",
            "is_delayed": is_delayed,
            "delay_probability": round(prob_delay * 100, 1),
            "estimated_delay_minutes": max(0.0, est_delay_min),
            "risk_level": risk,
            "prediction_summary": summary,
            "recommendation": recommendation,
            "flight_metadata": {
                "airline": airline,
                "aircraft_type": aircraft_type,
                "weather_condition": weather,
                "calculated_speed_kmh": round(speed, 1)
            }
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    print(">>> Starting AeroPredict Flask Production Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
