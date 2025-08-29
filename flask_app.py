from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_er_model.cbm")

# Features order (must match training)
features = [
    "Gender", "Age", "BMI", "HbA1c", "Cholesterol (Total)", "ER Visits (past 12m)",
    "Vigorous Activity", "(Smoked 100 cigarettes)", "Calories", "Sugar", "Fiber",
    "Saturated Fat", "Dietary Cholesterol", "Sodium", "Potassium",
    "Systolic_BP_Avg", "Diastolic_BP_Avg"
]

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask server is running. Use POST /predict for predictions."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert categorical to numeric
    gender = 1 if data["Gender"] == "Male" else 0
    activity = 1 if data["VigorousActivity"] == "Yes" else 0
    smoked = 1 if data["Smoked"] == "Yes" else 0

    # Build dataframe for prediction
    new_data = pd.DataFrame([[
        gender,
        float(data["Age"]),
        float(data["BMI"]),
        float(data["HbA1c"]),
        float(data["Cholesterol"]),
        int(data["ER_Visits"]),
        activity,
        smoked,
        float(data["Calories"]),
        float(data["Sugar"]),
        float(data["Fiber"]),
        float(data["SatFat"]),
        float(data.get("DietaryCholesterol", 300)),  # default if missing
        float(data["Sodium"]),
        float(data["Potassium"]),
        float(data["SystolicBP"]),
        float(data["DiastolicBP"]),
    ]], columns=features)

    # Predict
    probs = model.predict_proba(new_data)[0]
    pred_class = int(model.predict(new_data)[0])

    return jsonify({
        "prediction": pred_class,
        "probabilities": {
            "No_ER_Visit": round(probs[0]*100, 2),
            "ER_Visit": round(probs[1]*100, 2)
        }
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

