from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "disease_prediction_model.pkl"
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # Get data from HTML form
        # ------------------------------------------

        data = {
            "Age": float(request.form["Age"]),
            "Gender": request.form["Gender"],
            "BMI": float(request.form["BMI"]),
            "Systolic_BP": float(request.form["Systolic_BP"]),
            "Diastolic_BP": float(request.form["Diastolic_BP"]),
            "Heart_Rate": float(request.form["Heart_Rate"]),
            "Glucose": float(request.form["Glucose"]),
            "Cholesterol": float(request.form["Cholesterol"]),

            "Smoking": request.form["Smoking"],
            "Alcohol_Use": request.form["Alcohol_Use"],
            "Family_History": request.form["Family_History"],

            "Fever": int(request.form["Fever"]),
            "Cough": int(request.form["Cough"]),
            "Fatigue": int(request.form["Fatigue"]),
            "Headache": int(request.form["Headache"]),
            "Chest_Pain": int(request.form["Chest_Pain"]),
            "Breathing_Difficulty": int(request.form["Breathing_Difficulty"]),
            "Nausea": int(request.form["Nausea"]),
            "Joint_Pain": int(request.form["Joint_Pain"]),
            "Frequent_Urination": int(request.form["Frequent_Urination"])
        }

        # ------------------------------------------
        # Convert input into DataFrame
        # ------------------------------------------

        input_data = pd.DataFrame([data])

        # ------------------------------------------
        # Make prediction
        # ------------------------------------------

        prediction = model.predict(input_data)[0]

        # Convert prediction to normal Python string
        prediction = str(prediction)

        # ------------------------------------------
        # Get prediction probability if available
        # ------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability = round(
                float(max(probabilities)) * 100,
                2
            )

        # ------------------------------------------
        # If request came from API
        # ------------------------------------------

        if request.is_json:

            response = {
                "prediction": prediction
            }

            if probability is not None:
                response["confidence"] = probability

            return jsonify(response)

        # ------------------------------------------
        # Show result on website
        # ------------------------------------------

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=probability
        )

    except Exception as e:

        # Show error on website
        if not request.is_json:

            return render_template(
                "index.html",
                error=str(e)
            )

        # Return error for API request
        return jsonify({
            "error": str(e)
        }), 400


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)