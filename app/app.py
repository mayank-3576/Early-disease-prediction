from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import shap
import os


app = Flask(__name__)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "disease_prediction_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# =========================================================
# GET PIPELINE COMPONENTS
# =========================================================

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]


# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(classifier)


# =========================================================
# FEATURE NAMES AFTER PREPROCESSING
# =========================================================

feature_names = preprocessor.get_feature_names_out()


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# PREDICT ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =====================================================
        # GET FORM DATA
        # =====================================================

        data = {

            "Age": float(
                request.form["Age"]
            ),

            "Gender": request.form["Gender"],

            "BMI": float(
                request.form["BMI"]
            ),

            "Systolic_BP": float(
                request.form["Systolic_BP"]
            ),

            "Diastolic_BP": float(
                request.form["Diastolic_BP"]
            ),

            "Heart_Rate": float(
                request.form["Heart_Rate"]
            ),

            "Glucose": float(
                request.form["Glucose"]
            ),

            "Cholesterol": float(
                request.form["Cholesterol"]
            ),

            "Smoking": request.form["Smoking"],

            "Alcohol_Use": request.form["Alcohol_Use"],

            "Family_History": request.form["Family_History"],

            "Fever": int(
                request.form["Fever"]
            ),

            "Cough": int(
                request.form["Cough"]
            ),

            "Fatigue": int(
                request.form["Fatigue"]
            ),

            "Headache": int(
                request.form["Headache"]
            ),

            "Chest_Pain": int(
                request.form["Chest_Pain"]
            ),

            "Breathing_Difficulty": int(
                request.form["Breathing_Difficulty"]
            ),

            "Nausea": int(
                request.form["Nausea"]
            ),

            "Joint_Pain": int(
                request.form["Joint_Pain"]
            ),

            "Frequent_Urination": int(
                request.form["Frequent_Urination"]
            )
        }


        # =====================================================
        # CREATE DATAFRAME
        # =====================================================

        input_data = pd.DataFrame([data])


        # =====================================================
        # PREDICTION
        # =====================================================

        prediction = model.predict(input_data)[0]

        prediction = str(prediction)


        print()
        print("========================================")
        print(">>> PREDICTION ROUTE REACHED <<<")
        print(">>> PREDICTION:", prediction)
        print("========================================")


        # =====================================================
        # PROBABILITY
        # =====================================================

        probability = None

        class_probabilities = []


        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]


            print()
            print("===== CLASS PROBABILITIES =====")


            # -------------------------------------------------
            # CREATE PROBABILITY DATA FOR FRONTEND
            # -------------------------------------------------

            for class_name, prob in zip(
                classifier.classes_,
                probabilities
            ):

                percentage = round(
                    float(prob) * 100,
                    2
                )


                print(
                    f"{class_name}: {percentage:.2f}%"
                )


                class_probabilities.append({

                    "class": str(class_name),

                    "probability": percentage

                })


            print(
                "==============================="
            )


            # -------------------------------------------------
            # HIGHEST CLASS PROBABILITY
            # -------------------------------------------------

            probability = round(
                float(max(probabilities)) * 100,
                2
            )


        # =====================================================
        # SHAP PREPROCESSING
        # =====================================================

        transformed_input = preprocessor.transform(
            input_data
        )


        print()
        print(">>> SHAP CALCULATION STARTED <<<")


        # =====================================================
        # CALCULATE SHAP VALUES
        # =====================================================

        shap_values = explainer.shap_values(
            transformed_input
        )


        # =====================================================
        # FIND PREDICTED CLASS INDEX
        # =====================================================

        class_index = list(
            classifier.classes_
        ).index(
            prediction
        )


        # =====================================================
        # SHAP VALUES FOR PREDICTED CLASS
        # =====================================================

        values = shap_values[
            0,
            :,
            class_index
        ]


        # =====================================================
        # CREATE SHAP DATAFRAME
        # =====================================================

        explanation = pd.DataFrame({

            "Feature": feature_names,

            "SHAP_Value": values

        })


        # =====================================================
        # ABSOLUTE SHAP IMPORTANCE
        # =====================================================

        explanation["Absolute_SHAP"] = (
            explanation["SHAP_Value"].abs()
        )


        # =====================================================
        # SORT FEATURES
        # =====================================================

        explanation = explanation.sort_values(

            by="Absolute_SHAP",

            ascending=False

        )


        # =====================================================
        # TOP 5 SHAP FEATURES
        # =====================================================

        top_features = explanation.head(5)


        shap_features = []


        for _, row in top_features.iterrows():

            feature = row["Feature"]


            # -------------------------------------------------
            # REMOVE PREPROCESSING PREFIXES
            # -------------------------------------------------

            feature = (
                feature
                .replace("num__", "")
                .replace("cat__", "")
                .replace("bin__", "")
            )


            shap_value = float(
                row["SHAP_Value"]
            )


            # -------------------------------------------------
            # DIRECTION
            # -------------------------------------------------

            if shap_value >= 0:

                direction = "toward"

            else:

                direction = "away"


            # -------------------------------------------------
            # ADD TO SHAP LIST
            # -------------------------------------------------

            shap_features.append({

                "feature": feature,

                "value": round(
                    shap_value,
                    4
                ),

                "direction": direction

            })


        # =====================================================
        # DEBUG OUTPUT
        # =====================================================

        print()
        print("========================================")
        print("Prediction:", prediction)
        print("Predicted Class Probability:", probability)

        print()
        print("CLASS PROBABILITIES:")
        print(class_probabilities)

        print()
        print("SHAP FEATURES:")
        print(shap_features)

        print("========================================")
        print()


        # =====================================================
        # SEND DATA TO FRONTEND
        # =====================================================

        return render_template(

            "index.html",

            prediction=prediction,

            confidence=probability,

            class_probabilities=class_probabilities,

            shap_features=shap_features

        )


    # =========================================================
    # ERROR HANDLING
    # =========================================================

    except Exception as e:

        print()
        print("========================================")
        print("ERROR:")
        print(e)
        print("========================================")
        print()


        return render_template(

            "index.html",

            error=str(e)

        )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )