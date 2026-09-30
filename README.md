# 🩺 Early Disease Prediction Using Machine Learning

## 📌 Project Overview

Early Disease Prediction is a machine learning-based web application that predicts a probable disease category from demographic, clinical, lifestyle, and symptom-related health information.

The project uses a supervised machine learning approach for multiclass classification. The trained machine learning model is integrated with a Flask backend and a web-based frontend.

> ⚠️ This project is an academic machine learning prototype. The dataset used is synthetic, and the system is not intended to provide medical diagnosis or replace professional medical advice.

---

## 🎯 Objectives

The main objectives of this project are:

- Predict a probable disease category using health-related input data.
- Apply machine learning techniques to health data.
- Compare machine learning models.
- Evaluate model performance using multiple evaluation metrics.
- Provide explainability using SHAP.
- Provide What-If analysis for exploring changes in model predictions.
- Integrate the trained model into a Flask web application.

---

## 🧠 Machine Learning Approach

The project follows a supervised learning approach.

### Problem Type

**Multiclass Classification**

The model predicts one disease category from multiple possible categories.

### Models

- Logistic Regression — Baseline Model
- Random Forest Classifier — Main Candidate Model
- Tuned Random Forest — Final candidate after hyperparameter tuning

The final model is selected based on the observed validation and test performance rather than assuming a particular algorithm is best.

---

## 📊 Dataset

The project uses a synthetic health dataset containing approximately:

**20,000 patient records**

### Input Features

The model uses the following 20 features:

#### Numerical Features

- Age
- BMI
- Systolic_BP
- Diastolic_BP
- Heart_Rate
- Glucose
- Cholesterol

#### Categorical Features

- Gender
- Smoking
- Alcohol_Use
- Family_History

#### Symptom Features

- Fever
- Cough
- Fatigue
- Headache
- Chest_Pain
- Breathing_Difficulty
- Nausea
- Joint_Pain
- Frequent_Urination

### Target

- Disease

`Patient_ID` is not used as a machine learning feature.

---

## 🔄 Project Workflow

```text
Health Dataset
      ↓
Data Exploration
      ↓
Data Cleaning
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
Data Preprocessing
      ↓
Logistic Regression
      ↓
Random Forest
      ↓
Model Evaluation
      ↓
Cross Validation
      ↓
Hyperparameter Tuning
      ↓
Final Model
      ↓
Feature Importance
      ↓
SHAP Explainability
      ↓
What-If Analysis
      ↓
Save Model
      ↓
Flask Backend
      ↓
Web Interface
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| ML Development | Google Colab |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Model Saving | Joblib |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Methodology | Agile |

---

## 🧹 Data Preprocessing

The preprocessing pipeline handles different types of input features.

### Numerical Data

Numerical features are processed using:

- Missing value imputation
- Standard scaling

### Categorical Data

Categorical features are processed using:

- Missing value imputation
- One-Hot Encoding

### Binary Features

Binary symptom features are handled using missing-value imputation while retaining their binary representation.

A preprocessing pipeline is used so that the same transformations are applied consistently during training and prediction.

---

## 🤖 Model Training

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to learn patterns, while the testing data is used to evaluate the model on unseen records.

---

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

Cross-validation is also used to obtain a more robust estimate of model performance.

---

## ⚙️ Hyperparameter Tuning

Random Forest hyperparameters are tuned using `GridSearchCV`.

Parameters considered include:

- Number of estimators
- Maximum depth
- Minimum samples split
- Minimum samples leaf
- Maximum features

The tuned model is then evaluated on the test data.

---

## 🔎 Explainable AI with SHAP

SHAP is used to explain machine learning predictions.

Instead of only showing:

```text
Predicted Disease: X
```

the system can also explain which input features influenced the model's prediction.

For example:

```text
Prediction
    ↓
SHAP Explanation
    ↓
Important contributing features
```

SHAP explains the model's behavior; it does not establish medical causation.

---

## 🔄 What-If Analysis

The project includes a What-If analysis concept.

A user can modify health-related input values and observe how the model's prediction changes.

Example:

```text
Original Patient Data
        ↓
Prediction A
        ↓
Modify Input
        ↓
Prediction B
```

This feature is intended to demonstrate how model predictions respond to changes in input data.

---

## 🌐 Web Application

The trained machine learning model is integrated into a Flask web application.

### Application Flow

```text
User
 ↓
Web Form
 ↓
Health Information
 ↓
Flask Backend
 ↓
Trained ML Model
 ↓
Prediction
 ↓
Result Display
```

The web application accepts the 20 input features required by the trained model.

---

## 📁 Project Structure

```text
Early Disease Prediction/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── dataset/
│   └── health_data.csv
│
├── models/
│   └── disease_prediction_model.pkl
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── venv/
│
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask application

```bash
python app/app.py
```

### 6. Open the application

Open the local Flask URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

---

## 🧪 Example Application Flow

The user enters:

```text
Age
Gender
BMI
Blood Pressure
Heart Rate
Glucose
Cholesterol
Lifestyle information
Family history
Symptoms
```

The application sends the information to the Flask backend.

The trained machine learning model processes the input and returns a predicted disease category.

---

## ⚠️ Limitations

- The dataset used in this academic prototype is synthetic.
- The model has not been clinically validated.
- Predictions should not be interpreted as medical diagnoses.
- Model confidence does not represent a guaranteed probability that a patient has a disease.
- SHAP explanations describe model behavior and do not establish medical causation.
- Real-world clinical deployment would require appropriate clinical validation, representative datasets, privacy protections, and regulatory review.

---

## 🔮 Future Scope

Possible future improvements include:

- Integration with validated real-world health datasets.
- Additional machine learning models.
- Improved explainability dashboards.
- More advanced What-If analysis.
- User authentication.
- Patient history and prediction tracking.
- Cloud deployment.
- Integration with healthcare systems after appropriate validation.

---

## 👨‍💻 Project Methodology

The project follows the **Agile software development methodology**.

Development is divided into iterative stages:

```text
Sprint 1 → Data Understanding & EDA
Sprint 2 → Preprocessing
Sprint 3 → Model Training & Evaluation
Sprint 4 → Explainability, Flask & Final Application
```

---

## 📌 Disclaimer

This project is developed for academic and educational purposes.

It is an experimental machine learning prototype and should not be used for medical diagnosis, treatment decisions, or emergency healthcare decisions.

---

## 📜 License

This project is intended for academic and educational use.
