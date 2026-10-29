"""
Prediction module for HealthScope multi-disease analytics platform.
Handles feature validation, preprocessing, best-model loading, and risk inference.
"""

import os
import joblib
import pandas as pd
import numpy as np

from preprocessing import DISEASE_CONFIGS, load_preprocessor, clean_dataset


def get_feature_bounds(disease_key):
    """Returns realistic feature input validation limits for each disease."""
    if disease_key == 'diabetes':
        return {
            'Pregnancies': (0, 20),
            'Glucose': (0, 300),
            'BloodPressure': (0, 200),
            'SkinThickness': (0, 100),
            'Insulin': (0, 1000),
            'BMI': (0.0, 80.0),
            'DiabetesPedigreeFunction': (0.0, 3.5),
            'Age': (1, 120)
        }
    elif disease_key == 'heart_disease':
        return {
            'age': (1, 120),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (50, 250),
            'chol': (80, 600),
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (50, 230),
            'exang': (0, 1),
            'oldpeak': (0.0, 10.0),
            'slope': (0, 2),
            'ca': (0, 4),
            'thal': (0, 3)
        }
    elif disease_key == 'hypertension':
        return {
            'male': (0, 1),
            'age': (1, 120),
            'currentSmoker': (0, 1),
            'cigsPerDay': (0, 100),
            'BPMeds': (0, 1),
            'prevalentStroke': (0, 1),
            'diabetes': (0, 1),
            'totChol': (80, 600),
            'sysBP': (60, 300),
            'diaBP': (30, 200),
            'BMI': (10.0, 80.0),
            'heartRate': (30, 200),
            'glucose': (30, 500)
        }
    else:
        raise ValueError(f"Unknown disease key '{disease_key}'")


def validate_input_features(disease_key, input_dict):
    """
    Validates user input values against realistic clinical bounds.
    Returns (is_valid: bool, message: str).
    """
    bounds = get_feature_bounds(disease_key)
    for feature, (min_val, max_val) in bounds.items():
        if feature not in input_dict:
            return False, f"Missing input value for '{feature}'."
        val = input_dict[feature]
        if not isinstance(val, (int, float, np.number)):
            return False, f"Invalid type for '{feature}'. Expected numeric value."
        if val < min_val or val > max_val:
            return False, f"Value for '{feature}' ({val}) is out of reasonable range [{min_val}, {max_val}]."

    return True, "Inputs valid."


def get_best_model_info(disease_key):
    """Loads metadata for the best-performing model of the specified disease."""
    model_dir = DISEASE_CONFIGS[disease_key]['model_dir']
    info_path = os.path.join(model_dir, "best_model_info.joblib")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Best model metadata missing at '{info_path}'. Run train_models.py first.")
    return joblib.load(info_path)


def load_model(disease_key, model_name=None):
    """
    Loads a trained model for a disease.
    If model_name is None, automatically loads the best-performing model.
    """
    model_dir = DISEASE_CONFIGS[disease_key]['model_dir']
    if model_name is None:
        best_info = get_best_model_info(disease_key)
        model_name = best_info['best_model_name']

    filename = model_name.lower().replace(' ', '_') + '.joblib'
    filepath = os.path.join(model_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file missing at '{filepath}'. Run train_models.py first.")

    return joblib.load(filepath), model_name


def extract_key_risk_factors(disease_key, input_dict):
    """
    Identifies notable risk factors based on domain thresholds or high input values.
    Returns list of human-readable factor descriptions.
    """
    factors = []

    if disease_key == 'diabetes':
        if input_dict.get('Glucose', 0) >= 140:
            factors.append(f"Elevated Plasma Glucose level ({input_dict['Glucose']} mg/dL)")
        if input_dict.get('BMI', 0) >= 30:
            factors.append(f"Elevated Body Mass Index ({input_dict['BMI']:.1f} kg/m²)")
        if input_dict.get('Age', 0) >= 45:
            factors.append(f"Higher Age category ({input_dict['Age']} years)")
        if input_dict.get('BloodPressure', 0) >= 90:
            factors.append(f"Elevated Blood Pressure ({input_dict['BloodPressure']} mm Hg)")

    elif disease_key == 'heart_disease':
        if input_dict.get('cp', 0) > 0:
            factors.append(f"Reported Chest Pain (Type {input_dict['cp']})")
        if input_dict.get('trestbps', 0) >= 140:
            factors.append(f"Elevated Resting Blood Pressure ({input_dict['trestbps']} mm Hg)")
        if input_dict.get('chol', 0) >= 240:
            factors.append(f"High Serum Cholesterol ({input_dict['chol']} mg/dL)")
        if input_dict.get('exang', 0) == 1:
            factors.append("Exercise-Induced Angina present")
        if input_dict.get('oldpeak', 0) >= 1.5:
            factors.append(f"Significant ST Depression ({input_dict['oldpeak']})")

    elif disease_key == 'hypertension':
        if input_dict.get('sysBP', 0) >= 130:
            factors.append(f"Elevated Systolic Blood Pressure ({input_dict['sysBP']} mm Hg)")
        if input_dict.get('diaBP', 0) >= 85:
            factors.append(f"Elevated Diastolic Blood Pressure ({input_dict['diaBP']} mm Hg)")
        if input_dict.get('totChol', 0) >= 240:
            factors.append(f"High Total Cholesterol ({input_dict['totChol']} mg/dL)")
        if input_dict.get('currentSmoker', 0) == 1:
            factors.append(f"Active Smoking Status ({input_dict.get('cigsPerDay', 0)} cigs/day)")
        if input_dict.get('BPMeds', 0) == 1:
            factors.append("Currently taking Blood Pressure Medication")

    if not factors:
        factors.append("Clinical values within typical baseline reference ranges.")

    return factors


def predict_risk(disease_key, input_dict, model_name=None):
    """
    Executes end-to-end risk prediction workflow for a disease module.
    Automatically uses the best-performing validated model unless specified.
    """
    is_valid, msg = validate_input_features(disease_key, input_dict)
    if not is_valid:
        raise ValueError(f"Input validation error: {msg}")

    config = DISEASE_CONFIGS[disease_key]
    feature_cols = config['feature_columns']

    # Load preprocessor & model
    preprocessor = load_preprocessor(disease_key)
    model, used_model_name = load_model(disease_key, model_name)

    # Convert single input dict to DataFrame row
    df_single = pd.DataFrame([input_dict])[feature_cols]

    # Clean zeros if applicable (e.g. Diabetes)
    for col in config['zero_as_missing']:
        if df_single[col].iloc[0] == 0:
            df_single[col] = np.nan

    # Transform input
    X_scaled = preprocessor.transform(df_single)

    # Predict class & probability
    raw_pred = int(model.predict(X_scaled)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        prob_array = model.predict_proba(X_scaled)[0]
        probability = float(prob_array[1]) * 100.0

    risk_label = "HIGHER RISK" if raw_pred == 1 else "LOWER RISK"
    key_factors = extract_key_risk_factors(disease_key, input_dict)

    return {
        'disease_key': disease_key,
        'disease_name': config['display_name'],
        'risk_label': risk_label,
        'probability': probability,
        'raw_pred': raw_pred,
        'model_used': used_model_name,
        'key_factors': key_factors
    }
