"""
Preprocessing module for HealthScope multi-disease analytics platform.
Supports Diabetes, Heart Disease, and Hypertension dataset validation, cleaning, and transformation pipelines.
"""

import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Metadata for supported disease modules
DISEASE_CONFIGS = {
    'diabetes': {
        'display_name': 'Diabetes',
        'filepath': 'data/diabetes.csv',
        'target_column': 'Outcome',
        'expected_columns': [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ],
        'feature_columns': [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ],
        'zero_as_missing': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
        'model_dir': 'models/diabetes'
    },
    'heart_disease': {
        'display_name': 'Heart Disease',
        'filepath': 'data/heart_disease.csv',
        'target_column': 'target',
        'expected_columns': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ],
        'feature_columns': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ],
        'zero_as_missing': [],
        'model_dir': 'models/heart_disease'
    },
    'hypertension': {
        'display_name': 'Hypertension',
        'filepath': 'data/hypertension.csv',
        'target_column': 'prevalentHyp',
        'expected_columns': [
            'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
            'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
            'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
        ],
        'feature_columns': [
            'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
            'prevalentStroke', 'diabetes', 'totChol', 'sysBP', 'diaBP',
            'BMI', 'heartRate', 'glucose'
        ],
        'zero_as_missing': [],
        'model_dir': 'models/hypertension'
    }
}


def validate_dataset(disease_key, filepath_or_df=None):
    """
    Validates dataset structure, column schemas, non-emptiness, and target column.
    """
    if disease_key not in DISEASE_CONFIGS:
        return False, f"Unknown disease module '{disease_key}'."

    config = DISEASE_CONFIGS[disease_key]
    target_path = filepath_or_df if filepath_or_df is not None else config['filepath']

    if isinstance(target_path, str):
        if not os.path.exists(target_path):
            return False, f"Dataset file not found at path: {target_path}"
        try:
            df = pd.read_csv(target_path)
        except Exception as e:
            return False, f"Failed to read CSV dataset: {str(e)}"
    elif isinstance(target_path, pd.DataFrame):
        df = target_path
    else:
        return False, "Invalid dataset input type. Expected file path or pandas DataFrame."

    if df.empty:
        return False, "Dataset is empty (0 records)."

    missing_cols = [col for col in config['expected_columns'] if col not in df.columns]
    if missing_cols:
        return False, f"Dataset is missing required columns: {missing_cols}"

    target_col = config['target_column']
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' missing from dataset."

    return True, f"Dataset validation successful for {config['display_name']}."


def load_dataset(disease_key):
    """
    Loads and validates the specified disease dataset.
    """
    if disease_key not in DISEASE_CONFIGS:
        raise ValueError(f"Unknown disease key '{disease_key}'")

    config = DISEASE_CONFIGS[disease_key]
    is_valid, msg = validate_dataset(disease_key, config['filepath'])
    if not is_valid:
        raise ValueError(msg)
    
    return pd.read_csv(config['filepath'])


def clean_dataset(df, disease_key):
    """
    Handles missing/implausible zero values and removes exact duplicates where applicable.
    """
    config = DISEASE_CONFIGS[disease_key]
    df_clean = df.copy()

    # Drop exact duplicates if present
    df_clean = df_clean.drop_duplicates()

    # Handle medically implausible zero values for specific datasets (e.g. Diabetes)
    for col in config['zero_as_missing']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)

    return df_clean


def build_preprocessor():
    """
    Builds an sklearn Pipeline with median imputation and standard scaling.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def preprocess_data(df, disease_key):
    """
    Preprocesses dataset into clean feature matrix X and target array y.
    """
    config = DISEASE_CONFIGS[disease_key]
    df_clean = clean_dataset(df, disease_key)
    X = df_clean[config['feature_columns']]
    y = df_clean[config['target_column']]
    return X, y


def save_preprocessor(preprocessor, disease_key):
    """
    Saves preprocessor pipeline object for the specified disease.
    """
    model_dir = DISEASE_CONFIGS[disease_key]['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, filepath)


def load_preprocessor(disease_key):
    """
    Loads preprocessor pipeline for the specified disease.
    """
    model_dir = DISEASE_CONFIGS[disease_key]['model_dir']
    filepath = os.path.join(model_dir, "preprocessor.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor missing at '{filepath}'. Please run train_models.py first.")
    return joblib.load(filepath)
