"""
Multi-disease model training pipeline for HealthScope.
Trains, evaluates, benchmarks, and automatically selects the best-performing model
for Diabetes, Heart Disease, and Hypertension datasets.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import (
    DISEASE_CONFIGS, load_dataset, preprocess_data,
    build_preprocessor, save_preprocessor, validate_dataset
)


def train_disease_models(disease_key):
    """
    Trains Logistic Regression, Decision Tree, and Random Forest for a specific disease module.
    Evaluates metrics on a 20% stratified test split and selects the best model automatically.
    """
    config = DISEASE_CONFIGS[disease_key]
    disease_name = config['display_name']
    model_dir = config['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n--- Training Models for {disease_name} ---")

    # 1. Load and validate
    df = load_dataset(disease_key)
    is_valid, msg = validate_dataset(disease_key, df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed for {disease_name}: {msg}")

    print(f"Loaded {len(df)} records with {len(config['feature_columns'])} features.")

    # 2. Preprocess & Split
    X, y = preprocess_data(df, disease_key)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Fit pipeline on X_train only to prevent data leakage
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Save preprocessor pipeline
    save_preprocessor(preprocessor, disease_key)

    # 4. Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    results = {}
    best_model_name = None
    best_score = -1.0

    for model_name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Feature importances / coefficients where available
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            feature_importance = np.abs(model.coef_[0]).tolist()

        results[model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'Confusion Matrix': cm,
            'Feature Importance': feature_importance
        }

        # Save model file
        filename = model_name.lower().replace(' ', '_') + '.joblib'
        joblib.dump(model, os.path.join(model_dir, filename))

        # Selection rule: Highest F1-score, using Accuracy as tie-breaker
        selection_score = f1 * 0.7 + acc * 0.3
        if selection_score > best_score:
            best_score = selection_score
            best_model_name = model_name

    # Save evaluation summary results
    joblib.dump(results, os.path.join(model_dir, "evaluation_results.joblib"))

    # Save best model selection metadata
    best_info = {
        'best_model_name': best_model_name,
        'selection_metric': 'F1-score & Accuracy weighted composite',
        'metrics': results[best_model_name]
    }
    joblib.dump(best_info, os.path.join(model_dir, "best_model_info.joblib"))

    print(f"Top performing model for {disease_name}: '{best_model_name}' (F1: {results[best_model_name]['F1-score']:.4f}, Accuracy: {results[best_model_name]['Accuracy']:.4f})")
    
    summary_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-score']]
    print(summary_df.map(lambda x: f"{x:.4f}"))

    return results


def train_all_models():
    """Trains and benchmarks all 3 disease modules."""
    print("==================================================")
    print("HealthScope Multi-Disease Training Pipeline")
    print("==================================================")
    all_results = {}
    for disease_key in DISEASE_CONFIGS:
        all_results[disease_key] = train_disease_models(disease_key)
    print("\n==================================================")
    print("All Disease Modules Trained & Benchmarked Successfully!")
    print("==================================================")
    return all_results


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    train_all_models()
