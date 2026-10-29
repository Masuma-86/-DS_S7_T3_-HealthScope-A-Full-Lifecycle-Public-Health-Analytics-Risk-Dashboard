# HEALTHSCOPE – Public Health Risk & Multi-Disease Analytics Platform

**HealthScope** is an upgraded, full-lifecycle Data Science and Machine Learning platform providing public-health analytics and automated risk assessment across three major disease modules: **Diabetes**, **Heart Disease**, and **Hypertension**.

---

## 📌 Project Overview

HealthScope integrates three real-world medical datasets into a unified end-to-end data science framework. It performs automated dataset validation, feature cleaning, scaling, model training, evaluation, automatic best-model selection, and interactive risk assessment.

---

## 📊 Disease Modules & Real Datasets

### 1. Diabetes Module
- **Dataset**: Pima Indians Diabetes Dataset
- **Records**: 768 patient records
- **Features (8)**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Target**: `Outcome` (0 = No Diabetes, 1 = Diabetes)
- **Local Path**: `data/diabetes.csv`

### 2. Heart Disease Module
- **Dataset**: UCI Cleveland Heart Disease Dataset
- **Records**: 303 patient records
- **Features (13)**: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Target**: `target` (0 = No Heart Disease, 1 = Heart Disease)
- **Local Path**: `data/heart_disease.csv`

### 3. Hypertension Module
- **Dataset**: Framingham Heart Study Dataset
- **Records**: 4,240 patient records
- **Features (13)**: `male`, `age`, `currentSmoker`, `cigsPerDay`, `BPMeds`, `prevalentStroke`, `diabetes`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`
- **Target**: `prevalentHyp` (0 = No Hypertension, 1 = Hypertension)
- **Local Path**: `data/hypertension.csv`

---

## 🤖 Machine Learning Pipeline & Automatic Selection

For **each** disease module independently:
- **Models Trained**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
- **Leakage Prevention**: Preprocessing pipeline (`SimpleImputer` + `StandardScaler`) fitted strictly on training splits (80/20 stratified).
- **Evaluation**: Benchmarked using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- **Automatic Model Selection**: The system automatically determines the best-performing validated model based on F1-score and Accuracy metrics for the Risk Assessment page.

---

## 🖥️ Platform Sidebar Sections

1. **Executive Dashboard**: Multi-disease record totals (5,311 records), prevalence rates, cohort composition, and epidemiologic overview.
2. **Risk Assessment**: Disease module selection `[ Diabetes | Heart Disease | Hypertension ]`, custom disease-specific input parameters, automatic best-model inference, probability score, risk factors list, and medical disclaimer.
3. **Population Analytics**: Cohort demographic breakdowns, outcome distributions, correlation heatmaps, and descriptive statistics.
4. **Risk Factor Insights**: Model feature importance rankings and clinical parameter distributions.
5. **Model Intelligence**: Technical model benchmark comparisons, F1-scores, Accuracy metrics, and side-by-side confusion matrices.
6. **Data Quality**: Dataset integrity audit (raw vs processed counts, missing values, duplicates) and pipeline workflow diagram.
7. **Assessment Report**: Clean summary view of the most recent assessment with a downloadable `.txt` report option.
8. **Methodology**: Academic flow diagram, dataset documentation, technology stack details, and disclaimers.

---

## 📁 Project Structure

```
HealthScope/
│
├── app.py                      # Main Streamlit multi-disease dashboard app
├── requirements.txt            # Minimal required Python packages
├── README.md                   # Complete project documentation
│
├── .streamlit/
│   └── config.toml             # Theme configuration enforcing light mode
│
├── data/                       # Real public datasets
│   ├── diabetes.csv            # 768 records
│   ├── heart_disease.csv       # 303 records
│   └── hypertension.csv        # 4,240 records
│
├── models/                     # Saved model artifacts by disease module
│   ├── diabetes/
│   ├── heart_disease/
│   └── hypertension/
│
└── src/                        # Core source code modules
    ├── __init__.py
    ├── preprocessing.py        # Dataset validation & scaling pipelines
    ├── eda.py                  # Visualization functions
    ├── train_models.py         # Multi-disease model training script
    └── prediction.py           # Multi-disease inference pipeline
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional - Models are pre-trained)
```bash
python src/train_models.py
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## ⚠️ Medical Disclaimer

This platform is created strictly for **educational and data-science demonstration purposes only**. It is not a medical diagnosis or a substitute for professional clinical advice.
