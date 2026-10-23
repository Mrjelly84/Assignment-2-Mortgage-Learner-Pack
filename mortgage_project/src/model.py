# model.py - Improved ML Pipeline
"""
Developed  to provide a robust, non-linear model 
for mortgage risk prediction, moving beyond the limitations of the baseline.
"""
from pathlib import Path
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier # <<< THE NEW MODEL
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data import TARGET
import numpy as np

# --- MODEL CONFIGURATION ---
# This list contains strong risk indicators, replacing the old two-feature limitation.
IMPROVED_FEATURES = [
    'annual_income_gbp',
    'loan_to_value_pct',
    'monthly_debt_commitments_gbp',
    'debt_to_income_ratio',
    'credit_score',
    'employment_years',
    'missed_payments_24m',
    'credit_utilisation_pct',
    # Add other powerful features here if needed (e.g., savings_gbp)
]
MODEL_FILE = Path(__file__).resolve().parents[1] / 'models' / 'improved_model.joblib'


def build_improved_pipeline():
    """
    Defines the robust pipeline: Impute missing data -> Scale -> Train Random Forest.
    """
    return Pipeline(steps=[
        # Step 1: Handle Missing Data
        ('imputer', SimpleImputer(strategy='median')),
        
        # Step 2: Scale Numerical Data
        ('scaler', StandardScaler()),
        
        # Step 3: The Core Predictive Model
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])


def evaluate(model, X_test, y_test):
    """Reports metrics on held-out data."""
    predicted = model.predict(X_test)
    # Predict probabilities for the minority class (Class 1)
    scores = model.predict_proba(X_test)[:, 1]
    
    # Note: The metrics dictionary structure is maintained from your original code.
    return {
        'accuracy': accuracy_score(y_test, predicted),
        'precision_1': precision_score(y_test, predicted, zero_division=0),
        'recall_1': recall_score(y_test, predicted, zero_division=0),
        'f1_1': f1_score(y_test, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y_test, scores),
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0, 1]),
    }


def run_improved_model(df):
    """Fits the improved Random Forest model."""
    X = df[IMPROVED_FEATURES].copy()
    y = df[TARGET]

    # Split data (keeping the stratified split is essential)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )

    # 1. Reference Model (Same as before)
    reference = DummyClassifier(strategy='most_frequent')
    reference.fit(X_train, y_train)
    
    # 2. Improved Model Training
    pipeline = build_improved_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save the new model
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    
    # Return the evaluation results for both models
    return evaluate(reference, X_test, y_test), evaluate(pipeline, X_test, y_test)

