"""Worked TWO-FEATURE mortgage baseline – teaching scaffold, not a final solution.

Learners must explain the supplied choices, evaluate their limitations,
implement and evidence independent development, and integrate/test rule logic.
"""
from pathlib import Path
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data import TARGET

# TUTOR SELECTED: two numeric predictors only, intentionally limited.
BASELINE_FEATURES = ['credit_score', 'debt_to_income_ratio']
MODEL_FILE = Path(__file__).resolve().parents[1] / 'models' / 'baseline_model.joblib'


def build_baseline_pipeline():
    """Median-impute missing inputs, scale, then fit Logistic Regression."""
    return make_pipeline(
        SimpleImputer(strategy='median'),  # Median calculated on training data
        StandardScaler(),                   # Scaling fitted on training data
        LogisticRegression(max_iter=1000, random_state=42),
    )


def evaluate(model, X_test, y_test):
    """Report results on held-out data; not a substitute for full evaluation."""
    predicted = model.predict(X_test)       # 0/1 classifications at default threshold
    scores = model.predict_proba(X_test)[:, 1]  # Class-1 scores; calibration not assessed
    return {
        'accuracy': accuracy_score(y_test, predicted),
        'precision_1': precision_score(y_test, predicted, zero_division=0),
        'recall_1': recall_score(y_test, predicted, zero_division=0),
        'f1_1': f1_score(y_test, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y_test, scores),
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0, 1]),
    }


def run_baseline(df):
    """Fit reference and logistic models using the same reproducible split."""
    X = df[BASELINE_FEATURES].copy()  # Selected input columns; no target leakage
    y = df[TARGET]                    # Outcome: observed difficulty within 12 months
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )

    reference = DummyClassifier(strategy='most_frequent')
    reference.fit(X_train, y_train)          # Always predicts training majority class

    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)           # Preprocessing and model fitted together

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)        # Save local fitted pipeline for later use
    return evaluate(reference, X_test, y_test), evaluate(pipeline, X_test, y_test)
