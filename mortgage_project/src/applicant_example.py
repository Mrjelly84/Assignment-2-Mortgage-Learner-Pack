"""Week 7 mortgage scaffold: score ONE sample new applicant.

Tutor provided: shows loading a fitted mortgage model, formatting the same
features in the same order, and predicting a class-1 score.
NOT provided: learner's mortgage-specific validation and decision rules.

Run `python -m src.main` first, from the project root.
Then `python -m src.applicant_example`.
"""
import joblib
import pandas as pd

from src.model import MODEL_FILE, IMPROVED_FEATURES
from src.decisions import validate_application_fields, validate_score

# Sample values for the improved model. Do not use actual customer information.
EXAMPLE_APPLICATION = {
    "annual_income_gbp": 45000,
    "loan_to_value_pct": 80,
    "monthly_debt_commitments_gbp": 500,
    "debt_to_income_ratio": 0.32,
    "credit_score": 680,
    "employment_years": 5,
    "missed_payments_24m": 0,
    "credit_utilisation_pct": 25,
}


def score_example(application):
    """Generate model's estimated probability for class 1; no advice."""
    validate_application_fields(application, IMPROVED_FEATURES)
    row = pd.DataFrame([application], columns=IMPROVED_FEATURES)
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            "No trained model: first run python -m src.main"
        )
    # Load only the trusted, locally generated model artefact.
    model = joblib.load(MODEL_FILE)
    score = model.predict_proba(row)[0, 1]
    return validate_score(float(score))


def main():
    score = score_example(EXAMPLE_APPLICATION)
    print("MORTGAGE APPLICATION ASSESSMENT")
    print("Applicant details:")
    print(f"credit_score={EXAMPLE_APPLICATION['credit_score']}")
    print(f"debt_to_income_ratio={EXAMPLE_APPLICATION['debt_to_income_ratio']:.2f}")
    print(f"Repayment-difficulty class-1 score: {score:.3f}")
    print("Calibration not assessed; this is not a validated PD.")
    print("Provisional recommendation: PENDING — rules not yet implemented")
    print("Status: AWAITING HUMAN REVIEW — no mortgage approved or declined")
    print("NEXT: implement and test your own decision rules from Week 6")


if __name__ == "__main__":
    main()
