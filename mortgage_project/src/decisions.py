import math
from numbers import Real
from typing import Dict, Any, List

# Define a structure for the final decision outcome
DecisionOutcome = Dict[str, Any]


def validate_application_fields(application: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validates application fields, rejecting records missing, blank, or non-finite values.
    
    Requires fields to exist and contain finite, non-blank data.
    """
    invalid = []
    
    # CRITICAL FIX: Using the EXACT column names from the data output
    CORRECT_FIELDS = [
        'annual_income_gbp',
        'loan_to_value_pct',
        'monthly_debt_commitments_gbp',
        'debt_to_income_ratio',
        'credit_score',
        'employment_years',
        'missed_payments_24m' 
    ]
    
    for field in CORRECT_FIELDS:
        value = application.get(field)
        
        # Check for None, blank string, or non-finite numerical values
        if (value is None or
            (isinstance(value, str) and not value.strip()) or
            (isinstance(value, Real) and not math.isfinite(value))):
            invalid.append(field)
            
    if invalid:
        # Raise an error if critical data is missing, making the system fail safely
        raise ValueError(f'Missing or invalid critical fields: {invalid}. Application cannot be processed.')
    return True


def validate_score(score: Real) -> float:
    """
    Checks that a model score is numeric and falls within the required [0, 1] interval.
    """
    if not isinstance(score, Real):
        raise ValueError('Model score must be numeric.')
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Model score must be finite and between 0 and 1.')
    return float(score)


def advisory_decision(score: float, application: Dict[str, Any]) -> DecisionOutcome:
    """
    Implements the documented advisory decision rules. 
    The system provides a recommended action and rationale, NOT a final verdict, 
    thus supporting professional human judgment.
    
    The logic is engineered to prioritize Recall, ensuring high-risk applicants are flagged.
    """
    # 1. Initial Data Integrity Check (The Gatekeeper)
    try:
        # Ensure all required data is present before making a risk assessment
        validate_application_fields(application, []) # Pass empty list as fields are hardcoded inside the validator
    except ValueError as e:
        return {
            'status': 'Failed Check', 
            'risk_level': 'Data Incomplete', 
            'recommendation': 'Reject - Data Failure', 
            'reason': str(e)
        }

    # 2. Risk-Based Advisory Routing (The core decision support logic)
    
    # High Score (Score >= 0.75): Low Risk
    if score >= 0.75:
        return {
            'status': 'Recommended', 
            'risk_level': 'Low', 
            'recommendation': 'Advise Approval (Pending Final Human Review)', 
            'reason': 'High predicted likelihood of repayment. Minimal risk.'
        }
    
    # Medium Score (0.
