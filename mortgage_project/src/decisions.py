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
    
    for field in required_fields:
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
    Route repayment-difficulty scores to advisory outcomes, not final verdicts.
    """
    score = validate_score(score)

    try:
        validate_application_fields(application, [
            'annual_income_gbp',
            'loan_to_value_pct',
            'monthly_debt_commitments_gbp',
            'debt_to_income_ratio',
            'credit_score',
            'employment_years',
            'missed_payments_24m',
            'credit_utilisation_pct',
        ])
    except ValueError as e:
        return {
            'status': 'Failed Check',
            'risk_level': 'Data Incomplete',
            'recommendation': 'Do not assess; resolve missing or invalid data.',
            'reason': str(e),
        }

    # The model score is the probability of repayment difficulty (class 1).
    if score >= 0.75:
        return {
            'status': 'High Risk',
            'risk_level': 'High',
            'recommendation': 'Flag for deep human review; consider structural mitigation.',
            'reason': 'High predicted likelihood of repayment difficulty.',
        }

    if score >= 0.40:
        return {
            'status': 'Review',
            'risk_level': 'Medium',
            'recommendation': 'Flag for human review and assess possible mitigation.',
            'reason': 'Moderate predicted likelihood of repayment difficulty.',
        }

    return {
        'status': 'Lower Risk',
        'risk_level': 'Low',
        'recommendation': 'Proceed to normal human review; do not treat this as approval.',
        'reason': 'Lower predicted likelihood of repayment difficulty.',
    }
