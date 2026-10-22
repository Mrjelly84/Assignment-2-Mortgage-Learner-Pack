import math
from numbers import Real
from typing import Dict, Any, List

# Define a structure for the final decision output
DecisionOutcome = Dict[str, Any]


def validate_application_fields(application: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validates application fields, rejecting records missing, blank, or non-finite values.
    """
    invalid = []
    for field in required_fields:
        value = application.get(field)
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
    The system provides a recommended action, not a final verdict, supporting human judgment.
    """
    validate_score(score)
    
    # --- DOCUMENTED DECISION RULES START HERE ---
    
    # 1. Check Data Integrity (Primary Gatekeeper Rule)
    try:
        validate_application_fields(application, ['income', 'debt_ratio', 'credit_history'])
    except ValueError as e:
        return {
            'status': 'Rejected', 
            'risk_level': 'Data Incomplete', 
            'recommendation': 'Reject - Data Failure', 
            'reason': str(e)
        }

    # 2. Risk-Based Advisory Routing (The core decision support logic)
    if score >= 0.75:
        # Low Risk (Highly Favourable)
        return {
            'status': 'Recommended', 
            'risk_level': 'Low', 
            'recommendation': 'Advise Approval (Pending Final Human Review)', 
            'reason': 'High predicted likelihood of repayment.'
        }
    
    elif 0.40 <= score < 0.75:
        # Medium Risk (Needs Mitigating Action)
        # Here, the system provides actionable advice, NOT a denial.
        return {
            'status': 'Review', 
            'risk_level': 'Medium', 
            'recommendation': 'Flag for Human Review; Recommend Structural Mitigation (e.g., lower loan amount or co-signer).', 
            'reason': 'Moderate risk detected; human judgment is required to assess mitigation strategies.'
        }
        
    else: # score < 0.40
        # High Risk (Adverse)
        # Even here, the advice is structured, not simply a flat rejection.
        return {
            'status': 'High Risk', 
            'risk_level': 'High', 
            'recommendation': 'Advise Rejection or Deep Re-evaluation of Applicant Profile.', 
            'reason': 'Low predicted repayment likelihood requires rejection or significant policy change.'
        }

# Example of how the output is structured for the human reviewer:
# print(advisory_decision(0.8, {...})) 
