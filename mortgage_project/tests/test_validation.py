"""Examples of testing generic helper functions; not mortgage policy tests."""
import unittest
from src.decisions import (
    advisory_decision,
    validate_application_fields,
    validate_score,
)

class GenericValidationTests(unittest.TestCase):
    def setUp(self):
        self.application = {
            'annual_income_gbp': 45000,
            'loan_to_value_pct': 80,
            'monthly_debt_commitments_gbp': 500,
            'debt_to_income_ratio': 0.32,
            'credit_score': 680,
            'employment_years': 5,
            'missed_payments_24m': 0,
            'credit_utilisation_pct': 25,
        }

    def test_valid_score(self):
        self.assertEqual(validate_score(0.4), 0.4)

    def test_score_outside_range(self):
        with self.assertRaises(ValueError):
            validate_score(1.5)

    def test_required_value_present(self):
        self.assertTrue(validate_application_fields({'credit_score': 680}, ['credit_score']))

    def test_required_value_missing(self):
        with self.assertRaises(ValueError):
            validate_application_fields({}, ['credit_score'])

    def test_decision_rule_boundaries(self):
        cases = (
            (0.399, 'Low'),
            (0.40, 'Medium'),
            (0.749, 'Medium'),
            (0.75, 'High'),
        )
        for score, expected_risk in cases:
            with self.subTest(score=score):
                result = advisory_decision(score, self.application)
                self.assertEqual(result['risk_level'], expected_risk)
                self.assertIn('human review', result['recommendation'].lower())

    def test_decision_rejects_invalid_score(self):
        with self.assertRaises(ValueError):
            advisory_decision(1.5, self.application)

if __name__ == '__main__':
    unittest.main()
