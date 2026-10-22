"""Examples of testing generic helper functions; not mortgage policy tests."""
import unittest
from src.decisions import validate_application_fields, validate_score

class GenericValidationTests(unittest.TestCase):
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

# LEARNER: write tests for your own decision table, including invalid and boundary cases.
if __name__ == '__main__':
    unittest.main()
