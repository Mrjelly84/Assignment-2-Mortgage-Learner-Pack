"""Load and inspect the supplied synthetic mortgage observations.

Week 7 teaching scaffold. The audit provides evidence for learners to
interpret; it does not select features or complete the assessed data analysis.
"""
from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / 'data' / 'mortgage_data.csv'
TARGET = 'repayment_difficulty_12m'


def load_data(path=DATA_FILE):
    """Load CSV and confirm the identifier and observed target exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Mortgage dataset not found: {path}')
    df = pd.read_csv(path)
    missing = {'application_id', TARGET} - set(df.columns)
    if missing:
        raise ValueError(f'Required columns missing: {sorted(missing)}')
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError('Expected a complete binary repayment_difficulty_12m target')
    return df


def print_audit(df):
    """Show factual checks; learners must explain what the findings mean."""
    print('Rows and columns:', df.shape)
    print('\nData types:\n', df.dtypes.to_string())
    print('\nMissing values:\n', df.isna().sum().to_string())
    print('\nTarget distribution:\n', df[TARGET].value_counts().sort_index().to_string())
    print('\nDuplicate application IDs:', df['application_id'].duplicated().sum())
    # LEARNER: investigate outliers, meaning, availability and fairness-related proxies.


if __name__ == '__main__':
    print_audit(load_data())
