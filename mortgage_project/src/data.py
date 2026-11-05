

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


def print_model_results(results):
    """Print the held-out model metrics as a compact table."""
    measures = (
        ('Accuracy', 'accuracy'),
        ('Precision for class 1', 'precision_1'),
        ('Recall for class 1', 'recall_1'),
        ('F1 for class 1', 'f1_1'),
        ('ROC AUC', 'roc_auc'),
    )
    print('\nImproved model results:')
    print('| Measure               | Result |')
    print('| --------------------- | -----: |')
    for label, key in measures:
        print(f'| {label:<21} | {results[key]:6.3f} |')


if __name__ == '__main__':
    data = load_data()
    print_audit(data)

    from src.model import run_improved_model

    _, model_results = run_improved_model(data)
    print_model_results(model_results)
