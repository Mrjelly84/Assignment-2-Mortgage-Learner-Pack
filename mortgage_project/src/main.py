"""Run the supplied baseline only; assessed advisory rules are NOT provided."""
from src.data import load_data, print_audit
from src.model import BASELINE_FEATURES, MODEL_FILE, run_baseline
from src.decisions import validate_score, advisory_decision


def report(name, result):
    print(f'\n{name}')
    for metric in ('accuracy', 'precision_1', 'recall_1', 'f1_1', 'roc_auc'):
        print(f'{metric}: {result[metric]:.3f}')
    print('Confusion matrix [actual rows 0,1; predicted columns 0,1]:')
    print(result['confusion_matrix'])


def main():
    df = load_data()
    print_audit(df)
    reference_results, baseline_results = run_baseline(df)
    print('\nTutor-selected baseline features:', BASELINE_FEATURES)
    report('Majority-class reference', reference_results)
    report('Two-feature logistic-regression baseline', baseline_results)
    print(f'\nTrained baseline saved: {MODEL_FILE}')
    print('LEARNER: justify/extend model; create decision rules, application integration, tests, and improvement evidence.')


if __name__ == '__main__':
    main()
