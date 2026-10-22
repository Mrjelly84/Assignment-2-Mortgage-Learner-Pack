"""Run the supplied baseline only; assessed advisory rules are NOT provided."""
from src.data import load_data, print_audit
from src.model import IMPROVED_FEATURES, MODEL_FILE, run_improved_model
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
    reference_results, improved_results = run_improved_model(df)
    print('\nLearner-selected improved-model features:', IMPROVED_FEATURES)
    report('Majority-class reference', reference_results)
    report('Improved Random Forest model', improved_results)
    print(f'\nTrained improved model saved: {MODEL_FILE}')
   

if __name__ == '__main__':
    main()
