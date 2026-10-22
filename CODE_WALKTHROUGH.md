# Understanding your supplied mortgage starter

This is a guide. Your assignment requires you to understand it, make your own decisions and develop a working decision-support system from it.

## What each file does

| File | In plain English | Already working? |
|---|---|---|
| `src/data.py` | Opens the CSV and reports columns, missing values, classes and duplicates. | Yes |
| `src/model.py` | Builds the basic training pipeline: fill missing numeric values, scale inputs, fit Logistic Regression, report metrics and save the result. | Yes |
| `src/main.py` | Runs the audit and trains/evaluates the baseline model. | Yes |
| `src/applicant_example.py` | Loads the saved model, uses two sample input values and prints an estimated repayment-difficulty class-1 score. | Yes (scoring only) |
| `src/decisions.py` | Contains basic validation examples and a deliberately unfinished `advisory_decision()` function. | Not a completed mortgage rule system |
| `tests/test_validation.py` | Demonstrates four basic tests of generic functions. | Yes (generic tests only) |
| `README.md` / `requirements.txt` | Project setup, installation and information needed to reproduce the basic run. | Yes (you will update documentation for your final version) |

## The two model inputs used for teaching

The tutor chose **`credit_score`** and **`debt_to_income_ratio`** for a simple, accessible baseline. The observed outcome being predicted is **`repayment_difficulty_12m`**, not mortgage approval, and not necessarily default. There are many other columns available: you must justify your own feature choices and modelling decisions.

## What happens when you run the project?

1. `python -m src.data`: reads the data and prints useful facts. **You explain what they mean.**
2. `python -m src.main`: uses 80% of observations to fit the pipeline and reserves 20% for an initial test; it also compares Logistic Regression to a basic model that predicts the majority class. It saves a local file in `models/`.
3. `python -m src.applicant_example`: takes the **already trained** model, scores the example with credit score `680` and debt-to-income `0.32`, and prints an estimated score of approximately `0.204` with the supplied dataset and settings. It **does not** produce an approved/rejected mortgage result.
4. `python -m unittest discover -s tests -v`: runs four tests of the generic validation functions. They do **not** test your future mortgage recommendation rules.

## What the initial figures tell us

| Metric | Tutor baseline | Everyday interpretation |
|---|---:|---|
| Accuracy | 75.0% | 75 in 100 held-out classifications were correct overall. |
| Recall (difficulty) | 17.1% | It found **6 of 35** actual repayment-difficulty cases in the held-out data. |
| Precision (difficulty) | 35.3% | **6 of 17** cases it flagged had recorded repayment difficulty. |
| ROC-AUC | 0.691 | The model has some ability to rank difficult cases above non-difficult cases; this is not evidence of a valid lending system. |

Confusion matrix `[[114, 11], [29, 6]]`: 114 correct non-difficulty classifications, 11 false alerts, 29 missed difficulty cases, 6 correctly flagged difficulty cases. A majority-class-only reference gives **78.1% accuracy**, so the 75.0% headline accuracy alone is not impressive. A model score from `predict_proba()` is an **estimated class-1 probability, not a confirmed calibrated PD**.

## What you develop next (not supplied)

- Decide and **justify** what information and preprocessing your model should use; make and test any changes you choose.
- Return to your **Week 6 system diagram and decision table**, and implement mortgage-specific checks and decision rules in your own code.
- Connect a new applicant's input through the model to an **interpretable provisional advisory for a human reviewer**. A console application is enough; the brief does **not** require a GUI. You may use 'recommend review/accept/decline' language if you explain its meaning, limits and reviewer role. The software must not actually approve or decline mortgages automatically.
- Test expected versus actual rule behaviour, especially boundary values, missing information and conflicting signals.
- Identify **at least one meaningful limitation**, make an evidence-informed change, **retest and compare** with your original version.
- Document your changes in Git, explain how someone can reproduce the program and acknowledge tutor-provided code.

**Evaluation safeguard:** The given baseline reports an initial 20% held-out test result for teaching. During independent development, use training-only validation or cross-validation to compare candidate features, models or decision thresholds. Avoid repeatedly changing the approach based on that final held-out test set. Threshold changes can improve recall at the cost of more false alerts **without changing the trained model**.

