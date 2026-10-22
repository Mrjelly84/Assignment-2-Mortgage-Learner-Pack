# Mortgage decision-support starter · DN6MD001 Assignment 2

This is a **working, deliberately limited tutor-provided ML baseline**; it is **not a finished Assignment 2 solution**. Read `../START_HERE.md`, `../CODE_WALKTHROUGH.md`, `../DATA_DICTIONARY.md` and the approved brief in the parent folder.

## Quick start (Windows PowerShell from this folder)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.main
.\.venv\Scripts\python.exe -m src.applicant_example
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The supplied dataset is already in `data/mortgage_data.csv`. The program uses only **synthetic data**. Do not use real applicant details.

### What is working?
- Data audit (`src/data.py`)
- A basic two-feature Logistic Regression pipeline with training-only imputation/scaling, a majority-class reference and initial held-out evaluation (`src/model.py`, `src/main.py`)
- Saved model and sample applicant **scoring** (not decision-making) (`src/applicant_example.py`)
- Four generic validation tests (`tests/test_validation.py`)

### What is your independent assignment work?

Develop, justify and test your own logical decision rules and their integration with the model; make evidence-informed modelling/evaluation decisions, carry out meaningful improvement and retesting, document decisions and sources, and maintain Git/GitHub version history. The intentionally incomplete `advisory_decision()` in `src/decisions.py` is not an error in the supplied training/scoring example.

**Version-control and data note:** Synthetic CSV data and generated `.joblib` model files are excluded from Git by default. The tutor has provided the dataset in the assignment pack; explain how to obtain it in your README. Seek permission before putting data into a public repository. Never commit real personal or confidential data or credentials. A reviewer must be able to run the project using the supplied dataset and the documented commands.

**Academic integrity:** Acknowledge module-provided code and any permitted AI assistance under the approved brief. Keep the final report and project evidence consistent.
