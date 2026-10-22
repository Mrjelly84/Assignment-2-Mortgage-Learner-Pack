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

`python -m src.main` opens the graphical assessment form, prefilled with a random complete example from the supplied synthetic dataset. Select **Randomize Sample** to load another example, or edit the fields before running an assessment. If no saved model exists yet, the app trains one from the supplied dataset first.

### What is working?
- Data audit (`src/data.py`)
- A basic two-feature Logistic Regression pipeline with training-only imputation/scaling, a majority-class reference and initial held-out evaluation (`src/model.py`, `src/main.py`)
- Saved model, sample applicant scoring, and advisory rule outcomes (`src/applicant_example.py`, `src/decisions.py`)
- Validation and advisory decision tests (`tests/test_validation.py`)

### What is your independent assignment work?

The prototype routes repayment-difficulty scores below 0.40 to lower-risk human review, scores from 0.40 through 0.749 to medium-risk review, and scores of 0.75 or higher to deep human review. These are provisional advisory thresholds, not validated mortgage policy or an automated approval/decline. Justify and test any changes to these rules, and document modelling decisions, sources, and Git/GitHub version history.

**Version-control and data note:** Synthetic CSV data and generated `.joblib` model files are excluded from Git by default. The tutor has provided the dataset in the assignment pack; explain how to obtain it in your README. Seek permission before putting data into a public repository. Never commit real personal or confidential data or credentials. A reviewer must be able to run the project using the supplied dataset and the documented commands.

**Academic integrity:** Acknowledge module-provided code and any permitted AI assistance under the approved brief. Keep the final report and project evidence consistent.
