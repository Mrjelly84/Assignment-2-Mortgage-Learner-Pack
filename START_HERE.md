# DN6MD001 · Assignment 2 mortgage project · Start here

**Level 6 · Artificial Intelligence and Machine Learning **

This pack is the **learner starting point** for Assignment 2, a Python mortgage decision-support prototype. This guide explains the supplied materials and the route into independent assessed work.

## The four things you have been given

1. **Actual synthetic mortgage dataset** (`mortgage_project/data/mortgage_data.csv`) — 800 observations and 19 fields, including the observed `repayment_difficulty_12m` outcome. It is supplied in the correct folder and with the correct filename.
3. **Plain-English data dictionary** (`DATA_DICTIONARY.md`) — field descriptions, examples and important interpretation cautions.
4. **Working Python mortgage starter** (`mortgage_project/`) — a limited model that can load data, audit it, train Logistic Regression, display baseline results, save a trained model and score one sample applicant. See `CODE_WALKTHROUGH.md`.

Optional planning support: `DEVELOPMENT_AND_FORMATIVE_GUIDE.md` gives report-section prompts, evidence checklists and blank design/test tables. 

## Run the starter in VS Code on Windows

Open the **mortgage_project** folder in VS Code (not the outer folder), then choose **Terminal → New Terminal**. Keep the terminal in this project folder.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.main
.\.venv\Scripts\python.exe -m src.applicant_example
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If `py` is unavailable, use `python` to create the environment. On Mac/Linux use `python3 -m venv .venv` and `.venv/bin/python` in subsequent commands. Python 3.10+ is recommended.

The outputs are intentionally limited: 
`src.main` **trains and evaluates** a baseline
`src.applicant_example` **loads the saved model and scores** one example applicant. 
They are separate operations. The example **does not issue a mortgage recommendation** because that is part of your assessed work.

The brief requires you to design and document a logical decision-support system, justify data preparation and modelling choices, integrate model output with advisory rules, test **both** the model and decision behaviour, investigate and demonstrate a meaningful evidence-informed improvement, and evaluate remaining limitations.

You may retain or discard and justify a diffrent technique than Logistic Regression if you so wish. A new algorithm or graphical user interface is **not required** by the brief. A console-based Python prototype is sufficient if it meets the functionality, documentation, testing and evaluation requirements although if you want o build a gui that is your choice. 

The original 800-row dataset is included locally. The project's `.gitignore` excludes CSV data and generated model files by default. Keep them available through the supplied pack or an approved private handover route, and explain data access in your repository README. 
