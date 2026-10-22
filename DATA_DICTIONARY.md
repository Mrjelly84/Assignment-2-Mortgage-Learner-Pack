# Mortgage dataset: data dictionary

This table describes the **actual supplied synthetic CSV**. Definitions are interpretations of field names and recorded values; a formal lender policy or a precise definition of the event labelled 'repayment difficulty' has **not** been supplied. 

- **Records:** 800 synthetic mortgage applications, one row per `application_id` (no duplicate IDs in this file).
- **Columns:** 19.
- **Outcome:** `repayment_difficulty_12m`, containing 625 zeros and 175 ones.
- **Missing values:** `credit_score` (21), `employment_years` (31), `savings_gbp` (24), `credit_utilisation_pct` (19); other columns have none.

| Field | Meaning / interpretation | Example from dataset | Missing | Important notes |
|---|---|---:|---:|---|
| `application_id` | Identifier for an application | `MORT0001` | 0 | Do **not** use identifier as predictive evidence. |
| `annual_income_gbp` | Annual income stated in pounds | `66774` | 0 | Income measure; investigate availability and interpretation. |
| `property_value_gbp` | Property valuation in pounds | `258418` | 0 | Housing/property information. |
| `deposit_gbp` | Mortgage deposit in pounds | `21191` | 0 | May be related to loan amount and LTV. |
| `requested_loan_gbp` | Requested mortgage borrowing in pounds | `237227` | 0 | Amount of borrowing requested. |
| `loan_to_value_pct` | Loan-to-value percentage | `91.80` | 0 | Percentage, **not** decimal ratio. Investigate relationships with loan/property. |
| `monthly_debt_commitments_gbp` | Recorded monthly debt commitments in pounds | `95` | 0 | Do not assume how other ratios are calculated without checking. |
| `debt_to_income_ratio` | Supplied debt-to-income ratio | `0.212` | 0 | Decimal ratio: `0.32` means 32%, **not** 0.32%. |
| `credit_score` | Recorded credit-score value | `629` | **21** | Numeric field used in the tutor baseline. |
| `employment_type` | Employment category | `Employed` | 0 | Values: Employed, Contractor, Self-employed; encoding needs consideration. |
| `employment_years` | Years of employment recorded | `1.1` | **31** | Numeric years; some values missing. |
| `savings_gbp` | Recorded savings in pounds | `12147` | **24** | Some values missing. |
| `missed_payments_24m` | Count of missed payments in a 24-month period | `0` | 0 | Confirm timing/meaning before treating as a predictor. |
| `credit_utilisation_pct` | Credit-utilisation percentage | `52.41` | **19** | Numeric percentage, some missing. |
| `previous_default_flag` | Historical default indicator | `1` | 0 | Binary flag; verify its time reference relative to application. |
| `term_years` | Mortgage term in years | `20` | 0 | Numeric, whole years. |
| `interest_rate_pct` | Recorded interest rate as a percentage | `6.06` | 0 | Consider whether it was known at application time: potential timing/leakage risk. |
| `application_channel` | Recorded application channel | `Broker` | 0 | Values: Broker, Online, Branch; categorical. |
| `repayment_difficulty_12m` | Observed repayment difficulty indicator within 12 months | `1` | 0 | **Target outcome:** 0=no recorded difficulty, 1=recorded difficulty. Not necessarily default; do not label its score a PD. |

## Before you train a model, ask

- Which information could genuinely be known when a new mortgage application is assessed? Is anything recorded **after** the decision?
- Does an identifier or outcome leak the answer into the inputs?
- Are values missing, unusual, duplicated or inconsistent?
- Are highly related variables providing useful information, or simply repeating each other?
- What could false negatives and false positives mean for a real human reviewer?
- Could variables be unfair proxies, and how would you document and investigate this limitation?

