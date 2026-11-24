# Predicting Customer Churn in Subscription-Based Streaming Services

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuralee1/churn-prediction-project/blob/main/notebooks/Churn_Project_Presentation.ipynb)

**Student:** Yura | **Institution:** Texas A&M International University (TAMIU) | **Term project**

A complete analytics project that predicts which subscribers are likely to cancel, explains why, and turns the findings into retention recommendations. It uses the public IBM Telco Customer Churn dataset (7,043 customers) as a proxy for a subscription streaming service.

**Start here:** open [`notebooks/Churn_Project_Presentation.ipynb`](notebooks/Churn_Project_Presentation.ipynb). It walks through the business problem, objectives, data, preparation, exploration, methodology, a live code demonstration, findings, business value, and recommendations. Outputs are stored in the notebook, so it can be read on GitHub without running anything.

## Business problem
Subscription services usually react to churn only when a customer calls to cancel, which is too late. This project asks whether at-risk subscribers can be identified *before* they cancel, which attributes drive churn, and how many customers a retention team should contact.

## Approach
- **Data preparation:** quality checks, blank `TotalCharges` fixed (11 new customers, set to $0), consistent categories, no missing values.
- **Exploration:** descriptive statistics, churn rates by attribute, chi-square tests and Cramér's V.
- **Models:** Logistic Regression, Decision Tree, Random Forest. Each is a scikit-learn pipeline (scaling and encoding fit on training data only), tuned with 5-fold cross-validation, class-weighted for the 26.5% churn rate, and evaluated once on a held-out 20% test set.
- **Operating points:** default 0.50 threshold and a recall-favoring threshold chosen from training data (F2 score).
- **Quality controls:** built-in sanity checks (including a data-leakage check) and 13 automated tests.

## Key results
| Model | Test ROC-AUC | Recall at 0.50 | Precision at 0.50 |
|---|---|---|---|
| Logistic Regression | 0.839 | 0.778 | 0.509 |
| Decision Tree | 0.839 | 0.794 | 0.520 |
| Random Forest | 0.841 | 0.789 | 0.525 |

*Reference results from `outputs/reference/` (scikit-learn 1.8.0).*

- Month-to-month customers are 55% of the base but **89% of churners** (42.7% churn vs 2.8% on two-year contracts).
- **55% of churners leave in their first 12 months**; month-to-month fiber-optic customers churn at 54.6% and account for 62% of all churners.
- Contacting the **top 20% highest-risk customers reaches 50% of churners** (2.5x better than random targeting).
- The best contact depth depends on cost per contact versus the value of a saved customer (illustrative analysis; assumptions are editable in the notebook).

## Run it
**Google Colab:** click the badge above, then *Runtime > Run all* (about one minute). The first code cell clones the repository automatically.

**Locally (Python 3.10+):**
```bash
pip install -r requirements.txt
python run_churn_model.py                       # trains and evaluates; writes to outputs/latest_run/
python -m unittest discover -s tests -v         # 13 tests (about 40 seconds)
jupyter notebook notebooks/Churn_Project_Presentation.ipynb
```

## Repository structure
```
README.md                         this file
notebooks/
  Churn_Project_Presentation.ipynb   the project presentation (start here)
churn_pipeline/                   documented Python package
  config.py                         constants, feature lists, hyperparameter grids
  cleaning.py                       raw-data checks and cleaning
  data.py                           loading and stratified split
  modeling.py                       preprocessing, pipelines, tuning, threshold selection
  evaluation.py                     metrics, results table, feature tables
  plotting.py                       figures
  checks.py                         sanity and leakage checks
  pipeline.py                       end-to-end run
run_churn_model.py                command-line entry point
tests/                            unit, cleaning, and regression tests
data/
  raw/                              original IBM file
  processed/                        telco_clean.csv and telco_model_ready.csv
outputs/reference/                results and figures used in the final report
figures/                          figures used in the report
reports/                          proposal, milestone reports, and the final technical report
history/                          original prep script and earlier code versions (AI draft, pre-refactor)
requirements.txt                  minimum library versions
requirements-exact.txt            versions used for the reference results
```

## Reports
All written deliverables are in [`reports/`](reports): project proposal, data preparation, data report, AI prompt collection, AI-assisted code development, code refactoring summary, and the **final technical report**.

## Reproducibility notes
- Fixed random seed (42); repeated runs give identical results in the same environment.
- Reference results were produced with scikit-learn 1.8.0 and pandas 3.0.2. Logistic Regression and Decision Tree results are identical across versions tested; Random Forest can shift slightly (for example, recall at 0.50 of 0.789 vs 0.799 in scikit-learn 1.9.1). The regression test accounts for this with a tolerance.
- `outputs/latest_run/` is created when you run the code and is not stored in the repository.

## Data
IBM Telco Customer Churn, a public sample dataset published by IBM (https://github.com/IBM/telco-customer-churn-on-icp4d), downloaded October 5, 2026. It contains no real personal information and is used here for academic purposes. It describes a telecom company, so it is a structural proxy for streaming subscriptions, not streaming company data.

## Limitations
Proxy data; snapshot with no dates; findings are associations, not causes; business-value figures use illustrative assumptions; one train/test split of one dataset.

## Use of generative AI
Claude (Anthropic) assisted with prompts, code, tests, and documentation. AI-generated code was reviewed, run, and tested; the revisions made to the first AI draft are documented in `reports/AI_Code_Milestone_Report.docx`, and the original draft is kept in `history/`.
