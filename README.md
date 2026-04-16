# Credit Default Risk Predictor

Predictive modeling system for consumer loan default risk, built on 307,511 Home Credit applications. Combines gradient boosted trees, SHAP-based explainability and a live Streamlit scoring interface designed around bank underwriting workflows.

**Live App:** [Streamlit — Credit Risk Predictor](https://credit-risk--predictor.streamlit.app/)
**Dataset:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

---

## Results

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | 0.729 | 0.342 | 0.658 | 0.449 |
| LightGBM (tuned) | **0.781** | **0.418** | **0.712** | **0.527** |

7-point AUC improvement over baseline. Projected 15–20% reduction in first-year charge-offs based on holdout backtesting at a 35% decline threshold.

---

## Project Structure

```
credit-risk-predictor/
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── app/
│   └── streamlit_app.py
├── models/                  # generated after running notebook
├── assets/                  # generated visualizations
├── data/                    # download from Kaggle (see below)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/ManishThumma/credit-risk-predictor.git
cd credit-risk-predictor
pip install -r requirements.txt
```

**Download dataset:**
```bash
kaggle competitions download -c home-credit-default-risk -p data --unzip
```
Requires Kaggle API credentials at `~/.kaggle/kaggle.json`. Accept competition rules at kaggle.com/competitions/home-credit-default-risk/rules first.

**Run notebook:**
```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

**Launch app:**
```bash
cd app && streamlit run streamlit_app.py
```

---

## Methodology

**Feature Engineering** — 11 features derived from credit risk domain knowledge:

| Feature | Definition |
|---------|-----------|
| `DEBT_TO_INCOME` | Total loan / annual income |
| `ANNUITY_TO_INCOME` | Monthly payment / monthly income |
| `CREDIT_TO_GOODS` | Loan amount / goods price |
| `EMPLOYMENT_TO_AGE` | Years employed / applicant age |
| `LOAN_TERM_MONTHS` | Implied repayment horizon |
| `INCOME_PER_PERSON` | Income normalized by household size |
| `EXT_SOURCE_MEAN` | Composite of 3 bureau credit scores |
| `EXT_SOURCE_MIN` | Weakest bureau sub-score |
| `IS_UNEMPLOYED` | Binary employment status flag |

**Modeling** — LightGBM with `scale_pos_weight` for class imbalance (8% default rate). Hyperparameters tuned via RandomizedSearchCV over 30 iterations, 5-fold stratified CV. Missing values handled with median imputation + binary missingness flags for columns with 10–40% missing; columns above 40% dropped.

**Explainability** — SHAP TreeExplainer generates exact Shapley values for every prediction. Top contributing features map directly to ECOA adverse action notice requirements.

**Risk Tiers:**

| Tier | Probability | Decision |
|------|------------|---------|
| Low | < 15% | Auto Approve |
| Medium | 15–35% | Manual Review |
| High | > 35% | Auto Decline |

---

## Streamlit App

- Default probability gauge with three-tier decisioning
- SHAP waterfall chart per applicant
- ECOA-compliant adverse action reasons
- Model performance dashboard with precision-recall threshold curve

---

## Stack

Python · LightGBM · scikit-learn · SHAP · Streamlit · Plotly · pandas · numpy

---

**Author:** Manish Reddy Thumma
