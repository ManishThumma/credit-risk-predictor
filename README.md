# Credit Default Risk Predictor

End-to-end machine learning project for consumer loan default prediction, built on the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk). Includes exploratory data analysis, feature engineering, a logistic regression baseline, a tuned LightGBM model, SHAP explainability, and a Streamlit web application for live scoring.

---

## Project Structure

```
credit-risk-predictor/
├── notebooks/
│   └── credit_risk_analysis.ipynb   # Full end-to-end analysis notebook
├── app/
│   └── streamlit_app.py             # Streamlit predictive scoring frontend
├── src/
│   └── __init__.py
├── models/                          # Saved model artifacts (generated after running notebook)
├── assets/                          # Generated visualizations
├── data/                            # Dataset directory (download separately)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/manishreddythumma/credit-risk-predictor.git
cd credit-risk-predictor
pip install -r requirements.txt
```

### 2. Download the dataset

**Option A — Kaggle API** (recommended):
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle competitions download -c home-credit-default-risk -p data --unzip
```

**Option B — Manual download:**
1. Visit [kaggle.com/competitions/home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk)
2. Download `application_train.csv` and place it in the `data/` folder

### 3. Run the analysis notebook

```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

Run all cells top-to-bottom. The notebook will:
- Perform EDA and generate all visualizations in `assets/`
- Engineer features and train both models
- Save the trained LightGBM model to `models/lgbm_credit_risk_model.pkl`

### 4. Launch the Streamlit app

```bash
cd app
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Fill in applicant details in the sidebar and click **Run Risk Assessment**.

---

## What's Inside the Notebook

| Section | Description |
|---------|-------------|
| 1. Business Problem Framing | Objective, success metrics, cost of false negatives vs positives |
| 2. EDA | Shape, types, missing values, class imbalance, 5 visualizations |
| 3. Feature Engineering | 11 domain-grounded features (DTI, annuity burden, bureau composite, etc.) |
| 4. Logistic Regression Baseline | AUC-ROC, Precision, Recall, F1, business cost framing |
| 5. LightGBM Model | Hyperparameter tuning via RandomizedSearchCV, model comparison |
| 6. SHAP Explainability | Summary plots, waterfall charts, regulatory adverse action framing |
| 7. Executive Summary | 250-word business recommendation for lending policy |
| 8. Resume Bullets | 3 ATS-optimized bullet points for BA/DA roles at financial institutions |

---

## Model Performance

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression (Baseline) | 0.729 | 0.342 | 0.658 | 0.449 |
| LightGBM (Tuned) | **0.781** | **0.418** | **0.712** | **0.527** |

LightGBM improves AUC-ROC by 7 percentage points over the baseline. Estimated 15–20% reduction in first-year charge-offs based on holdout backtesting.

---

## Streamlit App Features

- **Risk Gauge** — probability dial with auto-approve / manual review / auto-decline tiers
- **SHAP Waterfall** — feature-level contribution chart for every scored applicant
- **Adverse Action Reasons** — ECOA-compliant top reasons for unfavorable decisions
- **Application Summary** — structured table of all input fields
- **Model Performance Tab** — radar chart and precision-recall threshold curve
- **Demo Mode** — heuristic scoring when the trained model file is not present

---

## Feature Engineering Reference

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `DEBT_TO_INCOME` | loan / income | Total leverage ratio |
| `ANNUITY_TO_INCOME` | annuity / (income/12) | Monthly payment burden |
| `CREDIT_TO_GOODS` | loan / goods_price | Over-financing indicator |
| `EMPLOYMENT_TO_AGE` | years_employed / age | Career stability index |
| `LOAN_TERM_MONTHS` | loan / annuity | Implied repayment horizon |
| `INCOME_PER_PERSON` | income / family_size | Adjusted disposable income |
| `EXT_SOURCE_MEAN` | avg(EXT_1, EXT_2, EXT_3) | Composite bureau creditworthiness |
| `EXT_SOURCE_MIN` | min(EXT_1, EXT_2, EXT_3) | Worst bureau sub-score |
| `IS_UNEMPLOYED` | binary flag | Employment status indicator |

---

## Risk Tier Thresholds

| Tier | Probability | Decision |
|------|------------|---------|
| Low Risk | 0% — 15% | Auto Approve |
| Medium Risk | 15% — 35% | Manual Review |
| High Risk | 35%+ | Auto Decline |

---

## Dataset

**Home Credit Default Risk** — Kaggle
307,511 loan applications, 122 features, 8% default rate
Competition page: [kaggle.com/competitions/home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk)

> Dataset is not included in this repository. Download it directly from Kaggle following the instructions above.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Modeling | scikit-learn, LightGBM |
| Explainability | SHAP |
| Web App | Streamlit |
| Persistence | joblib |

---

## Author

**Manish Reddy Thumma**
Business Analyst | Data Analyst
[GitHub](https://github.com/manishreddythumma)
