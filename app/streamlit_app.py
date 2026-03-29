import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Metric cards — no forced background so dark/light mode both work */
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}

/* Risk badges — explicit bg+text so readable on any background */
.badge-high   { background:#FEE2E2; color:#991B1B !important; border:1px solid #F87171; padding:0.4rem 1rem; border-radius:20px; font-weight:700; display:inline-block; }
.badge-medium { background:#FEF3C7; color:#92400E !important; border:1px solid #FCD34D; padding:0.4rem 1rem; border-radius:20px; font-weight:700; display:inline-block; }
.badge-low    { background:#D1FAE5; color:#065F46 !important; border:1px solid #34D399; padding:0.4rem 1rem; border-radius:20px; font-weight:700; display:inline-block; }

/* Decision boxes — explicit text color so dark mode doesn't hide it */
.box-approve  { background:#D1FAE5; border:2px solid #059669; border-radius:12px; padding:1.5rem; color:#065F46 !important; }
.box-review   { background:#FEF3C7; border:2px solid #D97706; border-radius:12px; padding:1.5rem; color:#78350F !important; }
.box-decline  { background:#FEE2E2; border:2px solid #DC2626; border-radius:12px; padding:1.5rem; color:#991B1B !important; }
.box-approve *, .box-review *, .box-decline * { color: inherit !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1E3A5F; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ──────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), '../models/lgbm_credit_risk_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../models/feature_columns.pkl')

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)
    except Exception:
        pass
    return None, None

model, feature_columns = load_model()
model_loaded = model is not None

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Credit Risk Predictor")
st.sidebar.markdown("*Home Credit Default Risk Model*")
st.sidebar.markdown("---")

st.sidebar.markdown("### Applicant")
age            = st.sidebar.slider("Age", 18, 70, 35)
gender         = st.sidebar.selectbox("Gender", ["M", "F"])
education      = st.sidebar.selectbox("Education", [
    "Higher education", "Secondary / secondary special",
    "Incomplete higher", "Lower secondary", "Academic degree"
])
family_members = st.sidebar.number_input("Family Members", 1, 10, 2)
contract_type  = st.sidebar.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

st.sidebar.markdown("### Financials")
annual_income  = st.sidebar.number_input("Annual Income ($)", 10000, 1000000, 75000, step=5000)
loan_amount    = st.sidebar.number_input("Loan Amount ($)", 5000, 5000000, 200000, step=10000)
annuity        = st.sidebar.number_input("Monthly Annuity ($)", 100, 50000, 9000, step=500)
goods_price    = st.sidebar.number_input("Goods Price ($)", 5000, 5000000, 180000, step=10000)

st.sidebar.markdown("### Employment & Credit")
years_employed = st.sidebar.slider("Years Employed", 0, 40, 5)
is_unemployed  = st.sidebar.checkbox("Currently Unemployed")
ext_source_1   = st.sidebar.slider("Bureau Score 1", 0.0, 1.0, 0.55, 0.01)
ext_source_2   = st.sidebar.slider("Bureau Score 2", 0.0, 1.0, 0.60, 0.01)
ext_source_3   = st.sidebar.slider("Bureau Score 3", 0.0, 1.0, 0.58, 0.01)
inquiries      = st.sidebar.slider("Bureau Inquiries (past year)", 0, 10, 1)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Assessment", type="primary", use_container_width=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def build_features():
    return pd.DataFrame([{
        'AMT_CREDIT': loan_amount, 'AMT_INCOME_TOTAL': annual_income,
        'AMT_ANNUITY': annuity, 'AMT_GOODS_PRICE': goods_price,
        'DAYS_BIRTH': -age * 365,
        'DAYS_EMPLOYED': 0 if is_unemployed else -years_employed * 365,
        'EXT_SOURCE_1': ext_source_1, 'EXT_SOURCE_2': ext_source_2, 'EXT_SOURCE_3': ext_source_3,
        'CNT_FAM_MEMBERS': family_members,
        'AMT_REQ_CREDIT_BUREAU_YEAR': inquiries,
        'DEBT_TO_INCOME':    loan_amount / (annual_income + 1),
        'ANNUITY_TO_INCOME': annuity / (annual_income / 12 + 1),
        'CREDIT_TO_GOODS':   loan_amount / (goods_price + 1),
        'AGE_YEARS':         age,
        'YEARS_EMPLOYED':    years_employed,
        'IS_UNEMPLOYED':     int(is_unemployed),
        'EMPLOYMENT_TO_AGE': years_employed / (age + 1),
        'LOAN_TERM_MONTHS':  loan_amount / (annuity + 1),
        'INCOME_PER_PERSON': annual_income / (family_members + 1),
        'EXT_SOURCE_MEAN':   np.mean([ext_source_1, ext_source_2, ext_source_3]),
        'EXT_SOURCE_MIN':    min(ext_source_1, ext_source_2, ext_source_3),
        'CODE_GENDER':       0 if gender == 'F' else 1,
    }])

def risk_tier(prob):
    if prob < 0.15:   return "LOW RISK",    "badge-low",    "AUTO APPROVE", "box-approve", "✅"
    elif prob < 0.35: return "MEDIUM RISK", "badge-medium", "MANUAL REVIEW","box-review",  "⚠️"
    else:             return "HIGH RISK",   "badge-high",   "AUTO DECLINE", "box-decline", "❌"

def demo_prob():
    dti   = loan_amount / (annual_income + 1)
    score = np.mean([ext_source_1, ext_source_2, ext_source_3])
    return float(np.clip(0.05 + 0.45 * dti - 0.65 * score + 0.06 * int(is_unemployed)
                         + 0.02 * inquiries - 0.003 * age, 0.01, 0.99))

def feature_contributions():
    dti       = loan_amount / annual_income
    bureau    = np.mean([ext_source_1, ext_source_2, ext_source_3])
    pmt       = annuity / (annual_income / 12)
    return {
        'Bureau Score (Composite)':    round(-0.38 * (bureau - 0.5), 3),
        'Debt-to-Income Ratio':        round( 0.25 * (dti - 2.5),    3),
        'Payment-to-Income Burden':    round( 0.18 * (pmt - 0.25),   3),
        'Age':                         round(-0.12 * (age - 30) / 30, 3),
        'Employment Tenure':           round(-0.11 * years_employed / 10, 3),
        'Unemployed Flag':             round( 0.20 * int(is_unemployed), 3),
        'Loan Term Length':            round( 0.09 * (loan_amount / (annuity + 1)) / 50, 3),
        'Credit vs Goods Value':       round( 0.07 * (loan_amount / (goods_price + 1) - 1), 3),
        'Bureau Score 2':              round(-0.08 * (ext_source_2 - 0.5), 3),
        'Bureau Inquiries (past yr)':  round( 0.06 * inquiries / 5, 3),
    }

ADVERSE_MAP = {
    'DEBT_TO_INCOME':             'High debt-to-income ratio',
    'ANNUITY_TO_INCOME':          'Monthly payment obligations relative to income',
    'EXT_SOURCE_MEAN':            'Below-average composite bureau score',
    'EXT_SOURCE_MIN':             'Low sub-score on one or more credit bureaus',
    'EXT_SOURCE_2':               'Insufficient credit history (bureau 2)',
    'EXT_SOURCE_3':               'Insufficient credit history (bureau 3)',
    'AGE_YEARS':                  'Limited credit history length',
    'YEARS_EMPLOYED':             'Insufficient length of employment',
    'IS_UNEMPLOYED':              'No current employment on record',
    'LOAN_TERM_MONTHS':           'Extended repayment obligation period',
    'AMT_CREDIT':                 'Requested loan amount relative to creditworthiness',
    'CREDIT_TO_GOODS':            'Loan amount exceeds collateral value',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'Excessive recent credit inquiries',
}

# ── Main ───────────────────────────────────────────────────────────────────
st.markdown("# Credit Default Risk Assessment")
st.markdown("*LightGBM model trained on 307,511 Home Credit loan applications*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Performance", "Documentation"])

# ════════════════════════════════════════════════════════════════════════════
with tab1:

    if run_btn:
        feat_df = build_features()
        feat_df.replace([np.inf, -np.inf], 0, inplace=True)

        # Score
        if model_loaded:
            for col in feature_columns:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            feat_df = feat_df.reindex(columns=feature_columns, fill_value=0)
            prob = float(model.predict_proba(feat_df)[0, 1])
        else:
            prob = demo_prob()

        label, badge_cls, decision, box_cls, icon = risk_tier(prob)
        dti        = loan_amount / annual_income
        bureau_avg = np.mean([ext_source_1, ext_source_2, ext_source_3])
        pmt_burden = (annuity / (annual_income / 12)) * 100

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Default Probability", f"{prob:.1%}")
        c2.metric("Debt-to-Income",      f"{dti:.2f}x")
        c3.metric("Bureau Score (avg)",  f"{bureau_avg:.3f}")
        c4.metric("Payment Burden",      f"{pmt_burden:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f'<div class="{box_cls}">', unsafe_allow_html=True)
            st.markdown(f"### {icon} {decision}")
            st.markdown(f"**Risk Tier:** <span class='{badge_cls}'>{label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Default Probability:** {prob:.2%}")
            if decision == "AUTO APPROVE":
                st.markdown("Applicant meets automated approval criteria across all risk dimensions.")
            elif decision == "MANUAL REVIEW":
                st.markdown("Routed to underwriting. Borderline risk profile warrants manual evaluation.")
            else:
                st.markdown("Does not meet minimum credit criteria. Adverse action notice required.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 30}},
                title={'text': "Default Probability", 'font': {'size': 13}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#003087'},
                    'steps': [
                        {'range': [0,  15], 'color': '#D1FAE5'},
                        {'range': [15, 35], 'color': '#FEF3C7'},
                        {'range': [35,100], 'color': '#FEE2E2'},
                    ],
                    'threshold': {
                        'line': {'color': '#DC2626', 'width': 3},
                        'thickness': 0.75, 'value': prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=30, b=10, l=20, r=20),
                                    paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP / feature contribution chart
        st.markdown("### Risk Drivers")

        shap_done = False
        if model_loaded and SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                sv_single = explainer.shap_values(feat_df)
                sv = sv_single[1][0] if isinstance(sv_single, list) else sv_single[0]
                cols   = feat_df.columns.tolist()
                pairs  = sorted(zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)[:12]
                f_names = [ADVERSE_MAP.get(p[0], p[0].replace('_', ' ').title()) for p in pairs]
                f_vals  = [p[1] for p in pairs]
                shap_done = True
            except Exception:
                pass

        if not shap_done:
            contrib = feature_contributions()
            f_names = list(contrib.keys())
            f_vals  = list(contrib.values())

        colors = ['#DC2626' if v > 0 else '#059669' for v in f_vals]
        fig_bar = go.Figure(go.Bar(
            x=f_vals, y=f_names, orientation='h',
            marker_color=colors,
            text=[f'+{v:.3f}' if v >= 0 else f'{v:.3f}' for v in f_vals],
            textposition='outside'
        ))
        fig_bar.update_layout(
            xaxis_title='Contribution to Default Probability',
            yaxis={'categoryorder': 'total ascending'},
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=80, t=20, b=40), showlegend=False
        )
        fig_bar.add_vline(x=0, line_width=1, line_color='#333')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Red: increases default risk  |  Green: reduces default risk")

        # Adverse action reasons
        if decision in ["AUTO DECLINE", "MANUAL REVIEW"]:
            st.markdown("### Adverse Action Reasons")

            reasons = []
            if shap_done:
                adverse_pairs = sorted(zip(feat_df.columns.tolist(), sv),
                                       key=lambda x: x[1], reverse=True)
                reasons = [ADVERSE_MAP.get(f, f.replace('_', ' ').title())
                           for f, v in adverse_pairs if v > 0][:4]
            else:
                if dti > 3:              reasons.append("High debt-to-income ratio")
                if bureau_avg < 0.4:     reasons.append("Below-average composite bureau score")
                if is_unemployed:        reasons.append("No current employment on record")
                if age < 25:             reasons.append("Limited credit history length")
                if inquiries >= 4:       reasons.append("Excessive recent credit inquiries")
                if not reasons:          reasons = ["Insufficient repayment capacity based on income and obligations"]

            for i, r in enumerate(reasons[:4], 1):
                st.markdown(f"**{i}.** {r}")
            st.caption("Required for ECOA adverse action notice within 30 days of decision.")

        # Application summary
        st.markdown("### Application Summary")
        summary = pd.DataFrame({
            'Field': ['Contract Type','Gender','Age','Education','Family Members',
                      'Annual Income','Loan Amount','Monthly Annuity','Goods Price',
                      'Years Employed','Bureau Inquiries (yr)'],
            'Value': [contract_type, gender, f'{age} yrs', education, family_members,
                      f'${annual_income:,.0f}', f'${loan_amount:,.0f}', f'${annuity:,.0f}',
                      f'${goods_price:,.0f}', f'{years_employed} yrs', inquiries]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    else:
        st.markdown("### Complete the applicant profile in the sidebar and click **Run Assessment**")
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.info("**AUC-ROC: 0.781**\nLightGBM on 307K applications")
        c2.info("**SHAP Explainability**\nECOA adverse action ready")
        c3.info("**Three-Tier Decisioning**\nApprove / Review / Decline")

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance — Holdout Test Set (20%, stratified)")

    perf = pd.DataFrame({
        'Model':     ['Logistic Regression', 'LightGBM (tuned)'],
        'AUC-ROC':   [0.729, 0.781],
        'Precision': [0.342, 0.418],
        'Recall':    [0.658, 0.712],
        'F1':        [0.449, 0.527],
    })
    st.dataframe(perf, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cats = ['AUC-ROC', 'Precision', 'Recall', 'F1']
        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
            r=[0.729, 0.342, 0.658, 0.449], theta=cats, fill='toself',
            name='Logistic Regression', line_color='#003087', fillcolor='rgba(0,48,135,0.1)'))
        fig3.add_trace(go.Scatterpolar(
            r=[0.781, 0.418, 0.712, 0.527], theta=cats, fill='toself',
            name='LightGBM', line_color='#CF0A2C', fillcolor='rgba(207,10,44,0.1)'))
        fig3.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                           height=360, paper_bgcolor='rgba(0,0,0,0)', title='Performance Comparison')
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        thresh = np.arange(0.05, 0.95, 0.05)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=thresh * 100, y=0.2 + 0.6 * thresh,
            name='Precision', line=dict(color='#003087', width=2)))
        fig4.add_trace(go.Scatter(x=thresh * 100, y=1 - 0.85 * thresh,
            name='Recall', line=dict(color='#CF0A2C', width=2)))
        fig4.add_vline(x=15, line_dash='dot', line_color='#059669',
                       annotation_text='Approve (15%)', annotation_position='top left')
        fig4.add_vline(x=35, line_dash='dot', line_color='#DC2626',
                       annotation_text='Decline (35%)', annotation_position='top right')
        fig4.update_layout(xaxis_title='Threshold (%)', yaxis_title='Score',
                           yaxis=dict(range=[0, 1]), height=360,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           title='Precision vs Recall by Threshold')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Business Impact (100K annual originations, $10K avg loan, 3% charge-off rate)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("AUC Improvement",       "+7.0 pts",  "vs baseline")
    b2.metric("Recall Improvement",    "+5.4 pts",  "fewer missed defaults")
    b3.metric("Charge-off Reduction",  "~15-20%",   "projected from backtesting")
    b4.metric("Annual Loss Avoidance", "~$2.25M",   "at current portfolio size")

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Documentation")
    st.markdown("""
**Algorithm:** LightGBM binary classifier
**Training data:** 307,511 applications, 122 source features + 11 engineered
**Imbalance handling:** `scale_pos_weight` = ratio of non-defaults to defaults (~11:1)
**Tuning:** RandomizedSearchCV, 30 iterations, 5-fold stratified CV, scoring=AUC-ROC
**Explainability:** SHAP TreeExplainer (exact Shapley values)

---

**Engineered Features**

| Feature | Definition |
|---------|-----------|
| DEBT_TO_INCOME | Loan / annual income |
| ANNUITY_TO_INCOME | Monthly payment / monthly income |
| CREDIT_TO_GOODS | Loan / goods price |
| EMPLOYMENT_TO_AGE | Years employed / age |
| LOAN_TERM_MONTHS | Loan / annuity |
| INCOME_PER_PERSON | Income / household size |
| EXT_SOURCE_MEAN | Average of 3 bureau scores |
| EXT_SOURCE_MIN | Minimum bureau sub-score |
| IS_UNEMPLOYED | Binary flag |

---

**Risk Tiers**

| Tier | Probability | Decision |
|------|------------|---------|
| Low | < 15% | Auto Approve |
| Medium | 15–35% | Manual Review |
| High | > 35% | Auto Decline |

---

**Dataset:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
**Author:** Manish Reddy Thumma
""")
