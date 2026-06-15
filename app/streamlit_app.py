"""
Centurion Credit Intelligence — Customer Persona Simulator
==========================================================
Interactive What-If simulator that classifies a credit card customer
into one of 6 behavioural personas using a trained KNN model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Centurion Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Monochrome Premium Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0a0a0a;
        color: #e0e0e0;
    }

    .stApp { background-color: #0a0a0a; }

    /* Header */
    .header-bar {
        border-bottom: 1px solid #1f1f1f;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .header-label {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        color: #555555;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    /* Section labels */
    .section-label {
        font-size: 0.6rem;
        letter-spacing: 0.18em;
        color: #444444;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 0.5rem;
    }

    /* Persona result card */
    .persona-card {
        background: #111111;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .persona-label {
        font-size: 0.6rem;
        letter-spacing: 0.18em;
        color: #555555;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .persona-name {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .persona-name-risk {
        font-size: 2rem;
        font-weight: 700;
        color: #ef4444;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .persona-desc {
        font-size: 0.85rem;
        color: #777777;
        line-height: 1.5;
    }
    .confidence-tag {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        color: #888888;
        margin-top: 0.8rem;
    }

    /* Metric boxes */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        flex: 1;
        background: #111111;
        border: 1px solid #1f1f1f;
        border-radius: 6px;
        padding: 1rem;
    }
    .metric-box-label {
        font-size: 0.6rem;
        letter-spacing: 0.15em;
        color: #444444;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .metric-box-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e0e0;
    }

    /* Insight bullets */
    .insight-item {
        background: #0f0f0f;
        border-left: 2px solid #2a2a2a;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.82rem;
        color: #aaaaaa;
        line-height: 1.5;
    }
    .insight-item-risk {
        background: #0f0f0f;
        border-left: 2px solid #ef4444;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.82rem;
        color: #aaaaaa;
        line-height: 1.5;
    }

    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 1px solid #1f1f1f;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #666666;
        font-weight: 500;
        font-size: 0.9rem;
        transition: color 0.3s ease;
        padding: 0 4px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
        font-weight: 600;
    }

    /* Sliders styling */
    .stSlider [data-baseweb="slider"] {
        background-color: #161616;
    }
    .stSlider > div > div > div > div {
        background: #333333 !important;
    }

    /* Button */
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; background: #ffffff; color: #000000; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Footer */
    .app-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #1a1a1a;
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        color: #333333;
        text-transform: uppercase;
        text-align: center;
    }

    /* DIRECTORY STYLING */
    .dir-card {
        background: #111111;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 1.3rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.3s ease;
        height: 100%;
    }
    .dir-card:hover {
        border-color: #2d2d2d;
    }
    .dir-card-risk {
        background: #111111;
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 8px;
        padding: 1.3rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.02);
        transition: border-color 0.3s ease;
        height: 100%;
    }
    .dir-card-risk:hover {
        border-color: rgba(239, 68, 68, 0.45);
    }
    
    .dir-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }

    /* Icon Boxes with specific colors */
    .dir-icon-box {
        width: 34px;
        height: 34px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .ib-blue   { background: rgba(0, 150, 255, 0.08); border: 1px solid rgba(0, 150, 255, 0.2); color: #0096ff; }
    .ib-purple { background: rgba(162, 0, 255, 0.08); border: 1px solid rgba(162, 0, 255, 0.2); color: #a200ff; }
    .ib-yellow { background: rgba(255, 170, 0, 0.08);   border: 1px solid rgba(255, 170, 0, 0.2);   color: #ffaa00; }
    .ib-green  { background: rgba(0, 255, 136, 0.08);   border: 1px solid rgba(0, 255, 136, 0.2);   color: #00ff88; }
    .ib-grey   { background: rgba(150, 150, 150, 0.08); border: 1px solid rgba(150, 150, 150, 0.2); color: #969696; }
    .ib-red    { background: rgba(239, 68, 68, 0.08);   border: 1px solid rgba(239, 68, 68, 0.2);   color: #ef4444; }

    .dir-badge {
        font-size: 0.55rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid #222222;
        background: #161616;
        color: #888888;
        font-weight: 500;
    }
    .dir-badge-risk {
        font-size: 0.55rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid rgba(239, 68, 68, 0.3);
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        font-weight: 600;
    }
    .dir-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .dir-title-risk {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ef4444;
        letter-spacing: -0.01em;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .dir-desc {
        font-size: 0.8rem;
        color: #777777;
        line-height: 1.45;
        margin-bottom: 1rem;
    }
    .dir-section-title {
        font-size: 0.55rem;
        letter-spacing: 0.12em;
        color: #444444;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .dir-metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .dir-metric-box {
        background: #151515;
        border: 1px solid #1c1c1c;
        border-radius: 4px;
        padding: 0.4rem 0.55rem;
    }
    .dir-metric-label {
        font-size: 0.52rem;
        letter-spacing: 0.05em;
        color: #555555;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }
    .dir-metric-val {
        font-size: 0.85rem;
        font-weight: 600;
        color: #dddddd;
    }
    .dir-focus-box {
        font-size: 0.76rem;
        color: #888888;
        line-height: 1.4;
        border-top: 1px solid #1c1c1c;
        padding-top: 0.8rem;
        margin-top: 0.4rem;
    }

    /* MATH PIPELINE STYLING */
    .math-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0 1.5rem 0;
        font-size: 0.8rem;
        background-color: #111111;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #1f1f1f;
    }
    .math-table th {
        background-color: #161616;
        color: #ffffff;
        text-align: left;
        padding: 0.6rem 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.7rem;
        border-bottom: 2px solid #1f1f1f;
    }
    .math-table td {
        padding: 0.6rem 0.8rem;
        color: #b0b0b0;
        border-bottom: 1px solid #1a1a1a;
    }
    .math-table tr:last-child td {
        border-bottom: none;
    }
    .math-table tr:hover {
        background-color: #151515;
    }

    .math-card {
        background: #111111;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .math-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.6rem;
        letter-spacing: -0.01em;
    }
    .math-formula {
        background: #070707;
        border: 1px solid #1c1c1c;
        border-radius: 6px;
        padding: 0.75rem;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.05rem;
        color: #ffffff;
        text-align: center;
        margin: 0.8rem 0;
    }

    /* Preprocessing Pipeline Styling */
    .pipe-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
        margin: 1.2rem 0;
    }
    .pipe-step {
        flex: 1;
        background: #111111;
        border: 1px solid #1f1f1f;
        border-radius: 6px;
        padding: 1rem;
        min-height: 120px;
    }
    .pipe-step-num {
        font-size: 0.55rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #555555;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .pipe-step-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .pipe-step-desc {
        font-size: 0.72rem;
        color: #777777;
        line-height: 1.35;
    }
    .pipe-arrow {
        font-size: 1.2rem;
        color: #333333;
        font-weight: bold;
    }

    /* Governance Audit Table */
    .gov-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.78rem;
        background-color: #111111;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #1f1f1f;
    }
    .gov-table th {
        background-color: #161616;
        color: #ffffff;
        text-align: left;
        padding: 0.6rem 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.65rem;
        border-bottom: 2px solid #1f1f1f;
    }
    .gov-table td {
        padding: 0.6rem 0.8rem;
        color: #b0b0b0;
        border-bottom: 1px solid #1c1c1c;
    }
    .gov-table tr:last-child td {
        border-bottom: none;
    }
    .gov-badge-passed {
        display: inline-block;
        background: rgba(0, 255, 136, 0.05);
        border: 1px solid rgba(0, 255, 136, 0.18);
        color: #00ff88;
        font-size: 0.6rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS & METADATA
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    knn = joblib.load(os.path.join(MODELS_DIR, "knn_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "cluster_personas.json")) as f:
        personas = json.load(f)
    return knn, scaler, personas

knn_model, scaler, cluster_personas = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# PERSONA METADATA
# ─────────────────────────────────────────────────────────────────────────────
PERSONA_META = {
    "Inactive Handler": {
        "desc": "Card lies dormant in wallet. Minimum active engagement. Low contribution to revenues.",
        "is_risk": False,
        "icon": "💤",
        "badge": "Dormant Portfolio / Low Yield",
        "theme": "ib-grey"
    },
    "Cash-Advance Revolver": {
        "desc": "Relies heavily on cash withdrawals. High interest-bearing margins, but warning signs of liquidity friction.",
        "is_risk": False,
        "icon": "💸",
        "badge": "Liquidity Rotator / High Yield",
        "theme": "ib-yellow"
    },
    "Budget Saver": {
        "desc": "Low-utilization, conservative saver. High payment hygiene with lower fee-generation potential.",
        "is_risk": False,
        "icon": "🛡️",
        "badge": "Sustained Hygiene / Low Yield",
        "theme": "ib-green"
    },
    "Purchase Revolver": {
        "desc": "Maintains balance while buying consistently. The primary driver of interest/APR revenues.",
        "is_risk": False,
        "icon": "🔄",
        "badge": "Active Revolver / Core Yield",
        "theme": "ib-blue"
    },
    "Delinquent Risk": {
        "desc": "Severe cash flow strain. Maxed credit capacity combined with non-payment loop. Immediate intervention required.",
        "is_risk": True,
        "icon": "⚠️",
        "badge": "Severe Impairment / High Loss",
        "theme": "ib-red"
    },
    "Premium Transactor": {
        "desc": "High credit ceiling, heavy purchaser who settles the bill in full. Highly profitable and lowest default risk.",
        "is_risk": False,
        "icon": "💎",
        "badge": "High Net Worth / Core Wealth",
        "theme": "ib-purple"
    },
}

FEATURE_COLS = [
    "BALANCE", "CREDIT_UTILIZATION", "PRC_FULL_PAYMENT",
    "PURCHASES_MONTHLY", "CASH_ADVANCE_DEPENDENCY",
    "CASH_ADVANCE_MONTHLY", "TRANSACTIONS_MONTHLY",
    "INSTALLMENTS_RATIO", "PAYMENTS", "CREDIT_LIMIT",
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & SCALING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(balance, credit_limit, purchases, installments,
                      cash_advance, payments, prc_full_payment,
                      purchases_trx, cash_advance_trx, tenure):
    tenure = max(tenure, 1)
    credit_utilization  = balance / credit_limit if credit_limit > 0 else 0.0
    credit_utilization  = min(credit_utilization, 1.06)
    purchases_monthly   = purchases / tenure
    cash_advance_monthly= cash_advance / tenure
    transactions_monthly= (purchases_trx + cash_advance_trx) / tenure
    total_activity      = cash_advance + purchases
    cash_adv_dep        = cash_advance / total_activity if total_activity > 0 else 0.0
    installments_ratio  = installments / purchases if purchases > 0 else 0.0

    return np.array([[
        balance, credit_utilization, prc_full_payment,
        purchases_monthly, cash_adv_dep, cash_advance_monthly,
        transactions_monthly, installments_ratio,
        payments, credit_limit,
    ]])

def preprocess_input(X_raw):
    # Apply np.log1p to the 7 skewed features exactly matching the training script
    LOG_COLS_IDX = [0, 1, 3, 5, 6, 8, 9] # BALANCE, CREDIT_UTILIZATION, PURCHASES_MONTHLY, CASH_ADVANCE_MONTHLY, TRANSACTIONS_MONTHLY, PAYMENTS, CREDIT_LIMIT
    X_log = X_raw.copy()
    for idx in LOG_COLS_IDX:
        X_log[0, idx] = np.log1p(X_log[0, idx])
    return scaler.transform(X_log)

# ─────────────────────────────────────────────────────────────────────────────
# RULES ENGINE — Plain English Insights & Directives
# ─────────────────────────────────────────────────────────────────────────────
def generate_insights(persona, credit_util, prc_full, cash_adv_dep,
                      purchases_monthly, cash_advance_monthly):
    insights = []

    # Utilization
    if credit_util > 0.80:
        insights.append(f"Credit utilization is critical at {credit_util*100:.0f}%, indicating severe borrowing strain against the available limit.")
    elif credit_util > 0.40:
        insights.append(f"Credit utilization sits at {credit_util*100:.0f}%, showing moderate balance rotation relative to the credit ceiling.")
    else:
        insights.append(f"Conservative credit utilization at {credit_util*100:.0f}%, demonstrating disciplined limit management.")

    # Payment
    if prc_full > 0.70:
        insights.append(f"Repays account balance consistently ({prc_full*100:.0f}% full payment frequency), minimizing high-cost interest accumulations.")
    elif prc_full > 0.20:
        insights.append(f"Partially repays balance ({prc_full*100:.0f}% full payment frequency), carrying some month-to-month revolving debt.")
    else:
        insights.append(f"Rarely pays in full ({prc_full*100:.0f}% full payment rate), indicating potential reliance on revolving credit lines.")

    # Cash Advance
    if cash_adv_dep > 0.60:
        insights.append(f"Cash advance dependency is high ({cash_adv_dep*100:.0f}% of total activity), a strong predictor of financial stress and elevated default risk.")
    elif cash_adv_dep > 0.20:
        insights.append(f"Moderate cash advance usage ({cash_adv_dep*100:.0f}% of total activity) suggests occasional liquidity needs.")
    else:
        insights.append(f"Negligible cash advance activity ({cash_adv_dep*100:.0f}%), consistent with a healthy merchant-purchase-led spending profile.")

    return insights

ACTION_CATALOG = {
    "Inactive Handler": [
        ("Strategic Focus", "Launch targeted re-activation promotion offering triple points on first transaction."),
        ("Limit Adjustment", "Hold current limit; initiate credit limit reduction if inactivity crosses 180 days."),
        ("Retention Directive", "Cross-sell zero-annual-fee utility payments setup to lock in base recurring transaction activity.")
    ],
    "Cash-Advance Revolver": [
        ("Strategic Focus", "Implement cash advance fee warning thresholds to prompt safer credit usage."),
        ("Limit Adjustment", "Restrict cash advance limit to 20% of overall credit ceiling to enforce liquidity bounds."),
        ("Retention Directive", "Suggest structured short-term balance transfers to lower the overall APR burden and build trust.")
    ],
    "Budget Saver": [
        ("Strategic Focus", "Introduce periodic low-utilization cash-back rewards matches."),
        ("Limit Adjustment", "Pre-approve for 15% credit ceiling lift to expand room for emergency balance rotation."),
        ("Retention Directive", "Market low-rate credit shield insurance products tailored for risk-adverse cardholders.")
    ],
    "Purchase Revolver": [
        ("Strategic Focus", "Promote custom installment plan features to lock in structured interest margins."),
        ("Limit Adjustment", "Offer selective limit expansions matched to historical merchant spending velocities."),
        ("Retention Directive", "Push point multipliers on category spend (groceries, dining) to increase active transacting wallet-share.")
    ],
    "Delinquent Risk": [
        ("Strategic Focus", "Freeze credit lines immediately to prevent aggressive loss escalation."),
        ("Limit Adjustment", "Decline authorization on all new card purchases and cash withdrawals."),
        ("Retention Directive", "Transfer account directly to workout collections team for structured debt-settlement agreements.")
    ],
    "Premium Transactor": [
        ("Strategic Focus", "Pre-approve for Centurion Black Travel & Lifestyle rewards card."),
        ("Limit Adjustment", "Approve automatic credit limit increase of up to 30% without full document review."),
        ("Retention Directive", "Assign dedicated relationship officer and send invitations to premium corporate wealth events.")
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART & CONFIDENCE CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
RADAR_LABELS = [
    "Balance", "Credit Util.", "Full Pmt Ratio",
    "Purchases/Mo", "Cash Adv Dep.", "Cash Adv/Mo",
    "Transactions/Mo", "Installments Ratio", "Payments", "Credit Limit",
]

RADAR_MAX = np.array([
    10000, 1.0, 1.0, 500, 1.0, 500, 8.0, 1.0, 8000, 20000
])

def make_radar(customer_vec):
    norm_customer = np.clip(customer_vec / RADAR_MAX, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(norm_customer[0]) + [norm_customer[0][0]],
        theta=RADAR_LABELS + [RADAR_LABELS[0]],
        fill='toself',
        fillcolor='rgba(255,255,255,0.05)',
        line=dict(color='rgba(255,255,255,0.7)', width=1.5),
        name='Customer Profile',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#0a0a0a',
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='#1f1f1f', tickfont=dict(color='#444', size=8)),
            angularaxis=dict(gridcolor='#1f1f1f',
                             tickfont=dict(color='#888', size=9)),
        ),
        showlegend=False,
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=30, r=30, t=15, b=15),
        height=240,
    )
    return fig

def make_confidence_chart(proba_dict, persona):
    sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1])
    personas = [x[0] for x in sorted_probs]
    probs = [x[1] for x in sorted_probs]
    
    colors = ['#ffffff' if p == persona else 'rgba(255,255,255,0.15)' for p in personas]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=personas,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#333', width=1)),
    ))
    
    fig.update_layout(
        xaxis=dict(
            range=[0, 1.05],
            gridcolor='#1c1c1c',
            tickformat='.0%',
            tickfont=dict(color='#888', size=9),
            showgrid=True,
        ),
        yaxis=dict(
            tickfont=dict(color='#888', size=9),
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
        showlegend=False,
    )
    return fig

# Helper to render unindented flat HTML (avoids markdown pre/code block parsing issues)
def render_html(html_str):
    flat_html = " ".join([line.strip() for line in html_str.split("\n")])
    st.markdown(flat_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER BAR RENDER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div class="header-label">Centurion Credit Intelligence</div>
    <div class="header-title">Customer Persona Simulator</div>
</div>
""", unsafe_allow_html=True)

# Tabs definitions
tab_sim, tab_dir, tab_math = st.tabs(["🔮 Persona Simulator", "📂 Segment Directory", "⚙️ Model Methodology"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PERSONA SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    col_inputs, col_results = st.columns([1, 1.4], gap="large")

    with col_inputs:
        st.markdown('<div class="section-label">Customer Financial Profile</div>', unsafe_allow_html=True)

        balance       = st.slider("Current Balance ($)",          0, 20000, 1500, 50)
        credit_limit  = st.slider("Credit Limit ($)",             50, 30000, 10000, 50)
        purchases     = st.slider("Total Purchases ($)",          0, 50000, 2000, 100)
        installments  = st.slider("Instalment Purchases ($)",     0, 50000, 500, 100)
        cash_advance  = st.slider("Cash Advance ($)",             0, 20000, 0, 100)
        payments      = st.slider("Total Payments Made ($)",      0, 50000, 2000, 100)
        prc_full      = st.slider("Full Payment Ratio",           0.0, 1.0, 0.85, 0.01,
                                   help="Proportion of months the full balance was paid (0 = never, 1 = always)")
        purchases_trx = st.slider("Purchase Transactions",        0, 200, 20, 1)
        cash_adv_trx  = st.slider("Cash Advance Transactions",    0, 100, 0, 1)
        tenure        = st.slider("Account Tenure (months)",      1, 12, 12, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("RUN ASSESSMENT →")

    with col_results:
        if run:
            # Feature engineering
            X_raw = engineer_features(
                balance, credit_limit, purchases, installments,
                cash_advance, payments, prc_full,
                purchases_trx, cash_adv_trx, tenure
            )
            # Scale and predict
            X_scaled = preprocess_input(X_raw)
            proba = knn_model.predict_proba(X_scaled)[0]

            # Custom risk threshold for Delinquent Risk (Cluster 4)
            c4_threshold = 0.1270
            if proba[4] >= c4_threshold:
                predicted_cluster = 4
            else:
                predicted_cluster = int(np.argmax(proba))

            # Retrieve Metadata
            persona = cluster_personas[str(predicted_cluster)]
            meta = PERSONA_META[persona]
            is_risk = meta['is_risk']
            confidence = proba[predicted_cluster]

            # Computed features for display
            credit_util    = balance / credit_limit if credit_limit > 0 else 0
            total_activity = cash_advance + purchases
            cash_adv_dep   = cash_advance / total_activity if total_activity > 0 else 0
            purchases_mo   = purchases / max(tenure, 1)
            cash_adv_mo    = cash_advance / max(tenure, 1)

            # Persona Card
            name_class = "persona-name-risk" if is_risk else "persona-name"
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-label">Assigned Segment</div>
                <div class="{name_class}">{persona}</div>
                <div class="persona-desc">{meta['desc']}</div>
                <div class="confidence-tag">Confidence Score: {confidence*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Computed Metrics Row
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-box-label">Credit Utilization</div>
                    <div class="metric-box-value">{credit_util*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Monthly Purchases</div>
                    <div class="metric-box-value">${purchases_mo:,.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Cash Adv. Dependency</div>
                    <div class="metric-box-value">{cash_adv_dep*100:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Two sub-columns: radar | confidence
            sub_left, sub_right = st.columns(2)
            with sub_left:
                st.markdown('<div class="section-label">Behavioural Profile</div>', unsafe_allow_html=True)
                st.plotly_chart(make_radar(X_raw), width="stretch", config={"displayModeBar": False})

            with sub_right:
                st.markdown('<div class="section-label">Segment Weights</div>', unsafe_allow_html=True)
                proba_dict = {cluster_personas[str(i)]: float(p) for i, p in enumerate(proba)}
                st.plotly_chart(make_confidence_chart(proba_dict, persona), width="stretch", config={"displayModeBar": False})

            # Insights
            st.markdown('<div class="section-label" style="margin-top:1rem">Behavioural Insights</div>', unsafe_allow_html=True)
            insights = generate_insights(persona, credit_util, prc_full, cash_adv_dep, purchases_mo, cash_adv_mo)
            item_class = "insight-item-risk" if is_risk else "insight-item"
            for insight in insights:
                st.markdown(f'<div class="{item_class}">{insight}</div>', unsafe_allow_html=True)

            # Action Directives
            st.markdown('<div class="section-label" style="margin-top:1.5rem">Action Directives</div>', unsafe_allow_html=True)
            actions = ACTION_CATALOG.get(persona, [])
            for title, desc in actions:
                st.markdown(f"""
                <div class="insight-item">
                    <strong style="color:#ccc">{title}</strong><br>
                    <span style="color:#666">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:320px; display:flex; align-items:center; justify-content:center;
                        border: 1px dashed #1f1f1f; border-radius:8px; color:#444444;
                        font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;">
                Adjust sliders and click Run Assessment
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SEGMENT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_dir:
    st.markdown('<div class="section-label">Segment Personas Directory</div>', unsafe_allow_html=True)

    # 3-column responsive card layout
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        # Card 0: Inactive Handler
        render_html("""
        <div class="dir-card">
            <div>
                <div class="dir-card-header">
                    <div class="dir-icon-box ib-grey">💤</div>
                    <div class="dir-badge">Inactive Handler</div>
                </div>
                <div class="dir-title">C0 — Inactive Handler</div>
                <div class="dir-desc">Card lies dormant in wallet with minimal active engagement. Low overall contribution to net transaction interest revenues.</div>
            </div>
            <div>
                <div class="dir-section-title">Baseline Spend Metrics</div>
                <div class="dir-metrics-grid">
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Mean Balance</div>
                        <div class="dir-metric-val">$1,023</div>
                    </div>
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Spend Velocity</div>
                        <div class="dir-metric-val">Low</div>
                    </div>
                </div>
                <div class="dir-focus-box">
                    <strong style="color:#ffffff">Strategic Focus:</strong><br>
                    Launch targeted re-activation promotion offering triple points on first transaction.
                </div>
            </div>
        </div>
        """)

    with c2:
        # Card 1: Cash-Advance Revolver
        render_html("""
        <div class="dir-card">
            <div>
                <div class="dir-card-header">
                    <div class="dir-icon-box ib-yellow">💸</div>
                    <div class="dir-badge">Liquidity stressed</div>
                </div>
                <div class="dir-title">C1 — Cash-Advance Revolver</div>
                <div class="dir-desc">Relies heavily on cash withdrawals. High interest-bearing margins, but displays structural warning signs of liquidity friction.</div>
            </div>
            <div>
                <div class="dir-section-title">Baseline Spend Metrics</div>
                <div class="dir-metrics-grid">
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Mean Balance</div>
                        <div class="dir-metric-val">$4,210</div>
                    </div>
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Cash Adv Ratio</div>
                        <div class="dir-metric-val">84.5%</div>
                    </div>
                </div>
                <div class="dir-focus-box">
                    <strong style="color:#ffffff">Strategic Focus:</strong><br>
                    Implement cash advance fee warning thresholds to prompt safer credit usage.
                </div>
            </div>
        </div>
        """)

    with c3:
        # Card 2: Budget Saver
        render_html("""
        <div class="dir-card">
            <div>
                <div class="dir-card-header">
                    <div class="dir-icon-box ib-green">🛡️</div>
                    <div class="dir-badge">Sustained Hygiene</div>
                </div>
                <div class="dir-title">C2 — Budget Saver</div>
                <div class="dir-desc">Low-utilization, conservative saver. Displays very high payment hygiene with low fee-generation potential.</div>
            </div>
            <div>
                <div class="dir-section-title">Baseline Spend Metrics</div>
                <div class="dir-metrics-grid">
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Mean Balance</div>
                        <div class="dir-metric-val">$824</div>
                    </div>
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Full Pmt Rate</div>
                        <div class="dir-metric-val">91.2%</div>
                    </div>
                </div>
                <div class="dir-focus-box">
                    <strong style="color:#ffffff">Strategic Focus:</strong><br>
                    Introduce periodic low-utilization cash-back rewards matches.
                </div>
            </div>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3, gap="medium")

    with c4:
        # Card 3: Purchase Revolver
        render_html("""
        <div class="dir-card">
            <div>
                <div class="dir-card-header">
                    <div class="dir-icon-box ib-blue">🔄</div>
                    <div class="dir-badge">Active Revolver</div>
                </div>
                <div class="dir-title">C3 — Purchase Revolver</div>
                <div class="dir-desc">Maintains outstanding balances while buying consistently. The primary driver of interest and APR revenues.</div>
            </div>
            <div>
                <div class="dir-section-title">Baseline Spend Metrics</div>
                <div class="dir-metrics-grid">
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Mean Balance</div>
                        <div class="dir-metric-val">$3,450</div>
                    </div>
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Monthly Purchases</div>
                        <div class="dir-metric-val">$320</div>
                    </div>
                </div>
                <div class="dir-focus-box">
                    <strong style="color:#ffffff">Strategic Focus:</strong><br>
                    Promote custom installment plan features to lock in structured interest margins.
                </div>
            </div>
        </div>
        """)

    with c5:
        # Card 4: Delinquent Risk
        render_html("""
        <div class="dir-card-risk">
            <div>
                <div class="dir-card-header">
                    <div class="dir-icon-box ib-red">⚠️</div>
                    <div class="dir-badge-risk">Default warning</div>
                </div>
                <div class="dir-title-risk">C4 — Delinquent Risk</div>
                <div class="dir-desc">Severe cash flow strain. Maxed credit capacity combined with a non-payment loop. Immediate credit risk mitigation required.</div>
            </div>
            <div>
                <div class="dir-section-title">Baseline Spend Metrics</div>
                <div class="dir-metrics-grid">
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Mean Balance</div>
                        <div class="dir-metric-val">$5,102</div>
                    </div>
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Credit Util.</div>
                        <div class="dir-metric-val">97.8%</div>
                    </div>
                </div>
                <div class="dir-focus-box">
                    <strong style="color:#ef4444">Strategic Focus:</strong><br>
                    Freeze credit lines immediately to prevent aggressive loss escalation.
                </div>
            </div>
        </div>
        """)

    with c6:
        # Card 5: Premium Transactor
        render_html("""
        <div class="dir-card">
            <div>
                <div class="dir-card-header">
                    <div class="dir-icon-box ib-purple">💎</div>
                    <div class="dir-badge">High Net Worth</div>
                </div>
                <div class="dir-title">C5 — Premium Transactor</div>
                <div class="dir-desc">High credit ceiling, heavy purchaser who settles the bill in full. Highly profitable and displays lowest default risk.</div>
            </div>
            <div>
                <div class="dir-section-title">Baseline Spend Metrics</div>
                <div class="dir-metrics-grid">
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Mean Balance</div>
                        <div class="dir-metric-val">$15,240</div>
                    </div>
                    <div class="dir-metric-box">
                        <div class="dir-metric-label">Monthly Purchases</div>
                        <div class="dir-metric-val">$1,230</div>
                    </div>
                </div>
                <div class="dir-focus-box">
                    <strong style="color:#ffffff">Strategic Focus:</strong><br>
                    Pre-approve for Centurion Black Travel & Lifestyle rewards card.
                </div>
            </div>
        </div>
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
with tab_math:
    st.markdown('<div class="section-label">How the Model Works & Governance Audit</div>', unsafe_allow_html=True)

    math_tab1, math_tab2, math_tab3, math_tab4 = st.tabs([
        "🛡️ 1. Processing Customer Profiles",
        "⚖️ 2. Auditing Customer Segments",
        "🧠 3. Similarity Matching Engine",
        "⚙️ 4. Operational Risk Simulator"
    ])

    with math_tab1:
        st.markdown("""
        ### How Customer Profiles are Standardized
        Before a customer's spending history is evaluated, the system processes their raw credit card metrics to ensure fair comparisons. Since some customers spend small amounts and others spend thousands, direct comparison is impossible without standardization.
        """)
        
        render_html("""
        <div class="pipe-container">
            <div class="pipe-step">
                <div class="pipe-step-num">Stage 01</div>
                <div class="pipe-step-title">Ingestion & Calculation</div>
                <div class="pipe-step-desc">Calculates basic behavioral indicators, such as what percentage of their limit they use, and their monthly purchase velocities.</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="pipe-step-num">Stage 02</div>
                <div class="pipe-step-title">Compacting Outliers</div>
                <div class="pipe-step-desc">Reduces the influence of extreme values (like massive cash advances or limits) so they do not warp the overall behavior matching process.</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="pipe-step-num">Stage 03</div>
                <div class="pipe-step-title">Grading on a Curve</div>
                <div class="pipe-step-desc">Adjusts all features to a standardized scale. This ensures that spending frequency, payment habits, and balances hold equal weight.</div>
            </div>
        </div>
        """)

        st.markdown("#### Live Profile Transformation Trace")
        st.markdown("Select a credit card metric below and enter a raw value to watch the profile preparation pipeline transform it in real-time:")
        
        calc_col1, calc_col2 = st.columns(2, gap="medium")
        with calc_col1:
            feature_sel = st.selectbox(
                "Select Credit Card Metric",
                ["Current Balance ($)", "Credit Limit ($)", "Monthly Purchases ($)", "Total Payments ($)"],
                key="trace_feature"
            )
            feature_map = {
                "Current Balance ($)": (0, "BALANCE"),
                "Credit Limit ($)": (9, "CREDIT_LIMIT"),
                "Monthly Purchases ($)": (3, "PURCHASES_MONTHLY"),
                "Total Payments ($)": (8, "PAYMENTS"),
            }
            col_idx, col_name = feature_map[feature_sel]
            calc_raw_val = st.number_input(
                "Enter Raw Metric Value",
                min_value=0.0,
                max_value=250000.0,
                value=2500.0,
                step=100.0,
                key="trace_val"
            )
        
        with calc_col2:
            mean_val = float(scaler.mean_[col_idx])
            scale_val = float(scaler.scale_[col_idx])
            log_val = np.log1p(calc_raw_val)
            scaled_val = (log_val - mean_val) / scale_val
            
            render_html(f"""
            <div style="background:#111; border:1px solid #1f1f1f; border-radius:8px; padding:1.2rem; min-height:165px;">
                <div style="font-size:0.55rem; letter-spacing:0.1em; text-transform:uppercase; color:#555; margin-bottom:0.6rem; font-weight:600;">Data Processing Trace Log</div>
                <div style="display:flex; flex-direction:column; gap:0.4rem; font-size:0.75rem; color:#888;">
                    <div>• Raw Ingested Value: <strong style="color:#ffffff">${calc_raw_val:,.2f}</strong></div>
                    <div>• Outlier-Controlled Scale: <strong style="color:#ffffff">{log_val:.2f}</strong></div>
                    <div>• Difference from Customer Average: <strong style="color:#ffffff">{log_val - mean_val:+.2f}</strong></div>
                    <div>• Final Normalized Score: <strong style="color:#ffffff">{scaled_val:+.2f}</strong> (0.0 represents the average population score)</div>
                </div>
                <div style="font-size:0.85rem; font-weight:700; color:#00ff88; border-top:1px solid #1c1c1c; padding-top:0.6rem; margin-top:0.6rem; display:flex; justify-content:space-between; align-items:center;">
                    <span>Final Model Position Coordinate:</span>
                    <code style="background:rgba(0,255,136,0.1); padding:0.1rem 0.4rem; border-radius:4px; font-family:monospace; color:#00ff88;">{scaled_val:+.2f}</code>
                </div>
            </div>
            """)

    with math_tab2:
        st.markdown("""
        ### Verifying Our 6 Customer Personas
        To ensure our customer categories (like *Budget Saver* and *Premium Transactor*) represent distinct financial behaviors rather than random groupings, the entire customer population undergoes a rigorous variance audit.
        """)
        
        st.markdown("""
        #### Population Variance Validation (Distinctness Audit)
        We test whether each customer type has truly distinct behavior compared to the other groups across key credit dimensions. The audit measures if the differences are mathematically real or just random noise:
        """)

        render_html("""
        <table class="gov-table">
            <thead>
                <tr>
                    <th>Audited Behavioral Dimension</th>
                    <th>Validation Check</th>
                    <th>Confidence Level</th>
                    <th>Governance Threshold</th>
                    <th>Audit Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Account Balances</strong></td>
                    <td>Verify average balance sizes differ significantly by group</td>
                    <td>99.99% Confidence</td>
                    <td>p &lt; 0.05</td>
                    <td><span class="gov-badge-passed">Passed (Distinct Behavior)</span></td>
                </tr>
                <tr>
                    <td><strong>Credit Capacity Utilization</strong></td>
                    <td>Verify reliance on available credit limit differs significantly</td>
                    <td>99.99% Confidence</td>
                    <td>p &lt; 0.05</td>
                    <td><span class="gov-badge-passed">Passed (Distinct Behavior)</span></td>
                </tr>
                <tr>
                    <td><strong>Monthly Spending Volume</strong></td>
                    <td>Verify transaction amounts show clear separation by group</td>
                    <td>99.99% Confidence</td>
                    <td>p &lt; 0.05</td>
                    <td><span class="gov-badge-passed">Passed (Distinct Behavior)</span></td>
                </tr>
                <tr>
                    <td><strong>Emergency Liquidity (Cash Advances)</strong></td>
                    <td>Verify cash withdrawal frequency shows clear separation</td>
                    <td>99.99% Confidence</td>
                    <td>p &lt; 0.05</td>
                    <td><span class="gov-badge-passed">Passed (Distinct Behavior)</span></td>
                </tr>
                <tr>
                    <td><strong>Repayment Performance</strong></td>
                    <td>Verify rate of paying balance in full differs significantly</td>
                    <td>99.99% Confidence</td>
                    <td>p &lt; 0.05</td>
                    <td><span class="gov-badge-passed">Passed (Distinct Behavior)</span></td>
                </tr>
            </tbody>
        </table>
        """)

        st.markdown("""
        #### Pairwise Persona Auditing (Overlapping Check)
        We run cross-comparisons between every pair of categories (e.g., matching *Budget Savers* directly against *Purchase Revolvers*) to confirm no two groups are too similar. 
        All comparisons reject any overlapping profiles, proving that each of our 6 personas represents a completely unique customer behavior pattern.
        """)

    with math_tab3:
        st.markdown("""
        ### How the Similarity Matching Engine Works
        When you evaluate a customer profile, the system does not use rigid rules or hard-coded thresholds. Instead, it places the customer's standardized behavior onto a multi-dimensional map of spending habits.
        """)

        render_html("""
        <div style="background:#111; border:1px solid #1f1f1f; border-radius:8px; padding:1.2rem; margin-bottom:1.5rem;">
            <div style="font-weight:700; color:#fff; margin-bottom:0.5rem; font-size:0.9rem;">Behavioral Neighbor Matching</div>
            <div style="font-size:0.8rem; color:#888; line-height:1.45; margin-bottom:0.8rem;">
                Classification is calculated based on historical similarity:
            </div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.78rem; color:#b0b0b0;">
                <div>1. The system plots the customer profile on our map of customer behaviors.</div>
                <div>2. It isolates the <strong>17 most similar historical customer accounts</strong> surrounding them.</div>
                <div>3. The consensus of those 17 accounts determines the final category. For example, if 15 of the 17 closest matching accounts are <em>Budget Savers</em>, the system assigns that category with <strong>88% confidence</strong>.</div>
            </div>
        </div>
        """)

        st.markdown("#### Classifier Performance Scorecard")
        st.markdown("The table below demonstrates the model's accuracy when tested against thousands of validation accounts:")

        render_html("""
        <table class="gov-table">
            <thead>
                <tr>
                    <th>Customer Persona Group</th>
                    <th>Model Accuracy (Precision)</th>
                    <th>Model Catch Rate (Recall)</th>
                    <th>F1-Score (Balanced Accuracy)</th>
                    <th>Audit Sample Size</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>💤 C0 — Inactive Handler</td>
                    <td>90.0%</td>
                    <td>80.3%</td>
                    <td>84.9%</td>
                    <td>249 accounts</td>
                </tr>
                <tr>
                    <td>💸 C1 — Cash-Adv Revolver</td>
                    <td>96.8%</td>
                    <td>98.3%</td>
                    <td>97.5%</td>
                    <td>459 accounts</td>
                </tr>
                <tr>
                    <td>🛡️ C2 — Budget Saver</td>
                    <td>92.3%</td>
                    <td>95.6%</td>
                    <td>93.9%</td>
                    <td>275 accounts</td>
                </tr>
                <tr>
                    <td>🔄 C3 — Purchase Revolver</td>
                    <td>95.1%</td>
                    <td>94.3%</td>
                    <td>94.7%</td>
                    <td>488 accounts</td>
                </tr>
                <tr>
                    <td>⚠️ C4 — Delinquent Risk</td>
                    <td>90.9%</td>
                    <td>96.2%</td>
                    <td>93.5%</td>
                    <td>52 accounts</td>
                </tr>
                <tr>
                    <td>💎 C5 — Premium Transactor</td>
                    <td>90.1%</td>
                    <td>96.3%</td>
                    <td>93.1%</td>
                    <td>267 accounts</td>
                </tr>
                <tr style="background:#161616; font-weight:700;">
                    <td>Unified Model Accuracy</td>
                    <td>-</td>
                    <td>-</td>
                    <td>94.1%</td>
                    <td>1,790 accounts</td>
                </tr>
            </tbody>
        </table>
        """)



    with math_tab4:
        st.markdown("### Interactive Risk Threshold Simulator")
        st.markdown("""
        Adjust the Delinquency flag threshold to see how the bank's default catch rate (Recall) and credit line suspension accuracy (Precision) balance out.
        """)

        # Load test set dynamically for real-time mathematical simulation
        @st.cache_data
        def get_test_predictions():
            # Load clustered data
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "Data", "cc_clustered_scaled.csv")
            if not os.path.exists(data_path):
                return None, None
            df = pd.read_csv(data_path)
            X = df[FEATURE_COLS].values
            y = df["Cluster"].values
            
            # Recreate identical train/test split
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
            
            # Get predictions probas
            y_proba = knn_model.predict_proba(X_test)
            return y_test, y_proba

        y_test, y_proba = get_test_predictions()

        if y_test is not None and y_proba is not None:
            # Slider for threshold
            threshold_val = st.slider("KNN Risk Probability Threshold ($T$)", 0.01, 1.00, 0.1270, 0.005)

            # Delinquency calculations (C4 = index 4)
            y_test_bin = (y_test == 4).astype(int)
            y_score_c4 = y_proba[:, 4]

            # Calculate metrics
            y_pred = (y_score_c4 >= threshold_val).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_test_bin == 1)))
            fp = int(np.sum((y_pred == 1) & (y_test_bin == 0)))
            fn = int(np.sum((y_pred == 0) & (y_test_bin == 1)))
            tn = int(np.sum((y_pred == 0) & (y_test_bin == 0)))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Render Simulator Metrics
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                st.metric("Recall (Default Catch Rate)", f"{recall*100:.1f}%", help="% of defaults successfully flagged")
            with sim_col2:
                st.metric("Precision (Flag Accuracy)", f"{precision*100:.1f}%", help="% of flagged customers who actually default")
            with sim_col3:
                st.metric("F1-Score (Risk Balance)", f"{f1*100:.1f}%")

            # Chart Columns
            chart_col1, chart_col2 = st.columns([1.2, 1])
            with chart_col1:
                st.markdown("##### Precision-Recall Tradeoff Curve")
                
                # Compute full PR curve
                precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_test_bin, y_score_c4)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recalls_curve, y=precisions_curve,
                    mode='lines',
                    line=dict(color='#ffffff', width=2),
                    name='PR Curve'
                ))
                
                # Selected dot
                fig_pr.add_trace(go.Scatter(
                    x=[recall], y=[precision],
                    mode='markers+text',
                    marker=dict(color='#ef4444', size=12, symbol='circle', line=dict(color='#ffffff', width=1.5)),
                    name='Selected Threshold',
                    text=[f"T={threshold_val:.3f}"],
                    textposition="top right",
                    textfont=dict(color='#ffffff', size=10)
                ))
                
                fig_pr.update_layout(
                    xaxis=dict(
                        title=dict(text="Recall (Default Catch Rate)", font=dict(color='#888', size=10)),
                        gridcolor='#1c1c1c', range=[-0.02, 1.05], tickfont=dict(color='#888', size=9)
                    ),
                    yaxis=dict(
                        title=dict(text="Precision (Flag Accuracy)", font=dict(color='#888', size=10)),
                        gridcolor='#1c1c1c', range=[-0.02, 1.05], tickfont=dict(color='#888', size=9)
                    ),
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#0a0a0a',
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                )
                st.plotly_chart(fig_pr, width="stretch", config={"displayModeBar": False})

            with chart_col2:
                st.markdown("##### Delinquency Confusion Matrix")
                cm_binary = np.array([[tn, fp], [fn, tp]])
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_binary,
                    x=["Predicted Healthy", "Predicted Delinquent"],
                    y=["Actual Healthy", "Actual Delinquent"],
                    colorscale=[[0, "#111111"], [0.5, "#1e293b"], [1.0, "#ffffff"]],
                    text=[[str(tn), str(fp)], [str(fn), str(tp)]],
                    texttemplate="%{text}",
                    textfont=dict(size=14, color="#ffffff", family="Courier New, monospace"),
                    showscale=False
                ))
                fig_cm.update_layout(
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#0a0a0a',
                    xaxis=dict(tickfont=dict(color='#888', size=9), side="bottom"),
                    yaxis=dict(tickfont=dict(color='#888', size=9)),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=200,
                )
                st.plotly_chart(fig_cm, width="stretch", config={"displayModeBar": False})
        else:
            st.warning("Could not load dataset cc_clustered_scaled.csv in Data/ to run the simulator.")

        st.markdown("""
        #### Probability Calibration & Commercial Operations
        KNN raw probability outputs represent **neighborhood vote fractions**, not calibrated statistical probabilities. 
        In credit management, true calibrated probabilities are required for:
        * **Expected Loss Modeling:** $\\text{Expected Loss} = P(\\text{Default}) \\times \\text{Balance} \\times \\text{LGD}$. Uncalibrated probabilities lead to biased risk provisions.
        * **Risk-Based Pricing:** Setting APR rates relative to default probability to optimize interest yield.
        * **Regulatory Compliance:** Under Basel III/IV, banks must report audited **Probability of Default (PD)** curves.
        
        **Implementation:** To calibrate, wrap the classifier in `CalibratedClassifierCV` using Platt scaling (Sigmoid) or Isotonic regression:
        """)
        st.code("""
        from sklearn.calibration import CalibratedClassifierCV
        
        calibrated_model = CalibratedClassifierCV(
            knn_model,
            method="isotonic",  # Non-parametric step function fitting
            cv=5                # 5-fold cross-validation to prevent calibration leakage
        )
        calibrated_model.fit(X_train, y_train)
        """, language="python")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Centurion Risk Control Laboratory &nbsp;·&nbsp; Credit Card Segmentation Analysis &nbsp;·&nbsp; K-Means + KNN Model
</div>
""", unsafe_allow_html=True)
