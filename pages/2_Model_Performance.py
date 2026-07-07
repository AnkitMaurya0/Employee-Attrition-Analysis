"""
pages/2_Model_Performance.py
─────────────────────────────────────────────────────────────────────────────
Model Performance Dashboard
  • KPI cards: Accuracy, Precision, Recall, F1, ROC-AUC
  • Interactive Confusion Matrix (Plotly heatmap)
  • Interactive ROC Curve
  • Feature Importance Chart (top 15)
  • Classification Report table
  • Plain-English metric explanations
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance | Employee Attrition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent
css_path = ROOT / "assets" / "style.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Plotly Dark Theme ─────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a", size=11),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# ─── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    """Load model metrics, feature importance, and regenerate test predictions."""
    results = pd.read_csv(ROOT / "results" / "model_results.csv")
    feat_imp = pd.read_csv(ROOT / "results" / "feature_importance.csv")
    return results, feat_imp

@st.cache_data
def get_test_predictions():
    """Regenerate test predictions for confusion matrix and ROC curve."""
    model = joblib.load(ROOT / "models" / "best_model.pkl")
    preprocessor = joblib.load(ROOT / "models" / "preprocessor.pkl")
    df = pd.read_csv(ROOT / "data" / "clean_data.csv")

    input_cols = [
        'Age', 'Gender', 'Department', 'JobRole', 'MonthlyIncome',
        'BusinessTravel', 'DistanceFromHome', 'Education', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'OverTime', 'YearsAtCompany',
        'YearsSinceLastPromotion', 'WorkLifeBalance', 'TrainingTimesLastYear'
    ]
    df_model = df[input_cols + ['Attrition']].copy()
    df_model['PromotionRatio'] = df_model['YearsSinceLastPromotion'] / (df_model['YearsAtCompany'] + 1)
    df_model['SatisfactionIndex'] = (
        df_model['EnvironmentSatisfaction'] + df_model['JobSatisfaction'] + df_model['WorkLifeBalance']
    ) / 3.0

    X = df_model.drop(columns=['Attrition'])
    y = df_model['Attrition'].map({'Yes': 1, 'No': 0})
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_proc = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]
    return y_test.values, y_pred, y_proba

results_df, feat_imp = load_results()
y_test, y_pred, y_proba = get_test_predictions()

# Parse metrics
metrics = dict(zip(results_df['Metric'], results_df['Value']))

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-wrap">
    <div class="brand-icon">📊</div>
    <div>
        <div class="brand-name">Model Performance</div>
        <div class="brand-tagline">Evaluation metrics, charts, and feature importance</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">🎯</div>
    <div class="title">Test Set Performance Metrics</div>
    <div class="subtitle">Evaluated on 20% held-out test set (294 employees)</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    ("Accuracy",  f"{metrics.get('Accuracy',0):.1%}",  "delta-up",   "↑ Overall correct predictions"),
    ("Precision", f"{metrics.get('Precision',0):.1%}",  "delta-warn",  "Of predicted Yes, truly Yes"),
    ("Recall",    f"{metrics.get('Recall',0):.1%}",     "delta-warn",  "Of actual Yes, captured"),
    ("F1 Score",  f"{metrics.get('F1 Score',0):.1%}",   "delta-warn",  "Harmonic mean of P & R"),
    ("ROC-AUC",   f"{metrics.get('ROC-AUC',0):.3f}",    "delta-info",  "Discrimination ability"),
]
for col, (label, value, badge, note) in zip([c1, c2, c3, c4, c5], kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size: 1.8rem;">{value}</div>
            <div class="metric-delta {badge}">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1: Confusion Matrix + ROC Curve ───────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">📉</div>
    <div class="title">Evaluation Charts</div>
</div>
""", unsafe_allow_html=True)

col_cm, col_roc = st.columns(2)

# Confusion Matrix
with col_cm:
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cm_labels = np.array([[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]])

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: Stay", "Predicted: Leave"],
        y=["Actual: Stay", "Actual: Leave"],
        text=cm_labels,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white", "family": "JetBrains Mono"},
        colorscale=[[0, "#111114"], [0.5, "#1d4ed8"], [1, "#2563eb"]],
        showscale=False,
        hoverongaps=False,
    ))
    fig_cm.update_layout(
        **PLOT_LAYOUT,
        height=350,
        xaxis=dict(tickfont=dict(size=11, color="#a1a1aa")),
        yaxis=dict(tickfont=dict(size=11, color="#a1a1aa")),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Actual vs Predicted classifications on the test set</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top: 0.5rem;">
        <p>
            ✅ <strong>True Positives (TP): {tp}</strong> employees correctly identified as likely to leave.<br>
            ⚠️ <strong>False Negatives (FN): {fn}</strong> at-risk employees missed by the model.<br>
            <strong>False Positives (FP): {fp}</strong> employees incorrectly flagged as at-risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ROC Curve
with col_roc:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Model (AUC = {auc_val:.3f})",
        line=dict(color="#2563eb", width=2.5),
        fill='tozeroy',
        fillcolor="rgba(37,99,235,0.08)"
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="#52525b", width=1.2, dash="dash")
    ))
    fig_roc.update_layout(
        **PLOT_LAYOUT,
        height=350,
        legend=dict(
            font=dict(color="#a1a1aa", size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.04)"),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">ROC Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">True Positive Rate vs False Positive Rate across thresholds</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top: 0.5rem;">
        <p>
            <strong>AUC = {auc_val:.3f}</strong> — The model is correct in distinguishing leavers from stayers
            <strong>{auc_val*100:.1f}%</strong> of the time. An AUC of 0.5 means random guessing.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Feature Importance Chart ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">🔍</div>
    <div class="title">Feature Importance</div>
    <div class="subtitle">Top 15 most impactful features as determined by the Random Forest</div>
</div>
""", unsafe_allow_html=True)

top15 = feat_imp.head(15).copy()

fig_fi = go.Figure(go.Bar(
    x=top15['Importance'][::-1].values,
    y=top15['Feature'][::-1].values,
    orientation='h',
    marker=dict(
        color=top15['Importance'][::-1].values,
        colorscale=[[0, "#1e3a6e"], [1, "#2563eb"]],
        showscale=False,
    )
))
fig_fi.update_layout(
    **PLOT_LAYOUT,
    height=440,
    xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.04)"),
    yaxis=dict(tickfont=dict(size=10, color="#a1a1aa")),
)

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">Top 15 Random Forest Feature Importances</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-subtitle">Based on mean decrease in impurity (Gini importance) across all trees</div>', unsafe_allow_html=True)
st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Metric Explanations ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">📚</div>
    <div class="title">Understanding the Metrics</div>
</div>
""", unsafe_allow_html=True)

e1, e2 = st.columns(2)
with e1:
    st.markdown("""
    <div class="rec-card" style="margin-bottom: 1rem;">
        <div class="rec-title">Accuracy</div>
        <div class="rec-item"><span class="bullet">›</span>
            The percentage of all predictions (both Stay and Leave) that were correct.
            <em>Misleading for imbalanced datasets</em> — a model that always predicts "Stay" would get 84% accuracy!
        </div>
    </div>
    <div class="rec-card" style="margin-bottom: 1rem;">
        <div class="rec-title">Precision</div>
        <div class="rec-item"><span class="bullet">›</span>
            Of all employees the model flagged as <em>likely to leave</em>, how many actually left?
            Low precision = many false alarms, wasting HR resources.
        </div>
    </div>
    <div class="rec-card">
        <div class="rec-title">F1 Score</div>
        <div class="rec-item"><span class="bullet">›</span>
            The harmonic mean of Precision and Recall. A balanced measure that penalises both false positives
            and false negatives. <em>Most useful metric for imbalanced classification.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

with e2:
    st.markdown("""
    <div class="rec-card" style="margin-bottom: 1rem;">
        <div class="rec-title">Recall (Sensitivity)</div>
        <div class="rec-item"><span class="bullet">›</span>
            Of all employees who <em>actually left</em>, how many did the model correctly identify?
            High recall = fewer at-risk employees are missed. <em>Critical for HR retention programs.</em>
        </div>
    </div>
    <div class="rec-card" style="margin-bottom: 1rem;">
        <div class="rec-title">ROC-AUC</div>
        <div class="rec-item"><span class="bullet">›</span>
            Area Under the ROC Curve. Measures how well the model <em>ranks</em> at-risk employees
            above stable ones, regardless of the decision threshold. AUC = 1.0 is perfect; 0.5 is random.
        </div>
    </div>
    <div class="rec-card">
        <div class="rec-title">Confusion Matrix</div>
        <div class="rec-item"><span class="bullet">›</span>
            A grid showing True Positives, True Negatives, False Positives, and False Negatives.
            Gives a clear picture of exactly where the model is succeeding and where it makes errors.
        </div>
    </div>
    """, unsafe_allow_html=True)
