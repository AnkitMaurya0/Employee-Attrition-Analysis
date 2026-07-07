"""
pages/1_Prediction.py
─────────────────────────────────────────────────────────────────────────────
Employee Attrition Prediction Page
  • 15 user-input fields
  • Predict button
  • Risk Gauge (Plotly)
  • Retention probability & confidence
  • HR Recommendations
  • SHAP Waterfall Explanation
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction | Employee Attrition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load CSS ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
css_path = ROOT / "assets" / "style.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Model Loading (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load and cache the trained Random Forest model and preprocessor."""
    model = joblib.load(ROOT / "models" / "best_model.pkl")
    preprocessor = joblib.load(ROOT / "models" / "preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# ─── Helper: SHAP Explanation ─────────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(_model):
    """Create and cache SHAP TreeExplainer."""
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

# ─── Feature Name Helper ──────────────────────────────────────────────────────
CATEGORICAL_COLS = ['Gender', 'Department', 'JobRole', 'BusinessTravel', 'OverTime']
NUMERICAL_COLS = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany',
    'YearsSinceLastPromotion', 'TrainingTimesLastYear', 'PromotionRatio',
    'SatisfactionIndex', 'Education', 'JobSatisfaction',
    'EnvironmentSatisfaction', 'WorkLifeBalance'
]

def get_feature_names():
    """Return encoded feature names from the fitted preprocessor."""
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS).tolist()
    return NUMERICAL_COLS + cat_names

# ─── HR Recommendation Logic ──────────────────────────────────────────────────
def get_recommendations(probability: float, inputs: dict) -> list[str]:
    """Generate contextual HR recommendations based on attrition risk."""
    recs = []
    if probability >= 0.6:
        recs.append("🔴 Conduct an urgent one-on-one retention conversation with this employee.")
    elif probability >= 0.4:
        recs.append("🟡 Schedule a career development discussion in the next 30 days.")
    else:
        recs.append("🟢 Employee appears stable. Continue standard engagement check-ins.")

    if inputs.get("OverTime") == "Yes":
        recs.append("⚡ Review workload — overtime is strongly correlated with burnout and attrition.")
    if inputs.get("MonthlyIncome", 10000) < 4000:
        recs.append("💰 Consider a compensation review — salary is below median and a key attrition driver.")
    if inputs.get("YearsSinceLastPromotion", 0) > 3:
        recs.append("📈 Discuss promotion pathways — stagnant career progression increases departure risk.")
    if inputs.get("JobSatisfaction", 4) <= 2:
        recs.append("😔 Low job satisfaction detected — identify pain points through an anonymous survey.")
    if inputs.get("WorkLifeBalance", 4) <= 2:
        recs.append("⚖️ Work-life balance concerns — explore flexible working arrangements.")
    return recs

# ─── Gauge Chart ──────────────────────────────────────────────────────────────
def draw_gauge(probability: float):
    """Draw a Plotly gauge chart showing attrition probability."""
    pct = round(probability * 100, 1)
    if pct >= 60:
        color = "#ef4444"
        label = "HIGH RISK"
    elif pct >= 35:
        color = "#f59e0b"
        label = "MEDIUM RISK"
    else:
        color = "#22c55e"
        label = "LOW RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "family": "JetBrains Mono", "color": color}},
        title={"text": f"<b>{label}</b>", "font": {"size": 14, "color": "#71717a", "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1e1e24",
                     "tickfont": {"color": "#52525b", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#111114",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "rgba(34,197,94,0.1)"},
                {"range": [35, 60], "color": "rgba(245,158,11,0.1)"},
                {"range": [60, 100],"color": "rgba(239,68,68,0.1)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": pct
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=280,
        font=dict(family="DM Sans, sans-serif")
    )
    return fig

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-wrap">
    <div class="brand-icon">🎯</div>
    <div>
        <div class="brand-name">Employee Attrition Prediction</div>
        <div class="brand-tagline">Enter employee details below and click Predict</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Input Form ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">📝</div>
    <div class="title">Employee Information</div>
    <div class="subtitle">15 input fields · All fields required</div>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", min_value=18, max_value=65, value=35, help="Employee's current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Employee gender")
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=25000, value=5000, step=100)

    with c2:
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance_from_home = st.slider("Distance From Home (km)", min_value=1, max_value=29, value=9)
        education = st.select_slider("Education Level",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {1:"Below College", 2:"College", 3:"Bachelor", 4:"Master", 5:"Doctor"}[x]
        )
        job_satisfaction = st.select_slider("Job Satisfaction",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x]
        )
        env_satisfaction = st.select_slider("Environment Satisfaction",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x]
        )

    with c3:
        overtime = st.selectbox("OverTime", ["No", "Yes"], help="Does employee work overtime?")
        years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=5)
        years_since_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        work_life_balance = st.select_slider("Work Life Balance",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1:"Bad", 2:"Good", 3:"Better", 4:"Best"}[x]
        )
        training_times = st.slider("Training Times Last Year", min_value=0, max_value=6, value=3)

    submitted = st.form_submit_button("🔍 Predict Attrition Risk", use_container_width=True, type="primary")

# ─── Prediction Logic ─────────────────────────────────────────────────────────
if submitted:
    # Build raw input dict (matching model training features)
    promotion_ratio = years_since_promotion / (years_at_company + 1)
    satisfaction_index = (env_satisfaction + job_satisfaction + work_life_balance) / 3.0

    input_dict = {
        'Age': age,
        'Gender': gender,
        'Department': department,
        'JobRole': job_role,
        'MonthlyIncome': monthly_income,
        'BusinessTravel': business_travel,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'JobSatisfaction': job_satisfaction,
        'EnvironmentSatisfaction': env_satisfaction,
        'OverTime': overtime,
        'YearsAtCompany': years_at_company,
        'YearsSinceLastPromotion': years_since_promotion,
        'WorkLifeBalance': work_life_balance,
        'TrainingTimesLastYear': training_times,
        'PromotionRatio': promotion_ratio,
        'SatisfactionIndex': satisfaction_index,
    }

    # Create DataFrame and preprocess
    input_df = pd.DataFrame([input_dict])
    input_processed = preprocessor.transform(input_df)

    # Get prediction & probabilities
    prediction = model.predict(input_processed)[0]
    probabilities = model.predict_proba(input_processed)[0]
    attrition_prob = probabilities[1]       # P(Attrition = Yes)
    retention_prob = probabilities[0]       # P(Attrition = No)
    confidence = max(attrition_prob, retention_prob)

    # ─── Results Layout ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="icon">🔮</div>
        <div class="title">Prediction Results</div>
    </div>
    """, unsafe_allow_html=True)

    left, mid, right = st.columns([1.2, 1, 1])

    # Gauge
    with left:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Attrition Risk Meter</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Predicted probability of leaving the company</div>', unsafe_allow_html=True)
        st.plotly_chart(draw_gauge(attrition_prob), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary Cards
    with mid:
        risk_class = "risk-high" if attrition_prob >= 0.6 else ("risk-medium" if attrition_prob >= 0.35 else "risk-low")
        risk_label = "🔴 HIGH RISK" if attrition_prob >= 0.6 else ("🟡 MEDIUM RISK" if attrition_prob >= 0.35 else "🟢 LOW RISK")

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-label">Employee Attrition Risk</div>
            <div class="metric-value {risk_class}">{round(attrition_prob*100,1)}%</div>
            <div class="metric-delta {'delta-down' if attrition_prob >= 0.6 else 'delta-warn' if attrition_prob >= 0.35 else 'delta-up'}">{risk_label}</div>
        </div>
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-label">Retention Probability</div>
            <div class="metric-value" style="color: #22c55e; font-size: 1.8rem;">{round(retention_prob*100,1)}%</div>
            <div class="metric-delta delta-up">↑ Likely to Stay</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value" style="font-size: 1.8rem;">{round(confidence*100,1)}%</div>
            <div class="metric-delta delta-info">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    # HR Recommendations
    with right:
        recommendations = get_recommendations(attrition_prob, {
            "OverTime": overtime,
            "MonthlyIncome": monthly_income,
            "YearsSinceLastPromotion": years_since_promotion,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance
        })
        recs_html = "".join([f'<div class="rec-item"><span class="bullet">›</span>{r}</div>' for r in recommendations])
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">💼 HR Recommendations</div>
            {recs_html}
        </div>
        """, unsafe_allow_html=True)

    # ─── SHAP Explanation ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="icon">🧠</div>
        <div class="title">SHAP Explanation — Why this prediction?</div>
        <div class="subtitle">Top factors influencing this employee's attrition risk</div>
    </div>
    """, unsafe_allow_html=True)

    # Compute SHAP values for this individual
    shap_vals = explainer(input_processed)
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    feature_names = get_feature_names()
    sv = shap_vals.values[0]
    base_val = shap_vals.base_values[0] if hasattr(shap_vals, 'base_values') else 0.5

    # Sort by absolute importance, take top 10
    sorted_idx = np.argsort(np.abs(sv))[::-1][:10]
    top_features = [feature_names[i] for i in sorted_idx]
    top_values = [sv[i] for i in sorted_idx]

    # Build a horizontal bar chart
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_values]
    hover = [f"SHAP = {v:+.4f} → {'↑ increases' if v>0 else '↓ decreases'} attrition risk" for v in top_values]

    fig_shap = go.Figure(go.Bar(
        x=top_values[::-1],
        y=top_features[::-1],
        orientation='h',
        marker_color=colors[::-1],
        hovertext=hover[::-1],
        hoverinfo='text'
    ))
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#71717a", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        height=360,
        xaxis=dict(
            title="SHAP Value (Impact on Attrition Probability)",
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.15)",
            tickfont=dict(size=10, color="#71717a"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=10, color="#a1a1aa"),
        )
    )

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Top 10 SHAP Feature Contributions</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Red = increases attrition risk &nbsp;|&nbsp; Green = decreases attrition risk</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with scol2:
        st.markdown('<div class="rec-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<div class="rec-title">📌 How to Read SHAP</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rec-item"><span class="bullet">›</span><strong style="color:#fafafa;">SHAP values</strong> show how much each feature <em>pushes</em> the prediction toward or away from attrition.</div>
        <div class="rec-item"><span class="bullet">›</span><strong style="color:#ef4444;">Red bars</strong> (positive SHAP) = this feature <em>increases</em> the predicted attrition risk.</div>
        <div class="rec-item"><span class="bullet">›</span><strong style="color:#22c55e;">Green bars</strong> (negative SHAP) = this feature <em>decreases</em> the predicted risk.</div>
        <div class="rec-item"><span class="bullet">›</span>Longer bars = stronger impact on the final prediction.</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Info message
    badge = "danger" if prediction == 1 else "success"
    outcome = "⚠️ This employee is <strong>likely to leave</strong> — act promptly." if prediction == 1 else "✅ This employee is <strong>likely to stay</strong> — maintain current engagement."
    st.markdown(f"""
    <div class="info-box {badge}" style="margin-top: 1rem;">
        <p>{outcome}</p>
    </div>
    """, unsafe_allow_html=True)
