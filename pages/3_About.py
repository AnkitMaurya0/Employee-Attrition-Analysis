"""
pages/3_About.py
─────────────────────────────────────────────────────────────────────────────
About Page
  • Project overview
  • Dataset description
  • ML Pipeline summary
  • Technologies used
  • Project workflow
  • Developer information
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="About | Employee Attrition",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent
css_path = ROOT / "assets" / "style.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-wrap">
    <div class="brand-icon">📖</div>
    <div>
        <div class="brand-name">About This Project</div>
        <div class="brand-tagline">Employee Attrition Analysis · End-to-End ML Project</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Project Overview ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">🎯</div>
    <div class="title">Project Overview</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>
        <strong>Employee Attrition Analysis</strong> is a complete end-to-end Machine Learning project
        that predicts whether an employee is at risk of leaving a company. Beyond prediction, it provides
        <strong>SHAP-based explanations</strong> so HR professionals understand <em>why</em> a prediction
        was made — enabling data-driven retention strategies.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="rec-card">
        <div class="rec-title">🎯 Objective</div>
        <div class="rec-item"><span class="bullet">›</span>Predict employee attrition using supervised ML classification.</div>
        <div class="rec-item"><span class="bullet">›</span>Explain predictions using SHAP (SHapley Additive exPlanations).</div>
        <div class="rec-item"><span class="bullet">›</span>Enable HR teams to proactively identify and retain at-risk talent.</div>
        <div class="rec-item"><span class="bullet">›</span>Present findings in an interactive, professional Streamlit dashboard.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="rec-card">
        <div class="rec-title">✨ Key Features</div>
        <div class="rec-item"><span class="bullet">›</span>Interactive Prediction Page with 15 employee input fields.</div>
        <div class="rec-item"><span class="bullet">›</span>Real-time Attrition Risk Gauge with visual probability display.</div>
        <div class="rec-item"><span class="bullet">›</span>Personalised SHAP Waterfall explanations for each prediction.</div>
        <div class="rec-item"><span class="bullet">›</span>Contextual HR Recommendations based on risk factors.</div>
        <div class="rec-item"><span class="bullet">›</span>Full Model Performance evaluation dashboard.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">🗃️</div>
    <div class="title">Dataset</div>
</div>
""", unsafe_allow_html=True)

d1, d2, d3, d4 = st.columns(4)
for col, (label, value) in zip([d1, d2, d3, d4], [
    ("Source", "IBM HR Analytics"),
    ("Employees", "1,470 rows"),
    ("Features", "35 original"),
    ("Attrition Rate", "16.1%"),
]):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size: 1.3rem; color: #2563eb;">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="rec-card">
    <div class="rec-title">About the Dataset</div>
    <div class="rec-item"><span class="bullet">›</span>
        The <strong>IBM HR Analytics Employee Attrition Dataset</strong> is a fictional dataset created by IBM data scientists, 
        widely used on Kaggle for HR analytics and ML classification practice.
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        It contains <strong>35 attributes</strong> covering demographics (Age, Gender, Marital Status), 
        job details (Department, Job Role, Job Level), compensation (Monthly Income, Stock Options), 
        satisfaction scores (Job Satisfaction, Environment Satisfaction), and tenure metrics.
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        The target variable <strong>Attrition</strong> is binary (Yes / No), with a notable 
        class imbalance of ~84% No and ~16% Yes.
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        The dataset has <strong>zero missing values</strong> and <strong>zero duplicate rows</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── ML Pipeline ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">⚙️</div>
    <div class="title">Machine Learning Pipeline</div>
</div>
""", unsafe_allow_html=True)

steps = [
    ("1", "📥 Data Loading", "Load raw CSV dataset with UTF-8 BOM handling. Validate shape, types, missing values, and duplicates."),
    ("2", "🔍 EDA", "Univariate and bivariate analysis using Plotly charts. Identify attrition drivers: Income, Age, Overtime, Promotion gaps."),
    ("3", "🧹 Data Cleaning", "Drop constant/non-informative columns: `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`."),
    ("4", "⚗️ Feature Engineering", "Create `PromotionRatio`, `SatisfactionIndex`, and `YearsPerCompany` to capture combined signals."),
    ("5", "🔢 Encoding & Preprocessing", "One-Hot Encoding for nominal categoricals. Standard Scaling for numerical features. Fit ONLY on training data."),
    ("6", "✂️ Train/Test Split", "Stratified 80/20 split to preserve class imbalance ratio in both sets. Split happens BEFORE preprocessing."),
    ("7", "🌲 Random Forest", "Train `RandomForestClassifier` with `class_weight='balanced_subsample'` to handle class imbalance."),
    ("8", "🔧 Hyperparameter Tuning", "`RandomizedSearchCV` with 20 iterations, 5-fold CV, optimising for F1 score."),
    ("9", "📊 Evaluation", "Compute Accuracy, Precision, Recall, F1, ROC-AUC. Plot Confusion Matrix, ROC Curve, and Feature Importance."),
    ("10","🧠 SHAP", "TreeExplainer generates global (summary/bar) and local (waterfall) SHAP explanations for model interpretability."),
]

for step_num, title, desc in steps:
    st.markdown(f"""
    <div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.7rem;">
        <div style="min-width: 32px; height: 32px; background: rgba(37,99,235,0.15); border: 1px solid #2563eb; 
                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                    font-size: 0.75rem; font-weight: 700; color: #2563eb; flex-shrink: 0;">{step_num}</div>
        <div>
            <div style="font-size: 0.87rem; font-weight: 600; color: #fafafa; margin-bottom: 0.2rem;">{title}</div>
            <div style="font-size: 0.8rem; color: #71717a; line-height: 1.5;">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Technologies ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">🛠️</div>
    <div class="title">Technologies Used</div>
</div>
""", unsafe_allow_html=True)

tech_cols = st.columns(4)
techs = [
    ("🐍 Python 3.14", "Core language", "badge-blue"),
    ("🐼 Pandas", "Data manipulation", "badge-green"),
    ("🔢 NumPy", "Numerical computing", "badge-blue"),
    ("🤖 Scikit-learn", "ML pipeline, RF, metrics", "badge-purple"),
    ("📈 Plotly", "Interactive charts", "badge-amber"),
    ("🖥️ Streamlit", "Dashboard framework", "badge-red"),
    ("🧠 SHAP", "Model explainability", "badge-green"),
    ("💾 Joblib", "Model serialization", "badge-blue"),
]
for col, (name, role, badge) in zip(tech_cols * 2, techs):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.8rem; padding: 1rem;">
            <div style="font-size: 1.1rem; margin-bottom: 0.3rem;">{name.split()[0]}</div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #fafafa;">{name.split(' ', 1)[1] if ' ' in name else name}</div>
            <div class="badge {badge}" style="margin-top: 0.4rem;">{role}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Project Structure ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">📁</div>
    <div class="title">Project Structure</div>
</div>
""", unsafe_allow_html=True)

st.code("""
Employee-Attrition-Analysis/
│
├── app.py                    ← Main landing page
├── requirements.txt          ← Python dependencies
├── README.md                 ← Project documentation
│
├── notebook/
│   ├── 01_EDA.ipynb          ← Exploratory Data Analysis
│   └── 02_Model.ipynb        ← Model Training & Evaluation
│
├── pages/
│   ├── 1_Prediction.py       ← Employee risk prediction form
│   ├── 2_Model_Performance.py ← Model metrics & charts
│   └── 3_About.py            ← Project info
│
├── models/
│   ├── best_model.pkl        ← Trained Random Forest model
│   └── preprocessor.pkl      ← Fitted preprocessing pipeline
│
├── assets/
│   └── style.css             ← Custom dark theme CSS
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← Raw dataset
│   └── clean_data.csv        ← Cleaned & engineered dataset
│
└── results/
    ├── model_results.csv      ← Evaluation metrics
    ├── feature_importance.csv ← Feature importance scores
    └── shap_*.png            ← SHAP plot images
""", language="text")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Developer Info ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon">👨‍💻</div>
    <div class="title">Developer Information</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rec-card">
    <div class="rec-title">Project Details</div>
    <div class="rec-item"><span class="bullet">›</span>
        <strong style="color: #fafafa;">Project Type:</strong>&nbsp; End-to-End Machine Learning · Interview-Ready · GitHub-Ready · Deployment-Ready
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        <strong style="color: #fafafa;">Stack:</strong>&nbsp; Python · Scikit-learn · SHAP · Plotly · Streamlit · Joblib
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        <strong style="color: #fafafa;">Dataset:</strong>&nbsp; IBM HR Analytics Employee Attrition Dataset (Kaggle)
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        <strong style="color: #fafafa;">Model:</strong>&nbsp; Random Forest Classifier with RandomizedSearchCV tuning and SHAP explainability
    </div>
    <div class="rec-item"><span class="bullet">›</span>
        <strong style="color: #fafafa;">License:</strong>&nbsp; MIT
    </div>
</div>
""", unsafe_allow_html=True)
