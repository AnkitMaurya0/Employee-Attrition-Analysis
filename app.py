from pathlib import Path
import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# Load CSS
# ---------------------------------------------------

css_path = Path(__file__).parent / "assets" / "style.css"

with open(css_path, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
"""
<div class="brand-wrap">

<div class="brand-icon">🏢</div>

<div>

<div class="brand-name">
Employee Attrition Analysis
</div>

<div class="brand-tagline">
Predict Employee Attrition using Machine Learning
</div>

</div>

</div>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------
# Hero Section
# ---------------------------------------------------

st.title("Employee Attrition Prediction Dashboard")

st.write(
"""
This dashboard predicts whether an employee is likely to leave the company.

The prediction is made using a **Random Forest Machine Learning model**.

It also explains the prediction using **SHAP**, so HR teams can understand the important factors behind the result.
"""
)

st.divider()

# ---------------------------------------------------
# Dashboard Features
# ---------------------------------------------------

st.subheader("Dashboard Pages")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
<div class="metric-card">

<div style="font-size:40px;">👨‍💼</div>

<div class="metric-label">
Prediction
</div>

<p>

Enter employee information and predict whether the employee is likely to leave the company.

</p>

</div>
""",
unsafe_allow_html=True)

with col2:

    st.markdown("""
<div class="metric-card">

<div style="font-size:40px;">📊</div>

<div class="metric-label">
Model Performance
</div>

<p>

View Accuracy, Precision, Recall, ROC-AUC, Confusion Matrix and Feature Importance.

</p>

</div>
""",
unsafe_allow_html=True)

with col3:

    st.markdown("""
<div class="metric-card">

<div style="font-size:40px;">📖</div>

<div class="metric-label">
About
</div>

<p>

Learn about the dataset, machine learning model, project workflow and technologies used.

</p>

</div>
""",
unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# Quick Information
# ---------------------------------------------------

st.subheader("Quick Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Dataset",
        "IBM HR"
    )

with c2:
    st.metric(
        "Employees",
        "1470"
    )

with c3:
    st.metric(
        "Model",
        "Random Forest"
    )

with c4:
    st.metric(
        "Explainability",
        "SHAP"
    )

st.divider()

# ---------------------------------------------------
# Workflow
# ---------------------------------------------------

st.subheader("Project Workflow")

st.markdown(
"""
```text
Employee Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Random Forest
        ↓
Prediction
        ↓
SHAP Explainability
```
"""
)

st.divider()