# Employee Attrition Analysis 🏢

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-0.52-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Predict & Explain Employee Attrition using Random Forest + SHAP**

*A professional, interview-ready, end-to-end Machine Learning project with an interactive Streamlit dashboard.*

**By [Ankit Maurya](https://github.com/AnkitMaurya0)**

### 🚀 [**Live Demo →**](https://employee-attrition-analysis-m.streamlit.app/)

</div>

---

## 📋 Project Description

**Employee Attrition Analysis** is a complete Machine Learning project that uses the IBM HR Analytics dataset to:

1. **Predict** whether an employee is likely to leave the company.
2. **Explain** *why* using SHAP (SHapley Additive exPlanations).
3. **Present** results through a modern, dark-themed Streamlit dashboard.

The dashboard gives HR teams actionable insights — not just predictions.

🔗 **Try it live:** [employee-attrition-analysis-m.streamlit.app](https://employee-attrition-analysis-m.streamlit.app/)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 Prediction Page | Enter 15 employee attributes and get instant attrition probability |
| 📊 Risk Gauge | Plotly gauge chart showing Low / Medium / High risk |
| 🧠 SHAP Explanation | Per-employee SHAP waterfall chart showing top risk factors |
| 💼 HR Recommendations | Contextual, actionable retention suggestions |
| 📈 Model Performance | Full evaluation dashboard with KPIs, confusion matrix, ROC curve |
| 🔍 Feature Importance | Top 15 most influential features visualised |
| 🌙 Dark Theme | Professional dark UI with DM Sans typography |

---

## 🔄 Machine Learning Workflow

```
Raw Dataset (IBM HR Analytics)
        │
        ▼
   Data Loading & Validation
   (Missing values, Duplicates, BOM handling)
        │
        ▼
   Exploratory Data Analysis (EDA)
   (Univariate, Bivariate, Correlation)
        │
        ▼
   Data Cleaning
   (Drop constants: EmployeeCount, Over18, StandardHours, EmployeeNumber)
        │
        ▼
   Feature Engineering
   (SatisfactionIndex, PromotionRatio, YearsPerCompany)
        │
        ▼
   Train / Test Split (80/20 Stratified)
        │
        ▼
   Preprocessing Pipeline
   (StandardScaler + OneHotEncoder — fit on train only)
        │
        ▼
   Random Forest Classifier
        │
        ▼
   Hyperparameter Tuning (RandomizedSearchCV, 5-fold CV, F1 optimised)
        │
        ▼
   Evaluation
   (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
        │
        ▼
   SHAP Analysis
   (Global: Summary + Bar | Local: Waterfall per employee)
        │
        ▼
   Streamlit Dashboard
```

---

## 📂 Folder Structure

```
Employee-Attrition-Analysis/
│
├── app.py                    ← Main Streamlit landing page
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
│   └── 3_About.py            ← Project info (this page)
│
├── models/
│   ├── best_model.pkl        ← Trained Random Forest model
│   └── preprocessor.pkl      ← Fitted preprocessing pipeline
│
├── assets/
│   └── style.css             ← Custom dark theme CSS
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← Raw IBM dataset
│   └── clean_data.csv        ← Cleaned & engineered dataset
│
└── results/
    ├── model_results.csv      ← Evaluation metrics
    ├── feature_importance.csv ← RF feature importances
    ├── roc_curve.png          ← ROC Curve image
    ├── feature_importance.png ← Feature importance chart
    ├── shap_summary.png       ← SHAP summary plot
    ├── shap_bar.png           ← SHAP bar plot
    └── shap_waterfall_sample.png ← Sample waterfall plot
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | IBM HR Analytics Employee Attrition |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Rows** | 1,470 employees |
| **Columns** | 35 features |
| **Target** | `Attrition` (Yes = 16.1%, No = 83.9%) |
| **Missing Values** | 0 |
| **Duplicates** | 0 |

---

## 🤖 Model

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Tuning** | RandomizedSearchCV (20 iterations, 5-fold CV) |
| **Optimization Metric** | F1 Score |
| **Class Imbalance Handling** | `class_weight='balanced_subsample'` |
| **Best Parameters** | n_estimators=300, max_depth=5, class_weight=balanced_subsample |

### Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | 80.6% |
| Precision | 41.1% |
| Recall | 48.9% |
| F1 Score | 44.7% |
| **ROC-AUC** | **77.4%** |

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) is used at two levels:

- **Global**: Summary plot and bar plot showing which features matter most across all employees.
- **Local**: Per-employee waterfall chart showing exactly why a specific prediction was made.

Top attrition drivers identified by SHAP:
1. `MonthlyIncome` — Lower income → higher attrition risk
2. `OverTime` — Working overtime strongly increases risk
3. `Age` — Younger employees have higher attrition rates
4. `PromotionRatio` — Longer since last promotion → higher risk
5. `SatisfactionIndex` — Combined satisfaction score

---

## 🌐 Live Demo

The dashboard is deployed and publicly accessible on Streamlit Community Cloud:

**👉 [https://employee-attrition-analysis-m.streamlit.app/](https://employee-attrition-analysis-m.streamlit.app/)**

No installation required — open the link and start exploring predictions, model performance, and SHAP explanations directly in your browser.

---

## ⚙️ Installation (Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/AnkitMaurya0/Employee-Attrition-Analysis.git
cd Employee-Attrition-Analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place it in the `data/` folder.

### 5. Run the ML training pipeline

```bash
# Step 1: Clean data and feature engineering
python notebook/run_eda.py

# Step 2: Train model, evaluate, and generate SHAP plots
python notebook/run_model.py
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

---

## 🚀 Deployment

The app is deployed on **Streamlit Community Cloud**:

🔗 **Live App:** [https://employee-attrition-analysis-m.streamlit.app/](https://employee-attrition-analysis-m.streamlit.app/)

### Deploy your own copy:

1. Push the repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo and set `app.py` as the main file.
4. Click **Deploy**.

> **Note**: Make sure `models/best_model.pkl`, `models/preprocessor.pkl`, and `results/*.csv` are committed to the repository before deploying.

---

## 🔮 Future Improvements

- [ ] **Threshold Tuning**: Lower decision threshold from 0.5 → 0.35 to boost Recall for HR use cases.
- [ ] **SMOTE Oversampling**: Address class imbalance at the data level.
- [ ] **Additional Models**: Compare Logistic Regression, XGBoost, and LightGBM against Random Forest.
- [ ] **Cross-Validation Dashboard**: Show 5-fold CV metrics alongside test set metrics.
- [ ] **Department-level Analysis**: Segment predictions by department for targeted HR strategy.
- [ ] **Export Reports**: Allow HR to download a PDF report for individual employees.
- [ ] **Live Data Integration**: Connect to an HR system API for real-time predictions.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Ankit Maurya**

Built as a professional, end-to-end ML portfolio project.

- **GitHub**: [@AnkitMaurya0](https://github.com/AnkitMaurya0)
- **Repository**: [AnkitMaurya0/Employee-Attrition-Analysis](https://github.com/AnkitMaurya0/Employee-Attrition-Analysis)
- **Dataset**: IBM HR Analytics (Kaggle)
- **Stack**: Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit · SHAP · Joblib
- **Type**: Classification · Explainable AI · HR Analytics
- **Live Demo**: [employee-attrition-analysis-m.streamlit.app](https://employee-attrition-analysis-m.streamlit.app/)

---

<div align="center">

### 🌐 [View Live App](https://employee-attrition-analysis-m.streamlit.app/)

⭐ Star this repo if you found it helpful!
</div>
