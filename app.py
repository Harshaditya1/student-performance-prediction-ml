"""
app.py  ← Entry Point
──────────────────────
Streamlit application entry point.

This file ONLY handles:
  1. Page configuration
  2. CSS injection (via styles.py)
  3. Data loading   (via data_loader.py)
  4. Model training (via model_trainer.py)
  5. Sidebar navigation and routing to the correct page module

Business logic, UI components, and ML code live in separate modules:
  ┌─────────────────────────────────────────────────────────────┐
  │  config.py          → constants (columns, labels, paths)    │
  │  styles.py          → CSS injection                         │
  │  data_loader.py     → load CSV, label encoding, X/y split   │
  │  model_trainer.py   → train models, CV, metrics             │
  │  utils.py           → grade_label(), encode_input()         │
  │  pages/                                                     │
  │    page_overview.py    → Dataset Overview page              │
  │    page_evaluation.py  → Model Evaluation page              │
  │    page_prediction.py  → Make a Prediction page             │
  └─────────────────────────────────────────────────────────────┘
"""

import warnings
import streamlit as st

# ── Project modules ───────────────────────────────────────────────────────────
from config import DATASET_PATH
from styles import inject_css
from data_loader import load_and_preprocess
from model_trainer import train_models
from views import page_overview, page_evaluation, page_prediction

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Page Config  (must be the first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. Inject CSS
# ─────────────────────────────────────────────
inject_css()

# ─────────────────────────────────────────────
# 3. Page Header
# ─────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🎓 Student Performance Prediction System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">UCI Student Performance Dataset · Math Subject · '
    'Decision Tree & Linear Regression · K-Fold CV (k=5)</div>',
    unsafe_allow_html=True
)
st.divider()

# ─────────────────────────────────────────────
# 4. Load & Preprocess Data
# ─────────────────────────────────────────────
try:
    df, df_enc, X, y, encoders = load_and_preprocess(DATASET_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️ Dataset file `{DATASET_PATH}` not found. "
        "Please ensure it is in the same folder as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────
# 5. Train Models
# ─────────────────────────────────────────────
results, X_train, X_test, y_train, y_test = train_models(X, y)

# ─────────────────────────────────────────────
# 6. Sidebar Navigation
# ─────────────────────────────────────────────
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📋 Dataset Overview", "📐 Model Evaluation", "🔮 Make a Prediction"],
)
st.sidebar.divider()
st.sidebar.markdown("**Dataset:** UCI Student Performance")
st.sidebar.markdown("**Records:** 395 students")
st.sidebar.markdown("**Features:** 32 input features")
st.sidebar.markdown("**Target:** G3 – Final Grade (0–20)")
st.sidebar.markdown("**Models:** Decision Tree, Linear Regression")
st.sidebar.markdown("**Validation:** K-Fold CV (k=5)")

# ─────────────────────────────────────────────
# 7. Route to the Correct Page
# ─────────────────────────────────────────────
if page == "📋 Dataset Overview":
    page_overview.render(df, encoders)

elif page == "📐 Model Evaluation":
    page_evaluation.render(X, y, results, X_train, X_test)

elif page == "🔮 Make a Prediction":
    page_prediction.render(results, encoders)
