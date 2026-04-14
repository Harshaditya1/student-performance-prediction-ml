"""
pages/page_evaluation.py
─────────────────────────
Renders the "📐 Model Evaluation" page.

Displays:
  - Train / test split sizes
  - Per-model metrics: R², MAE, MSE, CV Mean R² (k=5)
  - Per-fold CV score table
  - Side-by-side model comparison table
  - Best model recommendation
  - Metric definitions explanation
"""

import pandas as pd
import streamlit as st


def render(X, y, results: dict, X_train, X_test):
    """
    Render the Model Evaluation page.

    Parameters
    ----------
    X        : np.ndarray  - Full feature matrix (for display)
    y        : np.ndarray  - Full target vector (for display)
    results  : dict        - Output of model_trainer.train_models()
    X_train  : np.ndarray  - Training features
    X_test   : np.ndarray  - Testing features
    """
    # ── Train / Test Split Info ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Train / Test Split (80 / 20)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples",    len(X))
    c2.metric("Training Samples", len(X_train))
    c3.metric("Testing Samples",  len(X_test))

    # ── Per-Model Metrics ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    for name, res in results.items():
        with st.expander(f"📌 {name}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score",        f"{res['r2']:.4f}")
            col2.metric("MAE",             f"{res['mae']:.4f}")
            col3.metric("MSE",             f"{res['mse']:.4f}")
            col4.metric("CV Mean R² (k=5)", f"{res['cv_mean']:.4f}")

            # CV fold breakdown
            cv_df = pd.DataFrame({
                "Fold":     [f"Fold {i+1}" for i in range(5)],
                "R² Score": [round(s, 4) for s in res["cv_scores"]],
            })
            st.table(cv_df.set_index("Fold").T)
            st.caption(f"CV Std Dev: {res['cv_std']:.4f}")

    # ── Side-by-Side Comparison Table ────────────────────────────────────────
    st.markdown('<div class="section-header">Side-by-Side Comparison</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Model":       list(results.keys()),
        "R² (test)":   [f"{r['r2']:.4f}"      for r in results.values()],
        "MAE":         [f"{r['mae']:.4f}"     for r in results.values()],
        "MSE":         [f"{r['mse']:.4f}"     for r in results.values()],
        "CV Mean R²":  [f"{r['cv_mean']:.4f}" for r in results.values()],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ── Best Model ────────────────────────────────────────────────────────────
    best = max(results.items(), key=lambda kv: kv[1]["cv_mean"])
    st.success(f"✅  **Best Model: {best[0]}** — CV Mean R² = **{best[1]['cv_mean']:.4f}**")

    # ── Metric Definitions ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Metric Definitions</div>', unsafe_allow_html=True)
    st.markdown("""
    | Metric | What it means |
    |---|---|
    | **R² Score** | How much variance the model explains (1.0 = perfect) |
    | **MAE** | Average prediction error in grade points |
    | **MSE** | Mean squared error — penalises larger mistakes more |
    | **CV Mean R²** | Average R² over 5 folds — more reliable than a single split |
    """)
