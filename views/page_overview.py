"""
pages/page_overview.py
───────────────────────
Renders the "📋 Dataset Overview" page.

Displays:
  - Key dataset metrics (records, columns, missing values, avg G3)
  - Full column list with data types and sample values
  - Numeric summary statistics
  - Label encoding reference table
  - Raw sample records
  - Grade distribution bar chart
"""

import pandas as pd
import streamlit as st


def render(df: pd.DataFrame, encoders: dict):
    """
    Render the Dataset Overview page.

    Parameters
    ----------
    df       : pd.DataFrame  - Raw (unencoded) dataset
    encoders : dict          - {col_name: LabelEncoder} for the encoding table
    """
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    # ── Top Metrics ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",   df.shape[0])
    c2.metric("Total Columns",    df.shape[1])
    c3.metric("Missing Values",   int(df.isnull().sum().sum()))
    c4.metric("Target: G3 (avg)", f"{df['G3'].mean():.2f}")

    st.markdown(
        '<div class="info-box">✅ <b>No missing values</b> — the UCI dataset is pre-cleaned. '
        'Categorical columns are label-encoded for modelling.</div>',
        unsafe_allow_html=True
    )

    # ── Column Types & Sample Values ────────────────────────────────────────
    st.markdown('<div class="section-header">All Columns & Data Types</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column":       df.columns.tolist(),
        "Type":         [str(d) for d in df.dtypes],
        "Sample Value": [str(df[c].iloc[0]) for c in df.columns],
    })
    st.dataframe(dtype_df, width="stretch", hide_index=True)

    # ── Numeric Summary Statistics ───────────────────────────────────────────
    st.markdown('<div class="section-header">Numeric Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), width="stretch")

    # ── Label Encoding Reference ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Label Encoding Applied To</div>', unsafe_allow_html=True)
    enc_rows = []
    for col, le in encoders.items():
        enc_rows.append({
            "Column":          col,
            "Original Values": str(le.classes_.tolist()),
            "Encoded As":      str(list(range(len(le.classes_)))),
        })
    st.dataframe(pd.DataFrame(enc_rows), width="stretch", hide_index=True)

    # ── Sample Raw Records ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Records (raw)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch", hide_index=True)

    # ── Grade Distribution Chart ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Target Variable Distribution (G3)</div>', unsafe_allow_html=True)
    grade_counts = df["G3"].value_counts().sort_index().reset_index()
    grade_counts.columns = ["Grade (G3)", "Count"]
    st.bar_chart(grade_counts.set_index("Grade (G3)"))
