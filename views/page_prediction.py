"""
pages/page_prediction.py
─────────────────────────
Renders the "🔮 Make a Prediction" page.

Provides:
  - An input form for all 32 student features
  - Model selector (Decision Tree / Linear Regression)
  - Prediction result with grade category and colour coding
  - Input summary table
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import COL_LABELS
from utils import grade_label, encode_input


def render(results: dict, encoders: dict):
    """
    Render the Prediction page.

    Parameters
    ----------
    results  : dict  - Output of model_trainer.train_models()
    encoders : dict  - {col_name: LabelEncoder} from data_loader
    """
    st.markdown('<div class="section-header">Enter Student Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Fill in the student\'s details below and click '
        '<b>Predict Final Grade</b> to see the predicted G3 score (0–20).</div>',
        unsafe_allow_html=True
    )

    # ════════════════════════════════════════════════════════════════════════
    # Input Form
    # ════════════════════════════════════════════════════════════════════════
    with st.form("prediction_form"):

        # ── Section 1: Personal & Family ────────────────────────────────────
        st.markdown("**📌 Personal & Family Information**")

        r1c1, r1c2, r1c3 = st.columns(3)
        school  = r1c1.selectbox("School",        ["GP", "MS"])
        sex     = r1c2.selectbox("Sex",           ["F", "M"])
        age     = r1c3.slider("Age", 15, 22, 17)

        r2c1, r2c2, r2c3 = st.columns(3)
        address = r2c1.selectbox("Address",       ["U", "R"])
        famsize = r2c2.selectbox("Family Size",   ["GT3", "LE3"])
        Pstatus = r2c3.selectbox("Parent Status", ["T", "A"])

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        Medu = r3c1.slider("Mother Education (0–4)", 0, 4, 2)
        Fedu = r3c2.slider("Father Education (0–4)", 0, 4, 2)
        Mjob = r3c3.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = r3c4.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])

        st.markdown("---")

        # ── Section 2: School & Study ────────────────────────────────────────
        st.markdown("**📚 School & Study Information**")

        r4c1, r4c2, r4c3 = st.columns(3)
        reason     = r4c1.selectbox("Reason for School", ["course", "home", "reputation", "other"])
        guardian   = r4c2.selectbox("Guardian",          ["mother", "father", "other"])
        traveltime = r4c3.slider("Travel Time (1–4)", 1, 4, 1)

        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        studytime = r5c1.slider("Study Time (1–4)", 1, 4, 2)
        failures  = r5c2.slider("Past Failures (0–3)", 0, 3, 0)
        absences  = r5c3.slider("Absences (0–75)", 0, 75, 4)
        schoolsup = r5c4.selectbox("Extra School Support", ["yes", "no"])

        r6c1, r6c2, r6c3, r6c4 = st.columns(4)
        famsup     = r6c1.selectbox("Family Support", ["yes", "no"])
        paid       = r6c2.selectbox("Paid Classes",   ["yes", "no"])
        activities = r6c3.selectbox("Activities",     ["yes", "no"])
        nursery    = r6c4.selectbox("Nursery",        ["yes", "no"])

        r7c1, r7c2, r7c3 = st.columns(3)
        higher   = r7c1.selectbox("Wants Higher Edu",        ["yes", "no"])
        internet = r7c2.selectbox("Internet at Home",        ["yes", "no"])
        romantic = r7c3.selectbox("Romantic Relationship",   ["no",  "yes"])

        st.markdown("---")

        # ── Section 3: Lifestyle & Social ───────────────────────────────────
        st.markdown("**🎭 Lifestyle & Social**")

        r8c1, r8c2, r8c3, r8c4, r8c5 = st.columns(5)
        famrel   = r8c1.slider("Family Rel. (1–5)",       1, 5, 4)
        freetime = r8c2.slider("Free Time (1–5)",         1, 5, 3)
        goout    = r8c3.slider("Going Out (1–5)",         1, 5, 3)
        Dalc     = r8c4.slider("Workday Alcohol (1–5)",   1, 5, 1)
        Walc     = r8c5.slider("Weekend Alcohol (1–5)",   1, 5, 1)

        r9c1, _ = st.columns(2)
        health = r9c1.slider("Health (1–5)", 1, 5, 3)

        st.markdown("---")

        # ── Section 4: Previous Grades ───────────────────────────────────────
        st.markdown("**📊 Previous Grades**")

        rGc1, rGc2 = st.columns(2)
        G1 = rGc1.slider("First Period Grade – G1 (0–20)",  0, 20, 10)
        G2 = rGc2.slider("Second Period Grade – G2 (0–20)", 0, 20, 10)

        # ── Model Selector & Submit ──────────────────────────────────────────
        model_choice = st.selectbox("🧠 Choose Primary Prediction Model", list(results.keys()))
        submitted = st.form_submit_button(
            "🔮 Predict Final Grade", type="primary", width="stretch"
        )

    # ════════════════════════════════════════════════════════════════════════
    # Prediction Result (shown only after form submit)
    # ════════════════════════════════════════════════════════════════════════
    if submitted:
        # Build the feature vector in the same column order as training
        input_values = [
            encode_input(encoders, "school",    school),
            encode_input(encoders, "sex",       sex),
            age,
            encode_input(encoders, "address",   address),
            encode_input(encoders, "famsize",   famsize),
            encode_input(encoders, "Pstatus",   Pstatus),
            Medu, Fedu,
            encode_input(encoders, "Mjob",      Mjob),
            encode_input(encoders, "Fjob",      Fjob),
            encode_input(encoders, "reason",    reason),
            encode_input(encoders, "guardian",  guardian),
            traveltime, studytime, failures,
            encode_input(encoders, "schoolsup", schoolsup),
            encode_input(encoders, "famsup",    famsup),
            encode_input(encoders, "paid",      paid),
            encode_input(encoders, "activities",activities),
            encode_input(encoders, "nursery",   nursery),
            encode_input(encoders, "higher",    higher),
            encode_input(encoders, "internet",  internet),
            encode_input(encoders, "romantic",  romantic),
            famrel, freetime, goout, Dalc, Walc, health, absences,
            G1, G2,
        ]

        input_arr = np.array([input_values])
        predictions = {
            name: max(0.0, min(20.0, round(res["model"].predict(input_arr)[0], 2)))
            for name, res in results.items()
        }

        best_model = max(results.items(), key=lambda kv: kv[1]["cv_mean"])[0]
        selected_pred = predictions[model_choice]
        selected_cv   = results[model_choice]["cv_mean"]
        selected_label, selected_text_col, selected_bg_col = grade_label(selected_pred)

        # ── Selected Model Prediction Box ────────────────────────────────────
        st.markdown(f"""
        <div class="prediction-box" style="background:{selected_bg_col}; border: 2px solid {selected_text_col};">
            <div class="prediction-label">Predicted Final Grade G3 &nbsp;|&nbsp; {model_choice}</div>
            <div class="prediction-value" style="color:{selected_text_col};">{selected_pred:.1f} / 20</div>
            <span class="grade-badge" style="background:{selected_text_col}; color:#fff;">{selected_label}</span>
            <div style="font-size:0.82rem; color:#555; margin-top:10px;">
                Model CV Mean R² (k=5): <b>{selected_cv:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Comparative Output Table ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Comparative Prediction Analysis")
        comparison_df = pd.DataFrame({
            "Model": list(predictions.keys()),
            "Predicted G3": [f"{pred:.2f}" for pred in predictions.values()],
            "Grade Category": [grade_label(pred)[0] for pred in predictions.values()],
            "CV Mean R²": [f"{results[name]['cv_mean']:.4f}" for name in predictions.keys()],
        })
        st.dataframe(comparison_df, width="stretch", hide_index=True)

        dt_pred = predictions.get("Decision Tree")
        lr_pred = predictions.get("Linear Regression")
        if dt_pred is not None and lr_pred is not None:
            diff = dt_pred - lr_pred
            if diff > 0:
                diff_text = f"Decision Tree predicts {diff:.2f} points higher than Linear Regression."
            elif diff < 0:
                diff_text = f"Linear Regression predicts {abs(diff):.2f} points higher than Decision Tree."
            else:
                diff_text = "Both models predict the same final grade for this input."

            st.markdown("#### Comparison Summary")
            st.write(f"- Primary selected model: **{model_choice}**")
            st.write(f"- Current best model by cross-validation: **{best_model}**")
            st.write(f"- {diff_text}")

            chart_df = pd.DataFrame({
                "Predicted G3": [dt_pred, lr_pred],
            }, index=["Decision Tree", "Linear Regression"])
            st.bar_chart(chart_df)

        # ── Input Summary Table ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**📋 Input Summary**")
        raw_values = [
            school, sex, age, address, famsize, Pstatus,
            Medu, Fedu, Mjob, Fjob, reason, guardian,
            traveltime, studytime, failures, schoolsup,
            famsup, paid, activities, nursery, higher,
            internet, romantic, famrel, freetime, goout,
            Dalc, Walc, health, absences, G1, G2,
        ]
        summary_df = pd.DataFrame({
            "Feature": list(COL_LABELS.values()),
            "Value":   [str(value) for value in raw_values],
        })
        st.dataframe(summary_df, width="stretch", hide_index=True)
