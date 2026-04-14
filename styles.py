"""
styles.py
─────────
Contains the inject_css() function that injects all custom CSS
into the Streamlit app via st.markdown().

To add or modify styles, edit only this file — no other files need changing.
"""

import streamlit as st


def inject_css():
    """Inject custom CSS styling into the Streamlit page."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Page Header ── */
        .main-title {
            font-size: 2rem; font-weight: 700;
            color: #1565c0; margin-bottom: 0.1rem;
        }
        .sub-title {
            font-size: 0.95rem; color: #666; margin-bottom: 1rem;
        }

        /* ── Section Headers ── */
        .section-header {
            font-size: 1.1rem; font-weight: 600; color: #1a237e;
            border-left: 4px solid #1565c0; padding-left: 10px;
            margin-top: 1.2rem; margin-bottom: 0.6rem;
        }

        /* ── Info / Warning Boxes ── */
        .info-box {
            background: #e3f2fd; border-left: 4px solid #1976d2;
            padding: 10px 14px; border-radius: 4px;
            font-size: 0.88rem; color: #1a237e; margin-bottom: 1rem;
        }
        .warn-box {
            background: #fff8e1; border-left: 4px solid #ffa000;
            padding: 10px 14px; border-radius: 4px;
            font-size: 0.88rem; color: #555; margin-bottom: 1rem;
        }

        /* ── Prediction Result Box ── */
        .prediction-box {
            border-radius: 10px; padding: 24px;
            text-align: center; margin-top: 20px;
        }
        .prediction-label { font-size: 1rem; color: #444; }
        .prediction-value { font-size: 3rem; font-weight: 800; margin: 4px 0; }

        /* ── Grade Badge ── */
        .grade-badge {
            display: inline-block; font-size: 1rem; font-weight: 600;
            padding: 4px 14px; border-radius: 20px; margin-top: 6px;
        }
    </style>
    """, unsafe_allow_html=True)
