"""
data_loader.py
──────────────
Responsible for:
  1. Loading the raw CSV dataset
  2. Label-encoding all categorical columns
  3. Preparing X (features) and y (target) arrays

Usage:
    from data_loader import load_and_preprocess
    df, df_enc, X, y, encoders = load_and_preprocess("student-mat.csv")
"""

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from config import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL


@st.cache_data
def load_and_preprocess(filepath: str):
    """
    Load the dataset from a CSV file and preprocess it.

    Steps:
      - Read CSV (semicolon-separated)
      - Label-encode each categorical column
      - Build X (feature matrix) and y (target vector)

    Parameters
    ----------
    filepath : str
        Path to the dataset CSV file.

    Returns
    -------
    df       : pd.DataFrame  - Raw (unencoded) dataset
    df_enc   : pd.DataFrame  - Encoded dataset
    X        : np.ndarray    - Feature matrix
    y        : np.ndarray    - Target vector (G3)
    encoders : dict          - {col_name: fitted LabelEncoder}
    """
    # 1. Load raw data
    df = pd.read_csv(filepath, sep=";")

    # 2. Label-encode categorical columns
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 3. Split into features and target
    X = df_enc[FEATURE_COLS].values
    y = df_enc[TARGET_COL].values

    return df, df_enc, X, y, encoders
