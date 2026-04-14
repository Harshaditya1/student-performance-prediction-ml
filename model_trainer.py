"""
model_trainer.py
────────────────
Responsible for:
  1. Splitting data into train / test sets
  2. Training Decision Tree and Linear Regression models
  3. Running K-Fold Cross Validation (cv=5)
  4. Computing evaluation metrics: R², MAE, MSE

Usage:
    from model_trainer import train_models
    results, X_train, X_test, y_train, y_test = train_models(X, y)
"""

import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@st.cache_resource
def train_models(X, y):
    """
    Train both models, evaluate on test set, and run K-Fold CV.

    Parameters
    ----------
    X : np.ndarray  - Feature matrix
    y : np.ndarray  - Target vector

    Returns
    -------
    results  : dict  - Per-model dict with keys:
                       'model', 'r2', 'mae', 'mse',
                       'cv_scores', 'cv_mean', 'cv_std'
    X_train, X_test, y_train, y_test : split arrays
    """
    # ── Train / Test Split (80% / 20%) ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Model Definitions ────────────────────────────────────────────────────
    #    To add a new model, just add an entry here — the loop handles the rest.
    models = {
        "Decision Tree":    DecisionTreeRegressor(random_state=42, max_depth=6),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Test-set predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # K-Fold Cross Validation (cv=5)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

        results[name] = {
            "model":     model,
            "r2":        r2,
            "mae":       mae,
            "mse":       mse,
            "cv_scores": cv_scores,
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
        }

    return results, X_train, X_test, y_train, y_test
