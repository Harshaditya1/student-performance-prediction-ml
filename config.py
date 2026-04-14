"""
config.py
─────────
Central configuration file.
All constants used across the app are defined here:
  - Dataset path
  - Categorical column names (for label encoding)
  - Feature column names and their human-readable UI labels
  - Target column name
"""

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH = "student-mat.csv"
TARGET_COL   = "G3"          # Final grade (0–20)

# ── Categorical columns (will be label-encoded) ───────────────────────────────
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# ── Feature columns with readable UI labels ───────────────────────────────────
#    Order matters — must match the column order sent to the model.
COL_LABELS = {
    "school":     "School (GP/MS)",
    "sex":        "Sex (F/M)",
    "age":        "Age",
    "address":    "Address (U=Urban / R=Rural)",
    "famsize":    "Family Size (LE3 / GT3)",
    "Pstatus":    "Parent Status (T=Together / A=Apart)",
    "Medu":       "Mother Education (0–4)",
    "Fedu":       "Father Education (0–4)",
    "Mjob":       "Mother's Job",
    "Fjob":       "Father's Job",
    "reason":     "Reason for Choosing School",
    "guardian":   "Guardian",
    "traveltime": "Travel Time to School (1–4)",
    "studytime":  "Weekly Study Time (1–4)",
    "failures":   "Past Class Failures (0–3)",
    "schoolsup":  "Extra School Support (yes/no)",
    "famsup":     "Family Educational Support (yes/no)",
    "paid":       "Extra Paid Classes (yes/no)",
    "activities": "Extra-Curricular Activities (yes/no)",
    "nursery":    "Attended Nursery (yes/no)",
    "higher":     "Wants Higher Education (yes/no)",
    "internet":   "Internet Access at Home (yes/no)",
    "romantic":   "In a Romantic Relationship (yes/no)",
    "famrel":     "Family Relationship Quality (1–5)",
    "freetime":   "Free Time after School (1–5)",
    "goout":      "Going Out with Friends (1–5)",
    "Dalc":       "Workday Alcohol Consumption (1–5)",
    "Walc":       "Weekend Alcohol Consumption (1–5)",
    "health":     "Current Health Status (1–5)",
    "absences":   "Number of School Absences",
    "G1":         "First Period Grade (0–20)",
    "G2":         "Second Period Grade (0–20)",
}

# Ordered list of feature column names (used as X)
FEATURE_COLS = list(COL_LABELS.keys())
