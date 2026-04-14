"""
utils.py
────────
Shared utility / helper functions used across multiple pages.

  - grade_label()   : Maps a numeric grade to a category string + display colours
  - encode_input()  : Encodes a single categorical value using a fitted LabelEncoder
"""


def grade_label(grade: float) -> tuple:
    """
    Map a predicted G3 grade (0–20) to a human-readable category and colours.

    Parameters
    ----------
    grade : float  - Predicted final grade

    Returns
    -------
    (label, text_colour, background_colour) : tuple of str
    """
    if grade >= 17:
        return "Excellent 🏆", "#1b5e20", "#e8f5e9"
    if grade >= 14:
        return "Good 👍",       "#1565c0", "#e3f2fd"
    if grade >= 10:
        return "Sufficient ✅", "#e65100", "#fff3e0"
    return "At Risk ⚠️",       "#b71c1c", "#ffebee"


def encode_input(encoders: dict, col: str, val) -> int:
    """
    Encode a single categorical input value using its fitted LabelEncoder.

    Parameters
    ----------
    encoders : dict  - {col_name: LabelEncoder} returned by data_loader
    col      : str   - Column name
    val      : str   - Raw value to encode

    Returns
    -------
    int  - Encoded integer, or 0 if the value is unseen
    """
    le = encoders[col]
    if val in le.classes_:
        return int(le.transform([val])[0])
    return 0   # fallback for any unseen value
