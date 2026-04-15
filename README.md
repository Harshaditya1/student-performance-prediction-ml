# Student Performance Prediction App

## 1. Project Overview

This project is a Streamlit web application that predicts a student's final grade (`G3`) using the UCI Student Performance dataset. The app trains machine learning models on student data and provides:

- Dataset overview and statistics
- Model evaluation and comparison
- A prediction interface for new student input

## 2. Files and Responsibilities

- `app.py`
  - Streamlit entry point
  - Configures page layout and styles
  - Loads data and trains models
  - Routes between UI pages

- `config.py`
  - Dataset path and target column
  - Feature column list and readable labels
  - Categorical columns to encode

- `data_loader.py`
  - Reads `student-mat.csv`
  - Label-encodes categorical variables
  - Builds feature matrix `X` and target `y`

- `model_trainer.py`
  - Splits data into train/test sets
  - Trains `DecisionTreeRegressor` and `LinearRegression`
  - Computes metrics and 5-fold cross-validation

- `utils.py`
  - Converts predicted grade to category label
  - Encodes single categorical input values

- `styles.py`
  - Provides custom CSS for the Streamlit UI

- `views/page_overview.py`
  - Displays dataset metrics, raw records, and label encoding tables

- `views/page_evaluation.py`
  - Displays model metrics, CV results, and comparison

- `views/page_prediction.py`
  - Collects user input and shows predicted grade

## 3. Technologies Used

- Python
- Streamlit
- pandas
- numpy
- scikit-learn

## 4. Dataset and Inputs

### Dataset
- `student-mat.csv` (UCI Student Performance dataset)
- Target column: `G3` (final grade, range 0–20)

### Features
The model uses 32 features, including:

- Student demographics: `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`
- Parent education and jobs: `Medu`, `Fedu`, `Mjob`, `Fjob`
- School reasons and guardian: `reason`, `guardian`
- Study and school factors: `traveltime`, `studytime`, `failures`, `absences`
- Support flags: `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`
- Social/lifestyle metrics: `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`
- Previous grades: `G1`, `G2`

### Preprocessing
- Categorical columns are label-encoded using `sklearn.preprocessing.LabelEncoder`
- Numeric columns are used directly
- `X` is formed from all feature columns in the defined order
- `y` is the target vector containing `G3`

### Categorical Columns Encoded
- `school`, `sex`, `address`, `famsize`, `Pstatus`
- `Mjob`, `Fjob`, `reason`, `guardian`
- `schoolsup`, `famsup`, `paid`, `activities`
- `nursery`, `higher`, `internet`, `romantic`

## 5. Model Training and Evaluation

### Models
- Decision Tree Regressor (`max_depth=6`)
- Linear Regression

### Process
- Train/test split: 80% training, 20% testing
- Cross-validation: 5-fold CV on full dataset

### Metrics
- `R² Score`
- `MAE` (Mean Absolute Error)
- `MSE` (Mean Squared Error)
- `CV Mean R²`
- `CV Std Dev`

### Outputs shown in the app
- Train/test sample counts
- Per-model metrics and fold-wise CV scores
- Side-by-side performance comparison
- Best model based on CV mean R²
- Dedicated Comparative Analysis section for Decision Tree vs Linear Regression

## 6. Prediction Input / Output

### User Inputs
The prediction page collects the same 32 features used during training. Example input fields:

- `School`: GP or MS
- `Sex`: F or M
- `Age`: 15–22
- `Address`: U or R
- `Family Size`: GT3 or LE3
- `Parent Status`: T or A
- `Mother Education`, `Father Education`
- `Travel Time`, `Study Time`, `Past Failures`, `Absences`
- `Extra Support`, `Paid Classes`, `Activities`, `Nursery`, `Higher`, `Internet`, `Romantic`
- `Family Relationship`, `Free Time`, `Going Out`, `Alcohol Use`, `Health`
- `G1`, `G2`

### Output
- Predicted final grade `G3` (rounded and clamped to 0–20)
- Grade category label:
  - `Excellent 🏆`
  - `Good 👍`
  - `Sufficient ✅`
  - `At Risk ⚠️`
- CV R² score for the chosen model
- Input summary table showing entered values

## 7. How to Run the Project

### 1. Open a terminal in the project folder
- `D:\ANNProject`

### 2. Activate the virtual environment (optional)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the app
```bash
streamlit run app.py
```

### 4b. Or run with the specific Python executable
```bash
D:/ANNProject/.venv/Scripts/python.exe -m streamlit run D:/ANNProject/app.py
```

### 5. Open in browser
- `http://localhost:8501`

## 8. Requirements

From `requirements.txt`:
- `streamlit>=1.30.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `openpyxl>=3.1.0`
- `xlrd>=2.0.1`

## 9. Notes

- The app uses label encoding for categorical inputs, so prediction input values must match the expected categories.
- The model prediction is based on student features plus previous grades and returns a final grade estimate.
- The evaluation page helps compare the performance of the two models and identify the best model.
