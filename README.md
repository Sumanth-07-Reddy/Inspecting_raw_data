# Inspecting_raw_data

# Used Car Price Prediction - Data Preprocessing Pipeline

This project demonstrates a complete data cleaning and preprocessing pipeline for a used car price prediction task. The goal is to explore a messy real-world dataset, clean it systematically, encode categorical features, and compute a simple baseline model using Mean Absolute Error (MAE).

## Tasks Completed

### Task 1: Load, Inspect, and Clean
- Loaded the raw used cars dataset using `pd.read_csv()`
- Identified key data quality issues: missing values, duplicates, non-numeric values in `mileage`, `engine`, `max_power`, and inconsistent text in categorical columns
- Cleaned the data by:
  - Dropping rows with missing `selling_price` (target variable)
  - Removing duplicate rows
  - Extracting numeric values from `mileage`, `engine`, and `max_power`
  - Filling remaining missing values with median
  - Standardizing categorical columns (strip whitespace + lowercase)

### Task 2: Encode Categorical Features
- Applied **Label Encoding** to `transmission_type` (Manual → 0, Automatic → 1)
- Applied **One-Hot Encoding** to `fuel_type` and `seller_type` using `pd.get_dummies(drop_first=True)`
- Printed the final list of columns after encoding

### Task 3: Split and Compute Baseline MAE
- Defined features (`X`) and target (`y = selling_price`)
- Performed train-test split (80/20) with `random_state=42`
- Built a simple **baseline model** by predicting the mean `selling_price` for every test record
- Calculated and printed the **Baseline MAE**

## Key Learnings
- Importance of proper data cleaning before modeling
- Difference between Label Encoding and One-Hot Encoding
- Why we must drop rows with missing target values
- How to compute a baseline MAE for regression problems

---

**Submitted as part of Masai School Machine Learning Curriculum**

Technologies: Python, Pandas, NumPy, Scikit-learn
