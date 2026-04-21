import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("=== Used Car Price Prediction - Full Preprocessing Pipeline ===\n")

# ====================== TASK 1: Load, Inspect, and Clean ======================
print("Task 1: Loading and Cleaning Data...\n")

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/used_cars.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nMissing Values (%):")
print((df.isnull().sum() * 100 / len(df)).round(2))

df = df.dropna(subset=['selling_price'])                   
df = df.drop_duplicates()                                   

for col in ['mileage', 'engine', 'max_power']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract('(\d+\.?\d*)')[0].astype(float)

numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('selling_price', errors='ignore')
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

print(f"Cleaned DataFrame Shape: {df.shape}\n")

# ====================== TASK 2: Encode Categorical Features ======================
print("Task 2: Encoding Categorical Features...\n")


if 'transmission_type' in df.columns:
    df['transmission_type'] = df['transmission_type'].map({'manual': 0, 'automatic': 1})

df = pd.get_dummies(df, columns=['fuel_type', 'seller_type'], drop_first=True)

print("Final Columns after Encoding:")
print(df.columns.tolist())
print("\n")

# ====================== TASK 3: Split and Compute Baseline MAE ======================
print("Task 3: Splitting Data and Computing Baseline MAE...\n")

X = df.drop(columns=['selling_price'])
y = df['selling_price']

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_pred = np.full(shape=len(y_test), fill_value=y_train.mean())

baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline MAE: ₹{round(baseline_mae)}")
print(f"Mean Selling Price used for baseline: ₹{round(y_train.mean())}")