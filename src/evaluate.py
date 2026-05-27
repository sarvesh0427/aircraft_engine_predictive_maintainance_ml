import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

from feature_engineering import (
    create_rul_target,
    drop_non_informative_columns,
    create_rolling_features)

# paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train_FD001.txt"

MODEL_PATH = BASE_DIR / "models" / "catboost_rul.pkl"

# load model 
model = joblib.load(MODEL_PATH)

# load data
columns = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None
)

# Remove extra empty columns
df = df.iloc[:, :26]

# Assign names
df.columns = columns

# Sort
df = df.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

# feature engineering
df = create_rul_target(df)

df = drop_non_informative_columns(df)

# train test split
engine_ids = df["engine_id"].unique()

train_engines, test_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

test_df = df[df["engine_id"].isin(test_engines)]

# feature and target
X_test = test_df.drop(
    columns=["engine_id", "cycle", "RUL"]
)

y_test = test_df["RUL"]

# prediction
predictions = model.predict(X_test)

# metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# results
print("\nMODEL EVALUATION:")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

print(f"R2   : {r2:.4f}")