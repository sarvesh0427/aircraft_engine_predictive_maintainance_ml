import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from feature_engineering import (
    create_rul_target,
    drop_non_informative_columns,
    create_rolling_features
)

# project root path
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train_FD001.txt"
MODEL_DIR = BASE_DIR / "models"

# create models folder if not exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# load data
columns = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
df = df.iloc[:, :26]
df.columns = columns

df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

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

train_df = df[df["engine_id"].isin(train_engines)]
test_df = df[df["engine_id"].isin(test_engines)]

# features and target
X_train = train_df.drop(columns=["engine_id", "cycle", "RUL"])
y_train = train_df["RUL"]

X_test = test_df.drop(columns=["engine_id", "cycle", "RUL"])
y_test = test_df["RUL"]

# model
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    verbose=100,
    random_seed=42
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test)
)

# save model and features
joblib.dump(model, MODEL_DIR / "catboost_rul.pkl")
joblib.dump(X_train.columns.tolist(), MODEL_DIR / "feature_columns.pkl")

print("Training completed successfully!")