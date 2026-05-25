import joblib
import pandas as pd
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "catboost_rul.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# load model and feature
model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURE_PATH)

# prediction
def predict_rul(input_df: pd.DataFrame):

    # Ensure correct feature order
    input_df = input_df[feature_columns]

    # Predict
    prediction = model.predict(input_df)

    return prediction


# testing
if __name__ == "__main__":

    # Example dummy input
    sample_data = pd.DataFrame([{
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,

        "sensor_2": 641.82,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_7": 554.36,
        "sensor_8": 2388.06,
        "sensor_9": 9046.19,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_17": 392,
        "sensor_20": 39.06,
        "sensor_21": 23.4190
    }])

    prediction = predict_rul(sample_data)

    print(f"Predicted RUL: {prediction[0]:.2f}")