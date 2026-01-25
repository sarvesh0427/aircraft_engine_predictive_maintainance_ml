import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Aircraft Engine RUL Predictor",
    layout="wide"
)

st.title("Aircraft Engine RUL Prediction")
st.markdown(
    "Predict **Remaining Useful Life (RUL)** using sensor data "
    "from turbofan aircraft engines."
)


# Load Model
@st.cache_resource
def load_model():
    return joblib.load("app/rul_model.pkl")


model = load_model()

# Sensor Inputs (Side by Side)
st.subheader("Engine Sensor Inputs")

sensor_cols = [f"sensor_{i}" for i in range(1, 19)]
input_data = {}


cols_per_row = 4
for i in range(0, len(sensor_cols), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, sensor in enumerate(sensor_cols[i:i+cols_per_row]):
        input_data[sensor] = cols[j].number_input(
            sensor,
            value=0.0,
            format="%.4f"
        )


# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Feature Engineering (Simple)
for sensor in sensor_cols:
    input_df[f"{sensor}_roll_mean"] = input_df[sensor]
    input_df[f"{sensor}_roll_std"] = 0.0

# Prediction
if st.button("Predict RUL"):
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Remaining Useful Life: **{prediction:.2f} cycles**")

    if prediction < 30:
        st.warning("Engine nearing end-of-life. Maintenance recommended.")
    else:
        st.info("Engine operating within normal conditions.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Aircraft Engine Predictive Maintenance Project")
