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


# model = joblib.load("models/random_forest_rul.pkl")
# feature_cols = joblib.load("models/feature_columns.pkl")
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/random_forest_rul.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, feature_cols


model, FEATURE_COLUMNS = load_artifacts()
# -------------------------------
# UI
# -------------------------------

# Tabs
tab1, tab2 = st.tabs(["RUL Predictor", "About"])

with tab1:
    # Sensor Inputs (Side by Side)
    st.subheader("Engine Sensor Inputs")

    sensor_cols = [f"sensor_{i}" for i in range(1, 21)]
    input_data = {}

    # Create columns: 4 sensors per row (adjust as needed)
    cols_per_row = 4
    for i in range(0, len(sensor_cols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, sensor in enumerate(sensor_cols[i:i+cols_per_row]):
            input_data[sensor] = cols[j].number_input(
                sensor,
                value=0.0,
                format="%.4f"
            )

    # -------------------------------
    # Build full feature row
    # -------------------------------
    input_df = pd.DataFrame([input_data])

    # Recreate rolling features (deployment approximation)
    for s in sensor_cols:
        input_df[f"{s}_roll_mean"] = input_df[s]
        input_df[f"{s}_roll_std"] = 0.0

    # VERY IMPORTANT: align column order
    input_df = input_df.reindex(columns=FEATURE_COLUMNS)

    # -------------------------------
    # Predict
    # -------------------------------
    if st.button("Predict RUL"):
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Remaining Useful Life: **{prediction:.2f} cycles**")

        if prediction < 30:
            st.warning("Maintenance recommended")
        else:
            st.info("Engine operating normally")

with tab2:
    st.subheader("About the Project")

    st.markdown(
        """
        ### Aircraft Engine Predictive Maintenance System

        This project focuses on **predicting the Remaining Useful Life (RUL)**
        of turbofan aircraft engines using sensor data collected during engine operation.

        ### Objective
        - Predict how many operational cycles remain before engine failure  
        - Enable **predictive maintenance** instead of reactive maintenance  
        - Reduce unexpected failures and maintenance costs  

        ### Dataset
        - Based on the **NASA C-MAPSS dataset**
        - Includes:
          - Engine operational settings
          - Multiple sensor measurements
          - Engine lifecycle data

        ### Machine Learning Model
        - **Random Forest Regressor**
        - Trained on engineered sensor features
        - Outputs predicted Remaining Useful Life in cycles

        ### Technologies Used
        - Python
        - Pandas, NumPy
        - Scikit-learn
        - Streamlit

        ### Use Case
        This system can assist aviation engineers and maintenance teams
        in making data-driven decisions for engine servicing and replacement.
        """
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:0.85em; color:gray;'>"
    "Built with Streamlit | Aircraft Engine Predictive Maintenance"
    "</p>",
    unsafe_allow_html=True
)
