import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# =========================
# ADD SRC TO PATH
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

# =========================
# IMPORT PREDICTION FUNCTION
# =========================
from predict import predict_rul

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Aircraft Engine RUL Prediction",
    page_icon="✈️",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("✈️ Aircraft Engine Predictive Maintenance")

st.markdown(
    """
Predict the **Remaining Useful Life (RUL)** of an aircraft engine
using sensor and operational data.
"""
)

# =========================
# USER INPUTS
# =========================
st.subheader("Enter Engine Sensor Values")

op_setting_1 = st.number_input("Operational Setting 1", value=0.0)
op_setting_2 = st.number_input("Operational Setting 2", value=0.0)
op_setting_3 = st.number_input("Operational Setting 3", value=100.0)

sensor_2 = st.number_input("Sensor 2", value=641.82)
sensor_3 = st.number_input("Sensor 3", value=1589.70)
sensor_4 = st.number_input("Sensor 4", value=1400.60)
sensor_7 = st.number_input("Sensor 7", value=554.36)
sensor_8 = st.number_input("Sensor 8", value=2388.06)
sensor_9 = st.number_input("Sensor 9", value=9046.19)
sensor_11 = st.number_input("Sensor 11", value=47.47)
sensor_12 = st.number_input("Sensor 12", value=521.66)
sensor_13 = st.number_input("Sensor 13", value=2388.02)
sensor_14 = st.number_input("Sensor 14", value=8138.62)
sensor_15 = st.number_input("Sensor 15", value=8.4195)
sensor_17 = st.number_input("Sensor 17", value=392.0)
sensor_20 = st.number_input("Sensor 20", value=39.06)
sensor_21 = st.number_input("Sensor 21", value=23.4190)
sensor_1 = st.number_input("Sensor 1", value=518.67)
sensor_5 = st.number_input("Sensor 5", value=14.62)
sensor_6 = st.number_input("Sensor 6", value=21.61)
sensor_10 = st.number_input("Sensor 10", value=1.30)
sensor_16 = st.number_input("Sensor 16", value=0.03)
sensor_18 = st.number_input("Sensor 18", value=2388.00)
sensor_19 = st.number_input("Sensor 19", value=100.00)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict RUL"):

    input_data = pd.DataFrame([{

    "op_setting_1": op_setting_1,
    "op_setting_2": op_setting_2,
    "op_setting_3": op_setting_3,

    "sensor_1": sensor_1,
    "sensor_2": sensor_2,
    "sensor_3": sensor_3,
    "sensor_4": sensor_4,
    "sensor_5": sensor_5,
    "sensor_6": sensor_6,
    "sensor_7": sensor_7,
    "sensor_8": sensor_8,
    "sensor_9": sensor_9,
    "sensor_10": sensor_10,
    "sensor_11": sensor_11,
    "sensor_12": sensor_12,
    "sensor_13": sensor_13,
    "sensor_14": sensor_14,
    "sensor_15": sensor_15,
    "sensor_16": sensor_16,
    "sensor_17": sensor_17,
    "sensor_18": sensor_18,
    "sensor_19": sensor_19,
    "sensor_20": sensor_20,
    "sensor_21": sensor_21}])

    prediction = predict_rul(input_data)

    rul = prediction[0]

    st.success(f"Predicted Remaining Useful Life: {rul:.2f} cycles")

    # =========================
    # HEALTH STATUS
    # =========================
    if rul > 120:
        st.info("✅ Engine Health: GOOD")

    elif rul > 60:
        st.warning("⚠️ Engine Health: MODERATE")

    else:
        st.error("🚨 Engine Health: CRITICAL — Maintenance Required")