# Aircraft Engine Predictive Maintenance (RUL Prediction)

## Project Overview
This project focuses on **predictive maintenance for aircraft jet engines** by estimating the  
**Remaining Useful Life (RUL)** using multivariate sensor time-series data.

Predictive maintenance helps airlines:
- Reduce unexpected engine failures
- Lower maintenance costs
- Improve operational safety

The project is built using **Python, Data Science, and Machine Learning**, with a clear separation
between data exploration, feature engineering, modeling, and deployment.

---

## Problem Statement
Given historical sensor readings from aircraft engines operating until failure,  
the goal is to **predict how many operational cycles remain before engine failure**.

This is formulated as a **supervised regression problem**.

---

## Dataset
- **NASA Turbofan Engine Degradation Dataset (C-MAPSS)**
- Subset used: `FD001`

### Dataset Characteristics
- Multiple aircraft engines
- Each engine is a time-series (cycle-based)
- Features include:
  - 3 operational settings
  - 21 sensor measurements
- Each engine operates until failure

---

## Project Structure
```text
├── data/
│   ├── raw/                
│   └── processed/          
│
├── notebooks/
│   ├── 01_eda.ipynb        
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── app/
│   └── app.py             
│
├── models/
│   └── feature_columns.pkl
│   └── random_forest_rul.pkl
├── README.md
└── requirements.txt
```
---

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Git & GitHub
