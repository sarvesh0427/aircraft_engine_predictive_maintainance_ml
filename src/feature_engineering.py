import pandas as pd


def create_rul_target(df):

    max_cycles = df.groupby("engine_id")["cycle"].max()

    df["RUL"] = (
        df["engine_id"].map(max_cycles)
        - df["cycle"])

    return df


def drop_non_informative_columns(df):

    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    df = df.drop(columns=op_cols)
    return df


def create_rolling_features(df, window=20):

    sensor_cols = [
        col for col in df.columns
        if "sensor_" in col]

    df_fe = df.copy()

    for sensor in sensor_cols:

        df_fe[f"{sensor}_roll_mean"] = (
            df_fe.groupby("engine_id")[sensor]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True))

        df_fe[f"{sensor}_roll_std"] = (
            df_fe.groupby("engine_id")[sensor]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True))
    df_fe = df_fe.fillna(0)
    return df_fe