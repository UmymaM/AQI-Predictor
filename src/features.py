import pandas as pd
import numpy as np



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamps detected after conversion")

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"]=df["timestamp"].dt.dayofweek    
    df["day"]=df["timestamp"].dt.day
    df["month"]=df["timestamp"].dt.month

    # Lag features
    df["pm25_lag1"] = df["pm2_5"].shift(1)
    df["pm25_lag6"] = df["pm2_5"].shift(6)
    df["pm25_lag24"] = df["pm2_5"].shift(24)

    # Rolling features
    df["pm25_ma6"] = df["pm2_5"].rolling(6).mean()
    df["pm25_ma24"] = df["pm2_5"].rolling(24).mean()

    # Change feature
    df["pm25_change_1hr"] = df["pm2_5"] - df["pm25_lag1"]

    # Drop NaNs
    df = df.dropna()

    return df
    