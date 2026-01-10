import pandas as pd
import numpy as np



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()

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

    # Final feature selection
    drop_cols = [
        "carbon_dioxide",
        "cloud_cover_low",
        "wind_direction_10m"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


# if __name__ == "__main__":
#     merged_df=pd.read_parquet("data\\raw\\merged_data_bwp.parquet")
#     df_features=build_features(merged_df)
#     df_features.to_parquet("data\\processed\\aqi_features_bwp.parquet",index=False)
#     df_features.to_csv("data\\processed\\aqi_features_bwp.csv",index=False)
#     print("Fetaure generation complete! Data shape: ",df_features.shape)
    