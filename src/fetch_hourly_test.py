# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from features import build_features
# from train import  get_hopsworks_project
# FEATURE_GROUP_NAME = "aqi_features_hourly"
# FEATURE_GROUP_VERSION = 1

# # Get current hour
# now = datetime.now()
# current_hour = now.replace(minute=0, second=0, microsecond=0)

# FEATURE_DTYPES = {
#     "temperature_2m": "float32",
#     "relative_humidity_2m": "float32",
#     "wind_speed_10m": "float32",
#     "pressure_msl": "float32",
#     "precipitation": "float32",
#     "pm2_5": "float32",
#     "pm10": "float32",
#     "carbon_monoxide": "float32",
#     "nitrogen_dioxide": "float32",
#     "sulphur_dioxide": "float32",
#     "ozone": "float32",
#     "pm25_lag1": "float32",
#     "pm25_lag6": "float32",
#     "pm25_lag24": "float32",
#     "pm25_change_1hr": "float32"
# }


# def get_hourly_weather_data():
#     params = {
#         "latitude": 29.4,
#         "longitude": 71.68,
#         "current": [  # Use 'current' for real-time data
#             "temperature_2m",
#             "relative_humidity_2m",
#             "wind_speed_10m",
#             "pressure_msl",
#             "precipitation"
#         ],
#         "timezone": "UTC"
#     }

#     response = requests.get(
#         "https://api.open-meteo.com/v1/forecast",
#         params=params
#     )
#     data = response.json()


#     # Current values are in data['current']
#     current_weather = {
#         "timestamp": data['current']['time'],
#         "temperature_2m": float(data['current']['temperature_2m']),
#         "relative_humidity_2m": float(data['current']['relative_humidity_2m']),
#         "wind_speed_10m": float(data['current']['wind_speed_10m']),
#         "pressure_msl": float(data['current']['pressure_msl']),
#         "precipitation": float(data['current']['precipitation'])
#     }
#     return current_weather


# def get_hourly_pollutant_data():
#     params = {
#         "latitude": 29.4,
#         "longitude": 71.68,
#         "current": [  # Use 'current' for real-time data
#             "pm2_5",
#             "pm10",
#             "sulphur_dioxide",
#             "nitrogen_dioxide",
#             "ozone","carbon_monoxide"
#         ],
#         "timezone": "UTC"
#     }

#     response = requests.get(
#         "https://air-quality-api.open-meteo.com/v1/air-quality",
#         params=params
#     )
#     data = response.json()


#     # Current values
#     current_pollutants = {
#         "timestamp": data['current']['time'],
#         "pm2_5": float(data['current']['pm2_5']),
#         "pm10": float(data['current']['pm10']),
#         "ozone": float(data['current']['ozone']),
#         "sulphur_dioxide": float(data['current']['sulphur_dioxide']),
#         "nitrogen_dioxide": float(data['current']['nitrogen_dioxide']),
#         "carbon_monoxide": float(data['current']['carbon_monoxide'])
#     }
#     return current_pollutants


# def merge_weather_pollutants(weather: dict, pollutants: dict):
#     weather_df = pd.DataFrame([weather])
#     pollutant_df = pd.DataFrame([pollutants])

#     # Convert to datetime
#     weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
#     pollutant_df["timestamp"] = pd.to_datetime(pollutant_df["timestamp"], utc=True)

#     # Floor to hour (CRITICAL)
#     weather_df["timestamp"] = weather_df["timestamp"].dt.floor("H")
#     pollutant_df["timestamp"] = pollutant_df["timestamp"].dt.floor("H")

#     # Inner join on floored hour
#     merged_df = pd.merge(
#         weather_df,
#         pollutant_df,
#         on="timestamp",
#         how="inner"
#     )

#     if merged_df.empty:
#         raise ValueError("No overlapping hourly timestamp between weather and pollutants")

#     return merged_df


#     # if current_weather["timestamp"] != current_pollutants["timestamp"]:
#     #     raise ValueError("Weather and pollutant timestamps do not match")

#     # df = pd.DataFrame([{**current_weather, **current_pollutants}])
#     # df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

#     # return df
# # ____________________________________________________________________________
#     # current_weather_df=pd.DataFrame([current_weather])
#     # current_pollutant_df=pd.DataFrame([current_pollutants])
#     # merged_df = pd.merge(current_weather_df, current_pollutant_df, on="timestamp", how="inner")
#     # print("merged current data successfully!")
#     # return merged_df

# def enforce_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
#     # Add missing columns
#     for col in FEATURE_DTYPES:
#         if col not in df.columns:
#             df[col] = 0.0  # default value

#     # Convert all columns to the correct dtype
#     for col, dtype in FEATURE_DTYPES.items():
#         df[col] = df[col].astype(dtype)

#     # Ensure timestamp is datetime64[ns, UTC]
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

#     return df


# def build_hourly_df():
#     weather_df=get_hourly_weather_data()
#     pollutant_df=get_hourly_pollutant_data()
#     merged_df=merge_weather_pollutants(weather_df,pollutant_df)
#     feature_df=build_features(merged_df)
#     feature_df = (
#         feature_df
#         .sort_values("timestamp")
#         .drop_duplicates(subset=["timestamp"], keep="last")
#         .reset_index(drop=True)
#     )
#     feature_df = enforce_feature_schema(feature_df)
#     project=get_hopsworks_project()
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
#     try:
#         fg.insert(feature_df, write_options={"wait_for_job": True})
#         print("hourly data inserted!")
#     except Exception as e:
#         print(f"❌ Error inserting data: {e}")
#         raise    


# if __name__ == "__main__":
#     build_hourly_df()


#  THE ABOVE SCRIPT IS THRowinf an error

# the script below is from sonnet4.5:

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks
from dotenv import load_dotenv
import os

from features import build_features
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1

load_dotenv()


# ============================================================
# FETCH CURRENT WEATHER + POLLUTANTS (HOURLY-ALIGNED)
# ============================================================
def fetch_current_data():

    print("\n" + "=" * 70)
    print("FETCHING CURRENT DATA")
    print("=" * 70)

    # ---------------- WEATHER ----------------
    weather_params = {
        "latitude": 29.4,
        "longitude": 71.68,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "pressure_msl",
            "precipitation"
        ],
        "timezone": "UTC"
    }

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=weather_params,
        timeout=10
    ).json()

    weather_df = pd.DataFrame([{
        "timestamp": weather["current"]["time"],
        "temperature_2m": weather["current"]["temperature_2m"],
        "relative_humidity_2m": weather["current"]["relative_humidity_2m"],
        "wind_speed_10m": weather["current"]["wind_speed_10m"],
        "pressure_msl": weather["current"]["pressure_msl"],
        "precipitation": weather["current"]["precipitation"]
    }])

    # ---------------- POLLUTANTS ----------------
    pollutant_params = {
        "latitude": 29.4,
        "longitude": 71.68,
        "current": [
            "pm2_5",
            "pm10",
            "sulphur_dioxide",
            "nitrogen_dioxide",
            "ozone",
            "carbon_monoxide"
        ],
        "timezone": "UTC"
    }

    pollutant = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=pollutant_params,
        timeout=10
    ).json()

    pollutant_df = pd.DataFrame([{
        "timestamp": pollutant["current"]["time"],
        "pm2_5": pollutant["current"]["pm2_5"],
        "pm10": pollutant["current"]["pm10"],
        "sulphur_dioxide": pollutant["current"]["sulphur_dioxide"],
        "nitrogen_dioxide": pollutant["current"]["nitrogen_dioxide"],
        "ozone": pollutant["current"]["ozone"],
        "carbon_monoxide": pollutant["current"]["carbon_monoxide"]
    }])

    # ---------------- FIX-2: HOURLY NORMALIZATION ----------------
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
    pollutant_df["timestamp"] = pd.to_datetime(pollutant_df["timestamp"], utc=True)

    weather_df["timestamp"] = weather_df["timestamp"].dt.floor("H")
    pollutant_df["timestamp"] = pollutant_df["timestamp"].dt.floor("H")

    # ---------------- MERGE ----------------
    merged_df = pd.merge(
        weather_df,
        pollutant_df,
        on="timestamp",
        how="inner"
    )

    if merged_df.empty:
        raise ValueError("Hourly merge failed – timestamps still mismatched")

    print(f"✓ Merged hourly timestamp: {merged_df['timestamp'].iloc[0]}")
    print(f"PM2.5: {merged_df['pm2_5'].iloc[0]} µg/m³")

    return merged_df


# ============================================================
# MAIN PIPELINE
# ============================================================
def build_hourly_features():

    print("\n" + "=" * 70)
    print("HOURLY FEATURE PIPELINE")
    print("=" * 70)

    # 1. Fetch current data
    current_df = fetch_current_data()

    # 2. Connect to Hopsworks
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features for Bahawalpur",
        online_enabled=False
    )

    # 3. Load historical context (lags)
    hist_df = fg.read()
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], utc=True)
    hist_df = hist_df.sort_values("timestamp").tail(72)

    # 4. Combine
    combined_df = pd.concat([hist_df, current_df]).drop_duplicates(
        subset=["timestamp"],
        keep="last"
    ).sort_values("timestamp")

    # 5. Feature engineering
    feature_df = build_features(combined_df)

    latest = feature_df.tail(1).copy()

    # 6. Duplicate protection
    existing_ts = set(hist_df["timestamp"].astype(str))
    ts = str(latest["timestamp"].iloc[0])

    if ts in existing_ts:
        print(f"⚠️ Timestamp {ts} already exists — skipping insert")
        return

    # 7. Insert
    fg.insert(latest, write_options={"wait_for_job": True})

    print("\n" + "=" * 70)
    print("✅ HOURLY FEATURES INSERTED")
    print("=" * 70)
    print(f"Timestamp: {ts}")
    print(f"PM2.5: {latest['pm2_5'].iloc[0]:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    build_hourly_features()

