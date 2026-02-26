import requests
import pandas as pd
from dotenv import load_dotenv
from time import sleep
import urllib3
import os
import hopsworks
from features import build_features

FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1

load_dotenv()

# aligining data types
def align_dtypes_with_feature_group(df, fg):
    
    schema = {f.name: f.type for f in fg.features}
    df = df.copy()
    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        if dtype == "double":
            df[col] = df[col].astype("float64")
        elif dtype == "float":
            df[col] = df[col].astype("float32")
        elif dtype == "int":
            df[col] = df[col].astype("int32")
        elif dtype == "bigint":
            df[col] = df[col].astype("int64")

    return df

# fetching current weather+pollutant data
def fetch_current_data():
    print("Fetching current data.")
    # weather
    weather_params = {
        "latitude": 29.4,
        "longitude": 71.68,
        "current": ["temperature_2m","relative_humidity_2m", "wind_speed_10m",
            "pressure_msl","precipitation"],
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

    # pollutants
    pollutant_params = {
        "latitude": 29.4,
        "longitude": 71.68,
        "current": ["pm2_5","pm10","sulphur_dioxide",
            "nitrogen_dioxide","ozone","carbon_monoxide"],
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

    # houlry normalization
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
    pollutant_df["timestamp"] = pd.to_datetime(pollutant_df["timestamp"], utc=True)

    weather_df["timestamp"] = weather_df["timestamp"].dt.floor("h")
    pollutant_df["timestamp"] = pollutant_df["timestamp"].dt.floor("h")

    # merging dataframes
    merged_df = pd.merge(
        weather_df,
        pollutant_df,
        on="timestamp",
        how="inner"
    )

    if merged_df.empty:
        raise ValueError("Hourly merge failed – timestamps still mismatched")
    
    return merged_df


# to reduce pipeline failures
def insert_with_retry(fg, data, max_retries=3, delay=5):
    # inserting data with retry logic
    for attempt in range(max_retries):
        try:
            fg.insert(data, write_options={"wait_for_job": True})
            print(f"Insert successful on attempt {attempt + 1}")
            return True
        except (requests.exceptions.ConnectionError, 
                urllib3.exceptions.ProtocolError) as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt) 
                print(f"Connection failed on attempt {attempt + 1}")
                print(f"Retrying in {wait_time}s")
                sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed")
                raise


# main pipeline
def build_hourly_features():
    print("Hourly Feature Pipeline")
    try:
        current_df = fetch_current_data()
        project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT"),  # Replace with your project name
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=os.getenv("HOPSWORKS_API_KEY")  # Get from Hopsworks UI > Account Settings > API Keys
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
        print("Connected to feature group")

        hist_df = fg.read()
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], utc=True)
        hist_df = hist_df.sort_values("timestamp").tail(72)
        
        print(f"Loaded {len(hist_df)} historical rows")
        print(f"Range: {hist_df['timestamp'].min()} to {hist_df['timestamp'].max()}")

        print("Combining data")
        combined_df = pd.concat([hist_df, current_df]).drop_duplicates(
            subset=["timestamp"],
            keep="last"
        ).sort_values("timestamp")
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True)
        
        print(f"Combined: {len(combined_df)} rows")

        print("Feature Engineering")
        feature_df = build_features(combined_df)
        
        print(f"Features built: {feature_df.shape}")

        latest = feature_df.tail(1).copy()
        
        print(f"Latest timestamp: {latest['timestamp'].iloc[0]}")
        print(f"PM2.5: {latest['pm2_5'].iloc[0]:.2f} µg/m³")

        # Duplicate protection
        existing_ts = set(hist_df["timestamp"].astype(str))
        ts = str(latest["timestamp"].iloc[0])

        if ts in existing_ts:
            return

        latest = align_dtypes_with_feature_group(latest, fg)

        # fg.insert(latest, write_options={"wait_for_job": True})
        insert_with_retry(fg, latest, max_retries=3, delay=5)
        print("Hourly features inserted successfully")

    except Exception as e:
        print("Pipeline failed")
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    build_hourly_features()