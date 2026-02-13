# this script runs once to backfill historical data from open-meteo into hopsworks

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from features import build_features
from data_ingestion import fetch_historical, fetch_recent

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1


def get_hopsworks_project():
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project_name or not api_key:
        raise RuntimeError("Project or API key not set in .env")
    return hopsworks.login(project=project_name, api_key_value=api_key)

def main():
    project = get_hopsworks_project()
    fs = project.get_feature_store()

    # Create/get feature group
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features for Bahawalpur",
        online_enabled=False
    )

    # Fetch raw data
    
    print("Running historical backfill")
    raw_df = fetch_historical("2025-05-02", "2026-01-19")
    
    if raw_df.empty:
        print("No new data to insert.")
        return

    # Build features
    feature_df = build_features(raw_df)
    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"], utc=True)
    feature_df = feature_df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # Insert into Hopsworks
    fg.insert(feature_df, write_options={"wait_for_job": True})
    print(f"Inserted {len(feature_df)} rows into {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")

if __name__ == "__main__":
    main()
