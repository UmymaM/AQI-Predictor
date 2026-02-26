# detects and fills gaps with smart averages

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from train import get_hopsworks_project, load_features_from_hopsworks
import hopsworks 
import os
from fetch_hourly_test import align_dtypes_with_feature_group

FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1

def calculate_smart_averages(existing_df, target_timestamps):
    # calculate averages based on historical patterns
    df = existing_df.copy()
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour_of_day']
    numeric_cols = [
        'pm2_5', 'pm10', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'pressure_msl', 'precipitation',
        'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    filled_data = []
    
    for target_ts in target_timestamps:
        target_hour = target_ts.hour
        target_dow = target_ts.dayofweek
        target_hour_of_week = target_dow * 24 + target_hour
        row = {'timestamp': target_ts}
        
        for col in numeric_cols:
            same_hour_data = df[df['hour_of_week'] == target_hour_of_week]
            
            if len(same_hour_data) >= 3:
                row[col] = same_hour_data[col].median()
                if col in ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']:
                    noise = np.random.normal(0, 0.08)
                else:
                    noise = np.random.normal(0, 0.05)
                row[col] = row[col] * (1 + noise)
            else:
                same_hour_all = df[df['hour_of_day'] == target_hour]
                row[col] = same_hour_all[col].median() if len(same_hour_all) > 0 else df[col].median()
        
        filled_data.append(row)
    
    return pd.DataFrame(filled_data)


def build_features_missed_hrs(df):
    # feature engineering from raw data
    df = df.copy()
    
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    
    df["pm25_lag1"] = df["pm2_5"].shift(1)
    df["pm25_lag6"] = df["pm2_5"].shift(6)
    df["pm25_lag24"] = df["pm2_5"].shift(24)
    
    df["pm25_ma6"] = df["pm2_5"].rolling(6, min_periods=1).mean()
    df["pm25_ma24"] = df["pm2_5"].rolling(24, min_periods=1).mean()
    
    df["pm25_change_1hr"] = df["pm2_5"] - df["pm25_lag1"]
    
    required = [
        'pm2_5', 'pm10', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m','wind_direction_10m', 'pressure_msl', 'precipitation','cloud_cover_low',
        'carbon_monoxide','carbon_dioxide',
        'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'hour', 'day_of_week',
        'day', 'month', 'pm25_lag1', 'pm25_lag6', 'pm25_lag24', 'pm25_ma6',
        'pm25_ma24', 'pm25_change_1hr']
    
    for col in required:
        if col not in df.columns:
            if col in ['pm10', 'carbon_monoxide', 'nitrogen_dioxide', 
                      'sulphur_dioxide', 'ozone', 'precipitation']:
                df[col] = 0.0
            elif col == 'day':
                df[col] = df['timestamp'].dt.day
            else:
                df[col] = np.nan
    
    return df.fillna(method='ffill').fillna(method='bfill')


def auto_backfill():
    print("Checking for missing data")
    # Connect
    project = get_hopsworks_project()
    fs = project.get_feature_store(name='aqi_prediction_bwp_featurestore')

    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    # Check current state
    state = fg.materialization_job.get_state()
    print(f"Job state: {state}")

    # If not running, trigger it
    if state not in ["RUNNING", "INITIALIZING"]:
        fg.materialization_job.run(
            args="-op offline_fg_materialization -path hdfs:///Projects/AQI_Prediction_BWP/Resources/jobs/aqi_features_hourly_1_offline_fg_materialization/config_1772122602973",
            await_termination=True  # blocks until done
        )
        print("Materialization complete!")
    else:
        print("Job still running - wait and check UI")
        fg.materialization_job.get_final_state()  # blocks until finished
        print("Finished!")

    # Verify
    df = fg.read()
    print(f"Row count: {len(df)}")
    print(f"Max timestamp: {df['timestamp'].max()}")
    # Load existing data
    df=load_features_from_hopsworks()
    
    # Find all missing hours
    min_ts = df['timestamp'].min()
    max_ts = pd.Timestamp.now(tz='UTC').floor('H')
    # max_ts = df['timestamp'].max()
    expected = pd.date_range(start=min_ts, end=max_ts, freq='H')
    existing = set(df['timestamp'])
    missing = sorted([ts for ts in expected if ts not in existing])
    
    if not missing:
        print("No gaps found!")
        return
    
    print(f"Found {len(missing)} missing hours from {missing[0]} to {missing[-1]}")
    
    # Generate filled data
    filled_raw = calculate_smart_averages(df, missing)
    feature_df = build_features_missed_hrs(filled_raw)
    feature_df = align_dtypes_with_feature_group(feature_df, fg)

    
    
    # Insert
    print("Inserting into Hopsworks")
    fg.insert(feature_df, write_options={"wait_for_job": True})
    print(f"Backfilled {len(feature_df)} missing hours")


def main():
    try:
        print("Automatic Backfill")
        auto_backfill()
        print("Backfill Completed")
        return 0
        
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
