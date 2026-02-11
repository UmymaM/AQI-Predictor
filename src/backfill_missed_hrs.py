# """
# Backfill script for missed hours using SMART AVERAGES.
# Instead of failing on API calls, uses historical patterns to fill gaps.

# Usage:
#     # Backfill specific date range (using averages)
#     python backfill_missed_hrs.py --start "2026-01-20" --end "2026-01-23"
    
#     # Backfill last N hours
#     python backfill_missed_hrs.py --last-hours 24
    
#     # Check for gaps and backfill
#     python backfill_missed_hrs.py --auto
    
#     # Dry run (test without inserting)
#     python backfill_missed_hrs.py --last-hours 24 --dry-run
# """

# import argparse
# from datetime import datetime, timedelta
# from zoneinfo import ZoneInfo
# import pandas as pd
# import numpy as np
# import hopsworks
# from dotenv import load_dotenv
# import os
# import warnings
# warnings.filterwarnings('ignore')

# load_dotenv()

# BAHAWALPUR_LAT = 29.4
# BAHAWALPUR_LON = 71.68
# TIMEZONE = "Asia/Karachi"
# FEATURE_GROUP_NAME = "aqi_features_hourly"
# FEATURE_GROUP_VERSION = 4


# def get_hopsworks_project():
#     """Connect to Hopsworks."""
#     project_name = os.getenv("HOPSWORKS_PROJECT")
#     api_key = os.getenv("HOPSWORKS_API_KEY")
#     if not project_name or not api_key:
#         raise RuntimeError("Environment variables not set")
#     return hopsworks.login(project=project_name, api_key_value=api_key)


# def get_existing_data(fg):
#     """Get all existing data from feature store."""
#     print("Loading existing data from Hopsworks...")
#     df = fg.read()
#     df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
#     df = df.sort_values('timestamp')
#     print(f"✓ Loaded {len(df)} historical rows")
#     print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
#     return df


# def calculate_smart_averages(existing_df, target_timestamps):
#     """
#     Calculate smart averages for target timestamps based on historical patterns.
    
#     Strategy:
#     1. Hour-of-week patterns (Monday 9AM similar to previous Mondays 9AM)
#     2. Recent trends (last 7 days)
#     3. Seasonal patterns (same month/year)
#     """
#     print(f"\nCalculating smart averages for {len(target_timestamps)} hours...")
    
#     # Prepare existing data
#     df = existing_df.copy()
#     df['hour_of_day'] = df['timestamp'].dt.hour
#     df['day_of_week'] = df['timestamp'].dt.dayofweek
#     df['hour_of_week'] = df['day_of_week'] * 24 + df['hour_of_day']
#     df['month'] = df['timestamp'].dt.month
    
#     # Numeric columns to fill (exclude timestamp and derived features)
#     numeric_cols = [col for col in df.columns if col not in [
#         'timestamp', 'hour', 'day_of_week', 'day', 'month',
#         'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
#         'is_weekend', 'hour_of_day', 'hour_of_week',
#         'pm25_lag1', 'pm25_lag6', 'pm25_lag24',
#         'pm25_ma6', 'pm25_ma24', 'pm25_std6', 'pm25_std24',
#         'pm25_change_1hr', 'pm25_change_24hr'
#     ] and pd.api.types.is_numeric_dtype(df[col])]
    
#     print(f"  Filling {len(numeric_cols)} numeric columns")
    
#     # Create output DataFrame
#     filled_data = []
    
#     for target_ts in target_timestamps:
#         target_hour = target_ts.hour
#         target_dow = target_ts.dayofweek
#         target_month = target_ts.month
#         target_hour_of_week = target_dow * 24 + target_hour
        
#         # Initialize row with timestamp
#         row = {'timestamp': target_ts}
        
#         for col in numeric_cols:
#             # Strategy 1: Same hour-of-week pattern
#             same_hour_of_week_data = df[df['hour_of_week'] == target_hour_of_week]
            
#             if len(same_hour_of_week_data) >= 3:  # Need at least 3 samples
#                 # Use median (robust to outliers)
#                 row[col] = same_hour_of_week_data[col].median()
                
#                 # Add small random variation (5% noise)
#                 if not np.isnan(row[col]):
#                     noise = np.random.normal(0, 0.05)  # 5% standard deviation
#                     row[col] = row[col] * (1 + noise)
                    
#             else:
#                 # Strategy 2: Same hour of day across all days
#                 same_hour_data = df[df['hour_of_day'] == target_hour]
                
#                 if len(same_hour_data) >= 3:
#                     row[col] = same_hour_data[col].median()
#                 else:
#                     # Strategy 3: Overall median
#                     row[col] = df[col].median()
        
#         filled_data.append(row)
    
#     filled_df = pd.DataFrame(filled_data)
    
#     # Add small trend if we have recent data
#     if len(target_timestamps) > 1 and len(df) > 0:
#         # Calculate recent trend from last week
#         recent_cutoff = df['timestamp'].max() - timedelta(days=7)
#         recent_data = df[df['timestamp'] >= recent_cutoff]
        
#         if len(recent_data) > 24:  # At least a day of recent data
#             for col in numeric_cols:
#                 if col in recent_data.columns:
#                     recent_trend = recent_data[col].diff().mean()
                    
#                     if not np.isnan(recent_trend):
#                         # Apply trend linearly across filled hours
#                         for i in range(len(filled_df)):
#                             filled_df.loc[i, col] = filled_df.loc[i, col] + (recent_trend * i / 24)
    
#     print(f"✓ Generated averages with {len(filled_df)} rows")
#     return filled_df


# def build_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Build features from raw data."""
#     df = df.copy()
    
#     # Ensure timezone
#     if df['timestamp'].dt.tz is None:
#         df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
#     else:
#         df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
#     # Temporal features
#     df["hour"] = df["timestamp"].dt.hour
#     df["day_of_week"] = df["timestamp"].dt.dayofweek
#     df["day"] = df["timestamp"].dt.day
#     df["month"] = df["timestamp"].dt.month
    
#     # Cyclic encoding
#     df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
#     df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
#     df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
#     df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
#     df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#     df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
#     df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
#     # Lag features (will be NaN for first rows, that's OK)
#     df["pm25_lag1"] = df["pm2_5"].shift(1) if "pm2_5" in df.columns else np.nan
#     df["pm25_lag6"] = df["pm2_5"].shift(6) if "pm2_5" in df.columns else np.nan
#     df["pm25_lag24"] = df["pm2_5"].shift(24) if "pm2_5" in df.columns else np.nan
    
#     # Rolling features
#     df["pm25_ma6"] = df["pm2_5"].rolling(6, min_periods=1).mean() if "pm2_5" in df.columns else np.nan
#     df["pm25_ma24"] = df["pm2_5"].rolling(24, min_periods=1).mean() if "pm2_5" in df.columns else np.nan
#     df["pm25_std6"] = df["pm2_5"].rolling(6, min_periods=1).std() if "pm2_5" in df.columns else np.nan
#     df["pm25_std24"] = df["pm2_5"].rolling(24, min_periods=1).std() if "pm2_5" in df.columns else np.nan
    
#     # Change features
#     df["pm25_change_1hr"] = df["pm2_5"] - df["pm25_lag1"] if "pm2_5" in df.columns else np.nan
#     df["pm25_change_24hr"] = df["pm2_5"] - df["pm25_lag24"] if "pm2_5" in df.columns else np.nan
    
#     return df


# def find_missing_hours(fg) -> pd.DatetimeIndex:
#     """
#     Find missing hours in the feature store.
    
#     Returns:
#         DatetimeIndex of missing timestamps
#     """
#     print("\n=== Checking for missing hours ===")
    
#     try:
#         df = fg.read()
#         df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
#         # Get min and max timestamps
#         min_ts = df['timestamp'].min()
#         max_ts = df['timestamp'].max()
        
#         print(f"Data range: {min_ts} to {max_ts}")
#         print(f"Total hours in store: {len(df)}")
        
#         # Generate expected hourly range
#         expected = pd.date_range(start=min_ts, end=max_ts, freq='H')
#         print(f"Expected hours: {len(expected)}")
        
#         # Find missing
#         existing = set(df['timestamp'])
#         missing = [ts for ts in expected if ts not in existing]
        
#         if missing:
#             print(f"⚠️  Found {len(missing)} missing hours")
#             print(f"  First missing: {missing[0]}")
#             print(f"  Last missing: {missing[-1]}")
#             return pd.DatetimeIndex(missing)
#         else:
#             print("✓ No missing hours found")
#             return pd.DatetimeIndex([])
            
#     except Exception as e:
#         print(f"Error checking for gaps: {e}")
#         return pd.DatetimeIndex([])


# def backfill_with_averages(start_date: str, end_date: str, dry_run: bool = False):
#     """
#     Backfill missing data using smart historical averages.
#     No API calls needed!
    
#     Args:
#         start_date: Start date (YYYY-MM-DD)
#         end_date: End date (YYYY-MM-DD)
#         dry_run: If True, don't actually insert data
#     """
#     print("\n" + "="*70)
#     print("BACKFILL WITH SMART AVERAGES")
#     print("="*70)
#     print(f"Filling: {start_date} to {end_date}")
#     print("Using historical patterns instead of API calls")
    
#     # Connect to Hopsworks
#     project = get_hopsworks_project()
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
#     # 1. Get existing historical data
#     existing_df = get_existing_data(fg)
    
    
#     # 2. Generate timestamps for missing period
#     start_dt = pd.to_datetime(start_date).tz_localize('UTC')
#     end_dt = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(hours=23, minutes=59)
    
#     all_timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
    
#     # Check which timestamps are already present
#     existing_timestamps = set(existing_df['timestamp'])
#     missing_timestamps = [ts for ts in all_timestamps if ts not in existing_timestamps]
    
#     if not missing_timestamps:
#         print("✅ All hours already exist in feature store!")
#         return
    
#     print(f"\nFound {len(missing_timestamps)} missing hours to fill")
#     print(f"  From {missing_timestamps[0]} to {missing_timestamps[-1]}")
    
#     # 3. Generate filled data using smart averages
#     filled_raw_df = calculate_smart_averages(existing_df, missing_timestamps)
    
#     # 4. Build features
#     print("\nBuilding features for filled data...")
#     feature_df = build_features(filled_raw_df)
    
#     # Fill NaN values in lag/rolling features (they'll be NaN at boundaries)
#     feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
    
#     print(f"\n✓ Generated {len(feature_df)} filled rows")
#     print(f"  Shape: {feature_df.shape}")
    
#     # Show sample
#     print("\nSample of filled data (first 3 rows):")
#     sample_cols = ['timestamp', 'pm2_5', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
#     sample_cols = [col for col in sample_cols if col in feature_df.columns]
#     print(feature_df[sample_cols].head(3))
    
#     print("\nStatistics of filled data:")
#     for col in ['pm2_5', 'temperature_2m']:
#         if col in feature_df.columns:
#             print(f"  {col}: mean={feature_df[col].mean():.2f}, std={feature_df[col].std():.2f}")
  
#     # 5. Insert into Hopsworks
#     print("\n" + "="*70)
#     print("INSERTING INTO HOPSWORKS")
#     print("="*70)
    
#     try:
#         fg.insert(feature_df, write_options={"wait_for_job": True})
#         print(f"✅ Successfully inserted {len(feature_df)} filled rows!")
#         print(f"✅ Used smart averaging based on {len(existing_df)} historical rows")
#         print(f"✅ No external API calls needed!")
#     except Exception as e:
#         print(f"❌ Error inserting data: {e}")
#         raise
    
#     print("\n" + "="*70)
#     print("✅ BACKFILL WITH AVERAGES COMPLETED")
#     print("="*70)


# def backfill_last_n_hours(hours: int, dry_run: bool = False):
#     """
#     Backfill last N hours using averages.
#     """
#     now = datetime.now(ZoneInfo(TIMEZONE))
#     start = now - timedelta(hours=hours)
    
#     start_date = start.strftime("%Y-%m-%d")
#     end_date = now.strftime("%Y-%m-%d")
    
#     print(f"\nBackfilling last {hours} hours...")
#     print(f"From: {start_date} {start.hour:02d}:00")
#     print(f"To:   {end_date} {now.hour:02d}:00")
    
#     backfill_with_averages(start_date, end_date, dry_run)


# def auto_backfill(dry_run: bool = False):
#     """
#     Auto-detect gaps and fill them with averages.
#     """
#     print("\n" + "="*70)
#     print("AUTO-DETECT AND FILL GAPS WITH AVERAGES")
#     print("="*70)
    
#     # Connect to Hopsworks
#     project = get_hopsworks_project()
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
#     # Find missing hours
#     missing = find_missing_hours(fg)
    
#     if len(missing) == 0:
#         print("✅ No missing hours found.")
#         return
    
#     # Group missing hours into continuous ranges
#     missing_sorted = sorted(missing)
#     ranges = []
#     current_start = missing_sorted[0]
#     current_end = missing_sorted[0]
    
#     for ts in missing_sorted[1:]:
#         if ts == current_end + pd.Timedelta(hours=1):
#             current_end = ts
#         else:
#             ranges.append((current_start, current_end))
#             current_start = ts
#             current_end = ts
    
#     ranges.append((current_start, current_end))
    
#     print(f"\nFound {len(ranges)} gap(s):")
#     for i, (start, end) in enumerate(ranges, 1):
#         hours = (end - start).total_seconds() / 3600 + 1
#         print(f"  Gap {i}: {start.date()} {start.hour:02d}:00 to {end.date()} {end.hour:02d}:00 ({hours:.0f} hours)")
    
#     # Fill each gap
#     for start, end in ranges:
#         start_date = start.strftime("%Y-%m-%d")
#         end_date = end.strftime("%Y-%m-%d")
        
#         print(f"\n" + "-"*50)
#         print(f"Filling gap: {start_date} to {end_date}")
#         print("-"*50)
        
#         backfill_with_averages(start_date, end_date, dry_run)


# def main():
#     """Main entry point."""
#     parser = argparse.ArgumentParser(description="Backfill missed hours using smart averages")
#     parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
#     parser.add_argument("--end", help="End date (YYYY-MM-DD)")
#     parser.add_argument("--last-hours", type=int, help="Backfill last N hours")
#     parser.add_argument("--auto", action="store_true", help="Auto-detect and fill gaps")
#     parser.add_argument("--dry-run", action="store_true", help="Don't actually insert data")
    
#     args = parser.parse_args()
    
#     try:
#         if args.auto:
#             # Auto-detect and fill gaps
#             auto_backfill(dry_run=args.dry_run)
            
#         elif args.last_hours:
#             # Backfill last N hours
#             backfill_last_n_hours(args.last_hours, dry_run=args.dry_run)
            
#         elif args.start and args.end:
#             # Backfill specific range
#             backfill_with_averages(args.start, args.end, dry_run=args.dry_run)
            
#         else:
#             parser.print_help()
#             print("\n📋 EXAMPLES:")
#             print("  # Backfill specific range with averages")
#             print("  python backfill_missed_hrs.py --start 2026-01-20 --end 2026-01-23")
#             print("")
#             print("  # Backfill last 48 hours")
#             print("  python backfill_missed_hrs.py --last-hours 48")
#             print("")
#             print("  # Check for gaps and fill automatically")
#             print("  python backfill_missed_hrs.py --auto")
#             print("")
#             print("  # Dry run (test without inserting)")
#             print("  python backfill_missed_hrs.py --last-hours 24 --dry-run")
#             return 1
        
#         return 0
        
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1


# if __name__ == "__main__":
#     exit(main())



# """
# Backfill script for missed hours using SMART AVERAGES.
# Instead of failing on API calls, uses historical patterns to fill gaps.

# Usage:
#     # Backfill specific date range (using averages)
#     python backfill_missed_hrs.py --start "2026-01-20" --end "2026-01-23"
    
#     # Backfill last N hours
#     python backfill_missed_hrs.py --last-hours 24
    
#     # Check for gaps and backfill
#     python backfill_missed_hrs.py --auto
    
#     # Dry run (test without inserting)
#     python backfill_missed_hrs.py --last-hours 24 --dry-run
# """

# import argparse
# from datetime import datetime, timedelta
# from zoneinfo import ZoneInfo
# import pandas as pd
# import numpy as np
# import hopsworks
# from dotenv import load_dotenv
# import os
# import warnings
# warnings.filterwarnings('ignore')

# load_dotenv()

# BAHAWALPUR_LAT = 29.4
# BAHAWALPUR_LON = 71.68
# TIMEZONE = "Asia/Karachi"
# FEATURE_GROUP_NAME = "aqi_features_hourly"
# FEATURE_GROUP_VERSION = 1  # CHANGED FROM 4 TO 1


# def get_hopsworks_project():
#     """Connect to Hopsworks."""
#     project_name = os.getenv("HOPSWORKS_PROJECT")
#     api_key = os.getenv("HOPSWORKS_API_KEY")
#     if not project_name or not api_key:
#         raise RuntimeError("Environment variables not set")
#     return hopsworks.login(project=project_name, api_key_value=api_key)


# def get_existing_data(fg):
#     """Get all existing data from feature store."""
#     print("Loading existing data from Hopsworks...")
#     df = fg.read()
#     df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
#     df = df.sort_values('timestamp')
#     print(f"✓ Loaded {len(df)} historical rows")
#     print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
#     return df


# def calculate_smart_averages(existing_df, target_timestamps):
#     """
#     Calculate smart averages for target timestamps based on historical patterns.
    
#     Strategy:
#     1. Hour-of-week patterns (Monday 9AM similar to previous Mondays 9AM)
#     2. Recent trends (last 7 days)
#     3. Seasonal patterns (same month/year)
#     """
#     print(f"\nCalculating smart averages for {len(target_timestamps)} hours...")
    
#     # Prepare existing data
#     df = existing_df.copy()
#     df['hour_of_day'] = df['timestamp'].dt.hour
#     df['day_of_week'] = df['timestamp'].dt.dayofweek
#     df['hour_of_week'] = df['day_of_week'] * 24 + df['hour_of_day']
#     df['month'] = df['timestamp'].dt.month
    
#     # Numeric columns to fill - match train.py features plus the raw PM2.5 value
#     numeric_cols = [
#         'pm2_5',                # Raw PM2.5 value (for lag features)
#         'temperature_2m',       # Temperature
#         'relative_humidity_2m', # Humidity
#         'wind_speed_10m',       # Wind speed
#         'pressure_msl',         # Pressure
#     ]
    
#     # Only include columns that exist in the dataframe
#     numeric_cols = [col for col in numeric_cols if col in df.columns]
    
#     print(f"  Filling {len(numeric_cols)} numeric columns: {numeric_cols}")
    
#     # Create output DataFrame
#     filled_data = []
    
#     for target_ts in target_timestamps:
#         target_hour = target_ts.hour
#         target_dow = target_ts.dayofweek
#         target_month = target_ts.month
#         target_hour_of_week = target_dow * 24 + target_hour
        
#         # Initialize row with timestamp
#         row = {'timestamp': target_ts}
        
#         for col in numeric_cols:
#             # Strategy 1: Same hour-of-week pattern
#             same_hour_of_week_data = df[df['hour_of_week'] == target_hour_of_week]
            
#             if len(same_hour_of_week_data) >= 3:  # Need at least 3 samples
#                 # Use median (robust to outliers)
#                 row[col] = same_hour_of_week_data[col].median()
                
#                 # Add small random variation (5% noise)
#                 if not np.isnan(row[col]):
#                     noise = np.random.normal(0, 0.05)  # 5% standard deviation
#                     row[col] = row[col] * (1 + noise)
                    
#             else:
#                 # Strategy 2: Same hour of day across all days
#                 same_hour_data = df[df['hour_of_day'] == target_hour]
                
#                 if len(same_hour_data) >= 3:
#                     row[col] = same_hour_data[col].median()
#                 else:
#                     # Strategy 3: Overall median
#                     row[col] = df[col].median()
        
#         filled_data.append(row)
    
#     filled_df = pd.DataFrame(filled_data)
    
#     # Add small trend if we have recent data
#     if len(target_timestamps) > 1 and len(df) > 0:
#         # Calculate recent trend from last week
#         recent_cutoff = df['timestamp'].max() - timedelta(days=7)
#         recent_data = df[df['timestamp'] >= recent_cutoff]
        
#         if len(recent_data) > 24:  # At least a day of recent data
#             for col in numeric_cols:
#                 if col in recent_data.columns:
#                     recent_trend = recent_data[col].diff().mean()
                    
#                     if not np.isnan(recent_trend):
#                         # Apply trend linearly across filled hours
#                         for i in range(len(filled_df)):
#                             filled_df.loc[i, col] = filled_df.loc[i, col] + (recent_trend * i / 24)
    
#     print(f"✓ Generated averages with {len(filled_df)} rows")
#     return filled_df


# def build_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Build features from raw data - MUST MATCH train.py features!
#     """
#     df = df.copy()
    
#     # Ensure timezone
#     if df['timestamp'].dt.tz is None:
#         df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
#     else:
#         df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
#     # Temporal features - these are the features used in train.py
#     df["hour"] = df["timestamp"].dt.hour
#     df["day_of_week"] = df["timestamp"].dt.dayofweek
#     df["month"] = df["timestamp"].dt.month
    
#     # Lag features - MUST match train.py FEATURE_NAMES
#     df["pm25_lag1"] = df["pm2_5"].shift(1)
#     df["pm25_lag6"] = df["pm2_5"].shift(6)
#     df["pm25_lag24"] = df["pm2_5"].shift(24)
    
#     # Rolling features - MUST match train.py FEATURE_NAMES
#     df["pm25_ma6"] = df["pm2_5"].rolling(6, min_periods=1).mean()
#     df["pm25_ma24"] = df["pm2_5"].rolling(24, min_periods=1).mean()
    
#     # Change features - MUST match train.py FEATURE_NAMES
#     df["pm25_change_1hr"] = df["pm2_5"] - df["pm25_lag1"]
    
#     # Sort columns to match train.py order
#     feature_columns = [
#         'pm25_lag1', 'pm25_lag6', 'pm25_lag24',
#         'pm25_ma6', 'pm25_ma24', 'pm25_change_1hr',
#         'temperature_2m', 'relative_humidity_2m',
#         'wind_speed_10m', 'pressure_msl',
#         'hour', 'day_of_week', 'month'
#     ]
    
#     # Return only the features that exist in the dataframe
#     available_features = [col for col in feature_columns if col in df.columns]
    
#     # Make sure we include timestamp and pm2_5 as well
#     result_columns = ['timestamp', 'pm2_5'] + available_features
    
#     return df[result_columns]


# def find_missing_hours(fg) -> pd.DatetimeIndex:
#     """
#     Find missing hours in the feature store.
    
#     Returns:
#         DatetimeIndex of missing timestamps
#     """
#     print("\n=== Checking for missing hours ===")
    
#     try:
#         df = fg.read()
#         df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
#         # Get min and max timestamps
#         min_ts = df['timestamp'].min()
#         max_ts = df['timestamp'].max()
        
#         print(f"Data range: {min_ts} to {max_ts}")
#         print(f"Total hours in store: {len(df)}")
        
#         # Generate expected hourly range
#         expected = pd.date_range(start=min_ts, end=max_ts, freq='H')
#         print(f"Expected hours: {len(expected)}")
        
#         # Find missing
#         existing = set(df['timestamp'])
#         missing = [ts for ts in expected if ts not in existing]
        
#         if missing:
#             print(f"⚠️  Found {len(missing)} missing hours")
#             print(f"  First missing: {missing[0]}")
#             print(f"  Last missing: {missing[-1]}")
#             return pd.DatetimeIndex(missing)
#         else:
#             print("✓ No missing hours found")
#             return pd.DatetimeIndex([])
            
#     except Exception as e:
#         print(f"Error checking for gaps: {e}")
#         return pd.DatetimeIndex([])


# def backfill_with_averages(start_date: str, end_date: str, dry_run: bool = False):
#     """
#     Backfill missing data using smart historical averages.
#     No API calls needed!
    
#     Args:
#         start_date: Start date (YYYY-MM-DD)
#         end_date: End date (YYYY-MM-DD)
#         dry_run: If True, don't actually insert data
#     """
#     print("\n" + "="*70)
#     print("BACKFILL WITH SMART AVERAGES")
#     print("="*70)
#     print(f"Filling: {start_date} to {end_date}")
#     print("Using historical patterns instead of API calls")
    
#     # Connect to Hopsworks
#     project = get_hopsworks_project()
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
#     # 1. Get existing historical data
#     existing_df = get_existing_data(fg)
    
#     # 2. Generate timestamps for missing period
#     start_dt = pd.to_datetime(start_date).tz_localize('UTC')
#     end_dt = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(hours=23, minutes=59)
    
#     all_timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
    
#     # Check which timestamps are already present
#     existing_timestamps = set(existing_df['timestamp'])
#     missing_timestamps = [ts for ts in all_timestamps if ts not in existing_timestamps]
    
#     if not missing_timestamps:
#         print("✅ All hours already exist in feature store!")
#         return
    
#     print(f"\nFound {len(missing_timestamps)} missing hours to fill")
#     print(f"  From {missing_timestamps[0]} to {missing_timestamps[-1]}")
    
#     # 3. Generate filled data using smart averages
#     filled_raw_df = calculate_smart_averages(existing_df, missing_timestamps)
    
#     # 4. Build features - NOW MATCHES train.py
#     print("\nBuilding features for filled data...")
#     feature_df = build_features(filled_raw_df)
    
#     # Fill NaN values in lag/rolling features (they'll be NaN at boundaries)
#     feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
    
#     print(f"\n✓ Generated {len(feature_df)} filled rows")
#     print(f"  Shape: {feature_df.shape}")
    
#     # Show sample with train.py features
#     print("\nSample of filled data (first 3 rows):")
#     sample_cols = ['timestamp', 'pm2_5', 'temperature_2m', 'pm25_lag1', 'pm25_ma6', 'hour']
#     sample_cols = [col for col in sample_cols if col in feature_df.columns]
#     print(feature_df[sample_cols].head(3))
    
#     print("\nStatistics of filled data:")
#     for col in ['pm2_5', 'temperature_2m']:
#         if col in feature_df.columns:
#             print(f"  {col}: mean={feature_df[col].mean():.2f}, std={feature_df[col].std():.2f}")
    
#     # List all columns to verify they match train.py
#     print("\nFeatures generated (should match train.py FEATURE_NAMES):")
#     train_features = [
#         'pm25_lag1', 'pm25_lag6', 'pm25_lag24',
#         'pm25_ma6', 'pm25_ma24', 'pm25_change_1hr',
#         'temperature_2m', 'relative_humidity_2m',
#         'wind_speed_10m', 'pressure_msl',
#         'hour', 'day_of_week', 'month'
#     ]
    
#     for feature in train_features:
#         if feature in feature_df.columns:
#             print(f"  ✓ {feature}")
#         else:
#             print(f"  ✗ {feature} (MISSING!)")
    
#     # 5. Insert into Hopsworks (if not dry run)
#     if dry_run:
#         print("\n" + "="*70)
#         print("DRY RUN - NO DATA WILL BE INSERTED")
#         print("="*70)
#         print("To actually insert data, run without --dry-run flag")
#         return
    
#     print("\n" + "="*70)
#     print("INSERTING INTO HOPSWORKS")
#     print("="*70)
    
#     try:
#         fg.insert(feature_df, write_options={"wait_for_job": True})
#         print(f"✅ Successfully inserted {len(feature_df)} filled rows!")
#         print(f"✅ Used smart averaging based on {len(existing_df)} historical rows")
#         print(f"✅ Features match train.py requirements!")
#         print(f"✅ No external API calls needed!")
#     except Exception as e:
#         print(f"❌ Error inserting data: {e}")
#         raise
    
#     print("\n" + "="*70)
#     print("✅ BACKFILL WITH AVERAGES COMPLETED")
#     print("="*70)


# def backfill_last_n_hours(hours: int, dry_run: bool = False):
#     """
#     Backfill last N hours using averages.
#     """
#     now = datetime.now(ZoneInfo(TIMEZONE))
#     start = now - timedelta(hours=hours)
    
#     start_date = start.strftime("%Y-%m-%d")
#     end_date = now.strftime("%Y-%m-%d")
    
#     print(f"\nBackfilling last {hours} hours...")
#     print(f"From: {start_date} {start.hour:02d}:00")
#     print(f"To:   {end_date} {now.hour:02d}:00")
    
#     backfill_with_averages(start_date, end_date, dry_run)


# def auto_backfill(dry_run: bool = False):
#     """
#     Auto-detect gaps and fill them with averages.
#     """
#     print("\n" + "="*70)
#     print("AUTO-DETECT AND FILL GAPS WITH AVERAGES")
#     print("="*70)
    
#     # Connect to Hopsworks
#     project = get_hopsworks_project()
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
#     # Find missing hours
#     missing = find_missing_hours(fg)
    
#     if len(missing) == 0:
#         print("✅ No missing hours found.")
#         return
    
#     # Group missing hours into continuous ranges
#     missing_sorted = sorted(missing)
#     ranges = []
#     current_start = missing_sorted[0]
#     current_end = missing_sorted[0]
    
#     for ts in missing_sorted[1:]:
#         if ts == current_end + pd.Timedelta(hours=1):
#             current_end = ts
#         else:
#             ranges.append((current_start, current_end))
#             current_start = ts
#             current_end = ts
    
#     ranges.append((current_start, current_end))
    
#     print(f"\nFound {len(ranges)} gap(s):")
#     for i, (start, end) in enumerate(ranges, 1):
#         hours = (end - start).total_seconds() / 3600 + 1
#         print(f"  Gap {i}: {start.date()} {start.hour:02d}:00 to {end.date()} {end.hour:02d}:00 ({hours:.0f} hours)")
    
#     # Fill each gap
#     for start, end in ranges:
#         start_date = start.strftime("%Y-%m-%d")
#         end_date = end.strftime("%Y-%m-%d")
        
#         print(f"\n" + "-"*50)
#         print(f"Filling gap: {start_date} to {end_date}")
#         print("-"*50)
        
#         backfill_with_averages(start_date, end_date, dry_run)


# def main():
#     """Main entry point."""
#     parser = argparse.ArgumentParser(description="Backfill missed hours using smart averages")
#     parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
#     parser.add_argument("--end", help="End date (YYYY-MM-DD)")
#     parser.add_argument("--last-hours", type=int, help="Backfill last N hours")
#     parser.add_argument("--auto", action="store_true", help="Auto-detect and fill gaps")
#     parser.add_argument("--dry-run", action="store_true", help="Don't actually insert data")
    
#     args = parser.parse_args()
    
#     try:
#         if args.auto:
#             # Auto-detect and fill gaps
#             auto_backfill(dry_run=args.dry_run)
            
#         elif args.last_hours:
#             # Backfill last N hours
#             backfill_last_n_hours(args.last_hours, dry_run=args.dry_run)
            
#         elif args.start and args.end:
#             # Backfill specific range
#             backfill_with_averages(args.start, args.end, dry_run=args.dry_run)
            
#         else:
#             parser.print_help()
#             print("\n📋 EXAMPLES:")
#             print("  # Backfill specific range with averages")
#             print("  python backfill_missed_hrs.py --start 2026-01-20 --end 2026-01-23")
#             print("")
#             print("  # Backfill last 48 hours")
#             print("  python backfill_missed_hrs.py --last-hours 48")
#             print("")
#             print("  # Check for gaps and fill automatically")
#             print("  python backfill_missed_hrs.py --auto")
#             print("")
#             print("  # Dry run (test without inserting)")
#             print("  python backfill_missed_hrs.py --last-hours 24 --dry-run")
#             return 1
        
#         return 0
        
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1


# if __name__ == "__main__":
#     exit(main())


"""
Backfill script for missed hours using SMART AVERAGES.
Instead of failing on API calls, uses historical patterns to fill gaps.

Usage:
    # Backfill specific date range (using averages)
    python backfill_missed_hrs.py --start "2026-01-20" --end "2026-01-23"
    
    # Backfill last N hours
    python backfill_missed_hrs.py --last-hours 24
    
    # Check for gaps and backfill
    python backfill_missed_hrs.py --auto
    
    # Dry run (test without inserting)
    python backfill_missed_hrs.py --last-hours 24 --dry-run
"""

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

BAHAWALPUR_LAT = 29.4
BAHAWALPUR_LON = 71.68
TIMEZONE = "Asia/Karachi"
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1


def get_hopsworks_project():
    """Connect to Hopsworks."""
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project_name or not api_key:
        raise RuntimeError("Environment variables not set")
    return hopsworks.login(project=project_name, api_key_value=api_key)


def get_existing_data(fg):
    """Get all existing data from feature store."""
    print("Loading existing data from Hopsworks...")
    df = fg.read()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    print(f"✓ Loaded {len(df)} historical rows")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def calculate_smart_averages(existing_df, target_timestamps):
    """
    Calculate smart averages for target timestamps based on historical patterns.
    
    Strategy:
    1. Hour-of-week patterns (Monday 9AM similar to previous Mondays 9AM)
    2. Recent trends (last 7 days)
    3. Seasonal patterns (same month/year)
    """
    print(f"\nCalculating smart averages for {len(target_timestamps)} hours...")
    
    # Prepare existing data
    df = existing_df.copy()
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour_of_day']
    df['month'] = df['timestamp'].dt.month
    
    # ALL columns required by feature group v1 schema
    numeric_cols = [
        'pm2_5',                # Raw PM2.5 value
        'pm10',                 # PM10 (required by schema)
        'temperature_2m',       # Temperature
        'relative_humidity_2m', # Humidity
        'wind_speed_10m',       # Wind speed
        'pressure_msl',         # Pressure
        'precipitation',        # Precipitation (required by schema)
        'carbon_monoxide',      # CO (required by schema)
        'nitrogen_dioxide',     # NO2 (required by schema)
        'sulphur_dioxide',      # SO2 (required by schema)
        'ozone',                # O3 (required by schema)
    ]
    
    # Only include columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    print(f"  Filling {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    # Create output DataFrame
    filled_data = []
    
    for target_ts in target_timestamps:
        target_hour = target_ts.hour
        target_dow = target_ts.dayofweek
        target_month = target_ts.month
        target_hour_of_week = target_dow * 24 + target_hour
        
        # Initialize row with timestamp
        row = {'timestamp': target_ts}
        
        for col in numeric_cols:
            # Strategy 1: Same hour-of-week pattern
            same_hour_of_week_data = df[df['hour_of_week'] == target_hour_of_week]
            
            if len(same_hour_of_week_data) >= 3:  # Need at least 3 samples
                # Use median (robust to outliers)
                row[col] = same_hour_of_week_data[col].median()
                
                # Add small random variation (5% noise) for air quality metrics
                if not np.isnan(row[col]):
                    # More variation for air quality metrics
                    if col in ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']:
                        noise = np.random.normal(0, 0.08)  # 8% standard deviation for AQ
                    else:
                        noise = np.random.normal(0, 0.05)  # 5% standard deviation for weather
                    row[col] = row[col] * (1 + noise)
                    
            else:
                # Strategy 2: Same hour of day across all days
                same_hour_data = df[df['hour_of_day'] == target_hour]
                
                if len(same_hour_data) >= 3:
                    row[col] = same_hour_data[col].median()
                else:
                    # Strategy 3: Overall median
                    row[col] = df[col].median()
        
        filled_data.append(row)
    
    filled_df = pd.DataFrame(filled_data)
    
    # Add small trend if we have recent data
    if len(target_timestamps) > 1 and len(df) > 0:
        # Calculate recent trend from last week
        recent_cutoff = df['timestamp'].max() - timedelta(days=7)
        recent_data = df[df['timestamp'] >= recent_cutoff]
        
        if len(recent_data) > 24:  # At least a day of recent data
            for col in numeric_cols:
                if col in recent_data.columns:
                    recent_trend = recent_data[col].diff().mean()
                    
                    if not np.isnan(recent_trend):
                        # Apply trend linearly across filled hours
                        for i in range(len(filled_df)):
                            filled_df.loc[i, col] = filled_df.loc[i, col] + (recent_trend * i / 24)
    
    print(f"✓ Generated averages with {len(filled_df)} rows")
    return filled_df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from raw data - MUST include ALL columns from feature group schema.
    """
    df = df.copy()
    
    # Ensure timezone
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Temporal features - ALL required by feature group schema
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day  # REQUIRED by schema
    df["month"] = df["timestamp"].dt.month
    
    # Lag features
    df["pm25_lag1"] = df["pm2_5"].shift(1)
    df["pm25_lag6"] = df["pm2_5"].shift(6)
    df["pm25_lag24"] = df["pm2_5"].shift(24)
    
    # Rolling features
    df["pm25_ma6"] = df["pm2_5"].rolling(6, min_periods=1).mean()
    df["pm25_ma24"] = df["pm2_5"].rolling(24, min_periods=1).mean()
    
    # Change features
    df["pm25_change_1hr"] = df["pm2_5"] - df["pm25_lag1"]
    
    # Make sure we have all required columns from feature group schema
    # Add missing columns with NaN if they don't exist
    required_columns = [
        'timestamp', 'pm2_5', 'pm10', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'pressure_msl', 'precipitation', 'carbon_monoxide',
        'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'hour', 'day_of_week',
        'day', 'month', 'pm25_lag1', 'pm25_lag6', 'pm25_lag24', 'pm25_ma6',
        'pm25_ma24', 'pm25_change_1hr'
    ]
    
    # Check which columns are missing
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Adding missing columns: {missing_cols}")
        for col in missing_cols:
            if col == 'timestamp':
                continue
            # Use reasonable defaults
            if col in ['pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']:
                df[col] = 0.0  # Default low value for air quality
            elif col == 'precipitation':
                df[col] = 0.0  # Default no precipitation
            elif col == 'day':
                df[col] = df['timestamp'].dt.day
            else:
                df[col] = np.nan
    
    # Ensure all required columns exist
    return df


def find_missing_hours(fg) -> pd.DatetimeIndex:
    """
    Find missing hours in the feature store.
    
    Returns:
        DatetimeIndex of missing timestamps
    """
    print("\n=== Checking for missing hours ===")
    
    try:
        df = fg.read()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Get min and max timestamps
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        
        print(f"Data range: {min_ts} to {max_ts}")
        print(f"Total hours in store: {len(df)}")
        
        # Generate expected hourly range
        expected = pd.date_range(start=min_ts, end=max_ts, freq='H')
        print(f"Expected hours: {len(expected)}")
        
        # Find missing
        existing = set(df['timestamp'])
        missing = [ts for ts in expected if ts not in existing]
        
        if missing:
            print(f"⚠️  Found {len(missing)} missing hours")
            print(f"  First missing: {missing[0]}")
            print(f"  Last missing: {missing[-1]}")
            return pd.DatetimeIndex(missing)
        else:
            print("✓ No missing hours found")
            return pd.DatetimeIndex([])
            
    except Exception as e:
        print(f"Error checking for gaps: {e}")
        return pd.DatetimeIndex([])


def backfill_with_averages(start_date: str, end_date: str, dry_run: bool = False):
    """
    Backfill missing data using smart historical averages.
    No API calls needed!
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dry_run: If True, don't actually insert data
    """
    print("\n" + "="*70)
    print("BACKFILL WITH SMART AVERAGES")
    print("="*70)
    print(f"Filling: {start_date} to {end_date}")
    print("Using historical patterns instead of API calls")
    
    # Connect to Hopsworks
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    # 1. Get existing historical data
    existing_df = get_existing_data(fg)
    
    # 2. Generate timestamps for missing period
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(hours=23, minutes=59)
    
    all_timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
    
    # Check which timestamps are already present
    existing_timestamps = set(existing_df['timestamp'])
    missing_timestamps = [ts for ts in all_timestamps if ts not in existing_timestamps]
    
    if not missing_timestamps:
        print("✅ All hours already exist in feature store!")
        return
    
    print(f"\nFound {len(missing_timestamps)} missing hours to fill")
    print(f"  From {missing_timestamps[0]} to {missing_timestamps[-1]}")
    
    # 3. Generate filled data using smart averages
    filled_raw_df = calculate_smart_averages(existing_df, missing_timestamps)
    
    # 4. Build features - MUST include ALL schema columns
    print("\nBuilding features for filled data...")
    feature_df = build_features(filled_raw_df)
    
    # Fill NaN values
    feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"\n✓ Generated {len(feature_df)} filled rows")
    print(f"  Shape: {feature_df.shape}")
    
    # Show sample
    print("\nSample of filled data (first 3 rows):")
    sample_cols = ['timestamp', 'pm2_5', 'pm10', 'temperature_2m', 'precipitation', 'hour', 'day']
    sample_cols = [col for col in sample_cols if col in feature_df.columns]
    print(feature_df[sample_cols].head(3))
    
    print("\nStatistics of filled data:")
    stat_cols = ['pm2_5', 'pm10', 'temperature_2m', 'precipitation']
    for col in stat_cols:
        if col in feature_df.columns:
            print(f"  {col}: mean={feature_df[col].mean():.2f}, std={feature_df[col].std():.2f}")
    
    # Verify all required schema columns are present
    print("\nVerifying all required schema columns are present:")
    schema_required = [
        'pm2_5', 'pm10', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'pressure_msl', 'precipitation', 'carbon_monoxide',
        'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'hour', 'day_of_week',
        'day', 'month'
    ]
    
    missing_schema = []
    for col in schema_required:
        if col in feature_df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (MISSING!)")
            missing_schema.append(col)
    
    if missing_schema:
        print(f"\n❌ ERROR: Missing schema columns: {missing_schema}")
        if not dry_run:
            print("Cannot insert data with missing schema columns.")
            return
    
    # 5. Insert into Hopsworks (if not dry run)
    if dry_run:
        print("\n" + "="*70)
        print("DRY RUN - NO DATA WILL BE INSERTED")
        print("="*70)
        print("To actually insert data, run without --dry-run flag")
        return
    
    print("\n" + "="*70)
    print("INSERTING INTO HOPSWORKS")
    print("="*70)
    
    try:
        fg.insert(feature_df, write_options={"wait_for_job": True})
        print(f"✅ Successfully inserted {len(feature_df)} filled rows!")
        print(f"✅ Used smart averaging based on {len(existing_df)} historical rows")
        print(f"✅ All schema columns included!")
        print(f"✅ No external API calls needed!")
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        raise
    
    print("\n" + "="*70)
    print("✅ BACKFILL WITH AVERAGES COMPLETED")
    print("="*70)


def backfill_last_n_hours(hours: int, dry_run: bool = False):
    """
    Backfill last N hours using averages.
    """
    now = datetime.now(ZoneInfo(TIMEZONE))
    start = now - timedelta(hours=hours)
    
    start_date = start.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    
    print(f"\nBackfilling last {hours} hours...")
    print(f"From: {start_date} {start.hour:02d}:00")
    print(f"To:   {end_date} {now.hour:02d}:00")
    
    backfill_with_averages(start_date, end_date, dry_run)


def auto_backfill(dry_run: bool = False):
    """
    Auto-detect gaps and fill them with averages.
    """
    print("\n" + "="*70)
    print("AUTO-DETECT AND FILL GAPS WITH AVERAGES")
    print("="*70)
    
    # Connect to Hopsworks
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    # Find missing hours
    missing = find_missing_hours(fg)
    
    if len(missing) == 0:
        print("✅ No missing hours found.")
        return
    
    # Group missing hours into continuous ranges
    missing_sorted = sorted(missing)
    ranges = []
    current_start = missing_sorted[0]
    current_end = missing_sorted[0]
    
    for ts in missing_sorted[1:]:
        if ts == current_end + pd.Timedelta(hours=1):
            current_end = ts
        else:
            ranges.append((current_start, current_end))
            current_start = ts
            current_end = ts
    
    ranges.append((current_start, current_end))
    
    print(f"\nFound {len(ranges)} gap(s):")
    for i, (start, end) in enumerate(ranges, 1):
        hours = (end - start).total_seconds() / 3600 + 1
        print(f"  Gap {i}: {start.date()} {start.hour:02d}:00 to {end.date()} {end.hour:02d}:00 ({hours:.0f} hours)")
    
    # Fill each gap
    for start, end in ranges:
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
        
        print(f"\n" + "-"*50)
        print(f"Filling gap: {start_date} to {end_date}")
        print("-"*50)
        
        backfill_with_averages(start_date, end_date, dry_run)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backfill missed hours using smart averages")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--last-hours", type=int, help="Backfill last N hours")
    parser.add_argument("--auto", action="store_true", help="Auto-detect and fill gaps")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually insert data")
    
    args = parser.parse_args()
    
    try:
        if args.auto:
            # Auto-detect and fill gaps
            auto_backfill(dry_run=args.dry_run)
            
        elif args.last_hours:
            # Backfill last N hours
            backfill_last_n_hours(args.last_hours, dry_run=args.dry_run)
            
        elif args.start and args.end:
            # Backfill specific range
            backfill_with_averages(args.start, args.end, dry_run=args.dry_run)
            
        else:
            parser.print_help()
            print("\n📋 EXAMPLES:")
            print("  # Backfill specific range with averages")
            print("  python backfill_missed_hrs.py --start 2026-01-20 --end 2026-01-23")
            print("")
            print("  # Backfill last 48 hours")
            print("  python backfill_missed_hrs.py --last-hours 48")
            print("")
            print("  # Check for gaps and fill automatically")
            print("  python backfill_missed_hrs.py --auto")
            print("")
            print("  # Dry run (test without inserting)")
            print("  python backfill_missed_hrs.py --last-hours 24 --dry-run")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())