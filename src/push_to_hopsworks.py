# # data ingested from api and saved in parquets
# # eda has been performed, irrelevant features dropped
# # processed data saved in a parquet
# # pushing features to the feature store now
 
# import os
# import hopsworks
# import pandas as pd
# from dotenv import load_dotenv

# from features import build_features
# from data_ingestion import fetch_recent,fetch_historical


# load_dotenv()

# featuresPath="data\\processed\\aqi_features_bwp.parquet"
# featuresGroupName="aqi_features_hourly"
# featuresGroupVersion=1
# MODE = os.getenv("INGEST_MODE", "incremental").lower()
# # incremental is the default mode

# def main():
  
#     #getting hopsworks project name and api key from env   

#     project_name=os.getenv("HOPSWORKS_PROJECT")
#     api_key=os.getenv("HOPSWORKS_API_KEY")

#     if not project_name or not api_key:
#         raise RuntimeError("Project or key not set")
    
# #  ingesting + feature engineering
#     # raw_df=ingest()
#     # feature_df=build_features(raw_df)

#     # TODO
#     # add daily data to hopsworks 


#     # logging in
#     project=hopsworks.login(
#         project=project_name,
#         api_key_value=api_key
#     )

#     # feature store
#     fs=project.get_feature_store()

#     # feature group
#     fg = fs.get_or_create_feature_group(
#         name=featuresGroupName,
#         version=featuresGroupVersion,
#         primary_key=["timestamp"],
#         event_time="timestamp",
#         description="Hourly AQI features for Bahawalpur",
#         online_enabled=False,
#     )

#     if MODE == "historical":
#         print("Running historical backfill")
#         raw_df = fetch_historical("2025-11-02", "2026-01-15")
#     else:
#         print("Fetching recent data")
#         # Get last timestamp from feature store
#         try:
#             existing_df = fg.read()
#             last_ts = existing_df["timestamp"].max()
#         except Exception:
#             last_ts = None
#         raw_df = fetch_recent(last_timestamp=last_ts)

#     if raw_df.empty:
#         print(" No data fetched")
#         return

#     feature_df=build_features(raw_df)   


#  `  # ensuring timestamp integrity
#     feature_df["timestamp"]=pd.to_datetime(feature_df["timestamp"],utc=True)

#     if feature_df["timestamp"].duplicated().any():
#         raise ValueError("Duplicate timestamps detected")
#     feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)

#     # inserting features
#     fg.insert(feature_df, write_options={"wait_for_job": True})

#     print(f"Inserted {len(feature_df)} rows into {featuresGroupName} v{featuresGroupVersion}")


#     # #initializing hsfs connection
#     # project=hopsworks.login(
#     #     project=project_name,
#     #     api_key_value=api_key
#     # )
#     # # print(project)
#     # project=hopsworks.get_current_project()
#     # print("printing current project:",project.name)
        
#     # #connecting to hospworks feature store
#     # fs=project.get_feature_store()

#     # #creating feature group
#     # fg=fs.get_or_create_feature_group(
#     #             name=featuresGroupName,
#     #             version=featuresGroupVersion,
#     #             primary_key=["timestamp"],
#     #             event_time="timestamp",
#     #             description="Hourly AQI features for BWP",
#     #             online_enabled=False
#     #         )

#     # # hopsworks doesnt support timestamps as pk in online enabled versions

#     # fg.insert(df,write_options={"wait_for_job":True})
#     # print(f"Successfullyb inserted {len(df)} rows into {featuresGroupName} v{featuresGroupVersion}")

    

       
    



# if __name__ == "__main__":
#     main()    
#     # Successfully pushed features to hopsworks!


import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from features import build_features
from data_ingestion import fetch_historical, fetch_recent

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 3
# MODE = os.getenv("INGEST_MODE", "incremental").lower()  # "historical" or "incremental"
MODE= "historical"

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
    if MODE == "historical":
        print("Running historical backfill")
        raw_df = fetch_historical("2025-05-02", "2026-01-19")
        # chaging date from november 2 to august 2--> CHANEG ALTER
    else:
        print("Fetching recent incremental data")
        try:
            existing_df = fg.read()
            last_ts = existing_df["timestamp"].max()
        except Exception:
            last_ts = None
        raw_df = fetch_recent(last_timestamp=last_ts)

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
