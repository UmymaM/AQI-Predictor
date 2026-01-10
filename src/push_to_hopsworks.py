# data ingested from api and saved in parquets
# eda has been performed, irrelevant features dropped
# processed data saved in a parquet
# pushing features to the feature store now
 
import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

from data_ingestion import ingest
from features import build_features

load_dotenv()

featuresPath="data\\processed\\aqi_features_bwp.parquet"
featuresGroupName="aqi_features_hourly"
featuresGroupVersion=2

def main():
    # if not(os.path.exists(featuresPath)):
    #     raise FileNotFoundError("Invalid path.")
    # df=pd.read_parquet(featuresPath)
    
    # #normalizing timestamps
    # df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)

    #checking for duplicate timestampss
    # if df["timestamp"].duplicated().any():
    #     raise ValueError("Duplicate timestamps detected")
    # df=df.sort_values("timestamp").reset_index(drop=True)
    
    # replacing generic object-tyoed columns w more specific
    # # data types:
    # df=df.infer_objects(copy=False)

    # print("Project:", os.getenv("HOPSWORKS_PROJECT"))
    # print("Key loaded:", bool(os.getenv("HOPSWORKS_API_KEY")))
    
    #getting hopsworks project name and api key from env   
    project_name=os.getenv("HOPSWORKS_PROJECT")
    api_key=os.getenv("HOPSWORKS_API_KEY")

    if not project_name or not api_key:
        raise RuntimeError("Project or key not set")
    
#  ingesting + feature engineering
    raw_df=ingest()
    feature_df=build_features(raw_df)

    # ensuring timestamp integrity
    feature_df["timestamp"]=pd.to_datetime(feature_df["timestamp"],utc=True)

    if feature_df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")
    feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)

    # logging in
    project=hopsworks.login(
        project=project_name,
        api_key_value=api_key
    )
    # feature store
    fs=project.get_feature_store()

    # feature group
    fg = fs.get_or_create_feature_group(
        name=featuresGroupName,
        version=featuresGroupVersion,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features for Bahawalpur",
        online_enabled=False,
    )

    # inserting features
    fg.insert(feature_df, write_options={"wait_for_job": True})

    print(f"Inserted {len(feature_df)} rows into {featuresGroupName} v{featuresGroupVersion}")


    # #initializing hsfs connection
    # project=hopsworks.login(
    #     project=project_name,
    #     api_key_value=api_key
    # )
    # # print(project)
    # project=hopsworks.get_current_project()
    # print("printing current project:",project.name)
        
    # #connecting to hospworks feature store
    # fs=project.get_feature_store()

    # #creating feature group
    # fg=fs.get_or_create_feature_group(
    #             name=featuresGroupName,
    #             version=featuresGroupVersion,
    #             primary_key=["timestamp"],
    #             event_time="timestamp",
    #             description="Hourly AQI features for BWP",
    #             online_enabled=False
    #         )

    # # hopsworks doesnt support timestamps as pk in online enabled versions

    # fg.insert(df,write_options={"wait_for_job":True})
    # print(f"Successfullyb inserted {len(df)} rows into {featuresGroupName} v{featuresGroupVersion}")

    

       
    



if __name__ == "__main__":
    main()    
    # Successfully pushed features to hopsworks!