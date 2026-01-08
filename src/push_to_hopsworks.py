# data ingested from api and saved in parquets
# eda has been performed, irrelevant features dropped
# processed data saved in a parquet
# pushing features to the feature store now
 
import os
import hopsworks
from zoneinfo import ZoneInfo
import pandas as pd
from hsfs import connection as hsfs_connection
from dotenv import load_dotenv

load_dotenv()

featuresPath="data\\processed\\aqi_features_bwp.parquet"
featuresGroupName="aqi_features_hourly"
featuresGroupVersion=1

def main():
    if not(os.path.exists(featuresPath)):
        raise FileNotFoundError("Invalid path.")
    df=pd.read_parquet(featuresPath)
    
    # #normalizing timestamps
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)

    #checking for duplicate timestampss
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")
    df=df.sort_values("timestamp").reset_index(drop=True)
    
    # replacing generic object-tyoed columns w more specific
    # data types:
    df=df.infer_objects(copy=False)

    print("Project:", os.getenv("HOPSWORKS_PROJECT"))
    print("Key loaded:", bool(os.getenv("HOPSWORKS_API_KEY")))
    
    #getting hopsworks project name and api key from env   
    project_name=os.getenv("HOPSWORKS_PROJECT")
    api_key=os.getenv("HOPSWORKS_API_KEY")

    if not project_name or not api_key:
        raise RuntimeError("Project or key not set")

    #initializing hsfs connection
    project=hopsworks.login(
        project=project_name,
        api_key_value=api_key
    )
    # print(project)
    project=hopsworks.get_current_project()
    print("printing current project:",project.name)
        
    #connecting to hospworks feature store
    fs=project.get_feature_store()

    #creating feature group
    fg=fs.get_or_create_feature_group(
                name=featuresGroupName,
                version=featuresGroupVersion,
                primary_key=["timestamp"],
                event_time="timestamp",
                description="Hourly AQI features for BWP",
                online_enabled=False
            )

    # hopsworks doesnt support timestamps as pk in online enabled versions

    fg.insert(df,write_options={"wait_for_job":True})
    print(f"Successfullyb inserted {len(df)} rows into {featuresGroupName} v{featuresGroupVersion}")

    

       
    



if __name__ == "__main__":
    main()    
    # Successfully pushed features to hopsworks!