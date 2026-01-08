# data ingested from api and saved in parquets
# eda has been performed, irrelevant features dropped
# processed data saved in a parquet
# pushing features to the feature store now
 
import os
import hopsworks
from zoneinfo import ZoneInfo
import pandas as pd
from hsfs import connection as hsfs_connection

featuresPath="data\\processed\\aqi_features_bwp.parquet"
featuresGroupName="aqi_features_hourly"
featuresGroupVersion=1

def main():
    if not(os.path.exists(featuresPath)):
        raise FileNotFoundError("Invalid path.")
    df=pd.read_parquet(featuresPath)
    
    # #normalizing timestamps
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)

    #checking for duplicate timestamps
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")
    df=df.sort_values("timestamp").reset_index(drop=True)
    
    # replacing generic object-tyoed columns w more specific
    # data types:
    df=df.infer_objects(copy=False)
    print(df.describe)





if __name__ == "__main__":
    main()    