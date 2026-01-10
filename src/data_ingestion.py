#In this file, i will be ingesting data from openmeteos API
# the data comprises two months of historical weather+pollutant data for Bahawalpur
#the fetched data will be used to train the models

import pandas as pd
import openmeteo_requests
from pathlib import Path

basePath="data/raw"
lat, lon = 29.3978, 71.6752

def fetch2MonthsWeatherData(lat,lon):
    print("entering weather function")

    openmeteo = openmeteo_requests.Client()

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2025-11-02",
        "end_date": "2026-01-02",
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pressure_msl", "wind_direction_10m", "precipitation", "cloud_cover_low"],
    }
    print("Calling the weather api")
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(4).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"timestamp": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low

    weather_dataframe = pd.DataFrame(data = hourly_data)
    print("\nHourly data\n", weather_dataframe)
    return weather_dataframe

def fetch2MonthsPollutantData(lat,lon):

    openmeteo = openmeteo_requests.Client()

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "carbon_dioxide"],
        "past_days": 61,
    }
    print("Calling the air quality api")
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
    hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
    hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
    hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
    hourly_carbon_dioxide = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"timestamp": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["pm10"] = hourly_pm10
    hourly_data["pm2_5"] = hourly_pm2_5
    hourly_data["carbon_monoxide"] = hourly_carbon_monoxide
    hourly_data["nitrogen_dioxide"] = hourly_nitrogen_dioxide
    hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide
    hourly_data["ozone"] = hourly_ozone
    hourly_data["carbon_dioxide"] = hourly_carbon_dioxide

    pollutants_dataframe = pd.DataFrame(data = hourly_data)
    print(f"Fetched Pollutant Data:  {len(pollutants_dataframe)} rows")
    return pollutants_dataframe

# def ensureDirExists(path):
#     Path(path).mkdir(parents=True,exist_ok=True)

# def saveToParquet(df,path):
#     Path(path).parent.mkdir(parents=True,exist_ok=True)
#     df.to_parquet(path,index=False)        


# def mergeDataframes(df1,df2):

# def mergeDataframes(weather_df, pollutants_df):

#     # Ensure timestamps are datetime and UTC
#     weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
#     pollutants_df["timestamp"] = pd.to_datetime(pollutants_df["timestamp"], utc=True)

#     # Sort (important for time series)
#     weather_df = weather_df.sort_values("timestamp")
#     pollutants_df = pollutants_df.sort_values("timestamp")

#     # Merge on timestamp
#     merged_df = pd.merge(
#         weather_df,
#         pollutants_df,
#         on="timestamp",
#         how="inner"
#     )

#     print(f"Merged dataframe shape: {merged_df.shape}")
#     return merged_df

def ingest() -> pd.DataFrame:
    weather = fetch2MonthsWeatherData(lat, lon)
    pollutants = fetch2MonthsPollutantData(lat, lon)

    df = (
        weather
        .merge(pollutants, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# def main():

#     print("main function started!")
#     lat, lon = 29.3978, 71.6752
#     print("will start fetching data")
#     print("testing a theory")


#     weather_df=fetch2MonthsWeatherData(lat,lon)
#     pollutants_df=fetch2MonthsPollutantData(lat,lon)

#     # weather_df.to_csv(path=f"{basePath}/weather_data_bwp.csv")
#     # pollutants_df.to_csv(f"{basePath}/pollutant_data_bwp.csv")
#     # saveToParquet(weather_df,path=f"{basePath}/weather_data_bwp.parquet")
#     # saveToParquet(pollutants_df,path=f"{basePath}/pollutants_data_bwp.parquet")

#     merged_df=mergeDataframes(weather_df,pollutants_df)
#     saveToParquet(merged_df,path=f"{basePath}/merged_data_bwp.parquet")
#     merged_df.to_csv(path_or_buf=f"{basePath}/merged_data_bwp.csv")
    

#     print("Running test script:")
#     weather_df.to_csv("test.csv", index=False)


#     print("Data Ingested Successfully!!!! :D")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print("ERROR:", e)