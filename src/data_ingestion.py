import pandas as pd
import openmeteo_requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

lat, lon = 29.3978, 71.6752

def buildHourlyDf(hourly):
    timestamps=pd.date_range(
        start=pd.to_datetime(hourly.Time(),unit="s",utc=True),
        end=pd.to_datetime(hourly.TimeEnd(),unit="s",utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    return timestamps

def fetch2MonthsWeatherData(start_date,end_date):
    print("entering weather function")

    openmeteo = openmeteo_requests.Client()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", 
            "relative_humidity_2m", 
            "wind_speed_10m",
            "pressure_msl", 
            "wind_direction_10m",
            "precipitation", 
            "cloud_cover_low"],
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

    hourly_data = {"timestamp": buildHourlyDf(hourly)}

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

def fetch2MonthsPollutantData(start_date,end_date):

    openmeteo = openmeteo_requests.Client()

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "carbon_dioxide"],
    }
    print("Calling the air quality api")
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
    hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
    hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
    hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
    hourly_carbon_dioxide = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"timestamp": buildHourlyDf(hourly)}

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


def fetch_historical(start_date: str, end_date: str) -> pd.DataFrame:
    weather = fetch2MonthsWeatherData(start_date,end_date)
    pollutants = fetch2MonthsPollutantData(start_date,end_date)

    df = (
        weather
        .merge(pollutants, on="timestamp", how="inner")
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def fetch_recent(last_timestamp: pd.Timestamp = None) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = last_timestamp + timedelta(seconds=1) if last_timestamp else now - timedelta(hours=24)
    end = now
    return fetch_historical(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
