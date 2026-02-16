import pandas as pd
import openmeteo_requests

lat, lon = 29.3978, 71.6752

def buildHourlyDf(hourly):
    timestamps=pd.date_range(
        start=pd.to_datetime(hourly.Time(),unit="s",utc=True),
        end=pd.to_datetime(hourly.TimeEnd(),unit="s",utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    return timestamps

def fetchHistoricalWeatherData(start_date,end_date):
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

    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

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
    return weather_dataframe

def fetchHistoricalPollutantData(start_date,end_date):

    openmeteo = openmeteo_requests.Client()

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "carbon_dioxide"],
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

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

    return pollutants_dataframe


def fetch_historical(start_date: str, end_date: str) -> pd.DataFrame:
    weather = fetchHistoricalWeatherData(start_date,end_date)
    pollutants = fetchHistoricalPollutantData(start_date,end_date)

    df = (
        weather
        .merge(pollutants, on="timestamp", how="inner")
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df
