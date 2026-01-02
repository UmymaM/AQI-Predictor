# AQI-Predictor
This project focuses on building an end-to-end machine learning system to forecast AQI by predicting PM2.5 concentration levels for the next 24, 48, and 72 hours for Bahawalpur. This project is built with a 100% serverless stack and trains multiple models to predict pm2.5 values. 

The system ingests hourly pollutant and weather data from reliable public APIs, transforms this data into structured, ML-ready features, and trains multiple time-series regression models to forecast future PM2.5 values. The predicted PM2.5 concentrations are then converted into Air Quality Index (AQI) categories for user-friendly interpretation.

Key Objectives:

1. Build a reliable data ingestion pipeline for weather and air-pollution data
2. Engineer meaningful temporal and lag-based features for time-series forecasting
3. Train and evaluate multiple regression models for short-term air quality prediction
4. Use SHAP analysis to explain model predictions and justify feature importance
5. Store features and models using Hopsworks Feature Store and Model Registry
6. Automate data updates and retraining using GitHub Actions
7. Provide an interactive Streamlit dashboard for visualization and forecasts

Problem Framing:

Type: Time-series regression
Target variable: PM2.5 concentration (µg/m³)
Output horizons: +24h, +48h, +72h

Why PM2.5:
PM2.5 is the most health-critical pollutant and serves as the primary driver for AQI calculations.
AQI values are derived after prediction using standardized PM2.5-to-AQI conversion formulas rather than being directly predicted.

System Architecture Overview:

1. Data Ingestion

Hourly data is collected from:
OpenAQ (primary source for PM2.5)
Open-Meteo (weather and fallback air quality)

Collected variables include:

PM2.5 concentration, Temperature, Humidity, Wind speed and Surface pressure.
The data is aligned on a common UTC hourly timestamp and stored in Parquet format.

2. Feature Engineering

The raw data is transformed into ML-ready features, including:
1) Time-based features: hour, day of week, day of month, month
2) Lag features: previous PM2.5 values (1h, 6h, 24h)
3) Rolling statistics: moving averages and short-term trends
4) Change metrics: hour-to-hour PM2.5 variation
5) Future target labels are created for each forecast horizon using time shifts.

3. Feature Store (Hopsworks)

All engineered features are stored in a Hopsworks Feature Group, enabling:
Consistent training, inference feature reuse and versioned feature management

4. Model Training and Evaluation

For each prediction horizon (24h, 48h, 72h), multiple models are trained, including:

1) Linear baseline (Ridge Regression)
2) Tree-based ensemble models (Random Forest, Gradient Boosting)

Models are evaluated using regression metrics such as:

RMSE, MAE and R² score
The best-performing model for each horizon is selected and versioned.

5. Explainability and Analysis

SHAP (SHapley Additive exPlanations) is used to identify indluential features, validate feature relevance and support model selection decision. Exploratory Data Analysis (EDA) and visualizations are performed throughout development to understand trends, seasonality, and anomalies.

6. Model Registry and Automation

The selected models are:

1) Saved in the Hopsworks Model Registry
2) Version-controlled for traceability

Automated workflows using GitHub Actions:

Periodically ingest new data
Rebuild features
Retrain models when sufficient new data is available

7. Frontend Application

A Streamlit-based dashboard provides:
Historical PM2.5 and AQI visualizations
Short-term air quality forecasts
AQI category indicators with color coding
Model explainability insights

Final Outcome

The result is a production-ready air quality forecasting system. 
