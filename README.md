# **AQI Predictor — Bahawalpur, Pakistan**

**Multi-horizon PM2.5 forecasting system with 24h, 48h, and 72h predictions**

## **Overview**

This project implements an **end-to-end machine learning system** to forecast **PM2.5 concentrations** for the next **24, 48, and 72 hours** in **Bahawalpur, Pakistan**, and converts them into **standardized AQI categories**.

The system is designed as a **cloud-native, automated ML pipeline**, covering:
data ingestion → feature engineering → model training → explainability → deployment → visualization.

---

## **Problem Framing**

- **Problem Type:** Time-series regression  
- **Target Variable:** `PM2.5 concentration (µg/m³)`  
- **Forecast Horizons:** `+24h`, `+48h`, `+72h`  

### **Why PM2.5?**
PM2.5 is the **most health-critical air pollutant** and the primary driver of AQI.
Instead of predicting AQI directly, the system predicts **PM2.5**, which is then converted to AQI using **standard PM2.5 → AQI formulas**.

---

## **Data Overview**

- **Granularity:** Hourly
- **Dataset Size:** ~6,900 rows
- **Location:** Bahawalpur, Pakistan
- **Source:** Open-Meteo (air quality + weather data)

---

## **Modeling Strategy**

This project uses a **direct multi-horizon regression approach**:

- Separate future targets are created using time shifts:
  - `target_24h`
  - `target_48h`
  - `target_72h`
- Each horizon is learned **explicitly**, avoiding recursive error accumulation.
- The approach supports **multi-output regression pipelines** while maintaining horizon-specific evaluation.

**Why direct multi-horizon forecasting?**
- More stable than recursive forecasting
- Explicit control over each prediction horizon

---

## **Feature Engineering**

The following feature groups are used:

### **PM2.5 Temporal Features**
- `pm25_lag1`, `pm25_lag6`, `pm25_lag24`
- `pm25_ma6`, `pm25_ma24`
- `pm25_change_1hr`

### **Weather Features**
- `temperature_2m`
- `relative_humidity_2m`
- `wind_speed_10m`
- `pressure_msl`

### **Time-Based Features**
- `hour`
- `day_of_week`
- `day`
- `month`

### **Dropped Features (Low Predictive Value)**
Based on EDA:
- `precipitation`
- `cloud_cover_low`
- `wind_direction_low`
- `month` (retained only for analysis)

---

## **Exploratory Data Analysis (EDA)**

EDA confirms that:
- PM2.5 exhibits **strong short-term autocorrelation**
- Predictability **decreases with horizon** (24h → 72h)
- Lag features dominate short-term forecasts
- Weather features gain importance at longer horizons

---

## **Feature Store and Model Registry**

- All engineered features are stored in **Hopsworks Feature Store**
- Trained models are versioned using **Hopsworks Model Registry**
- Ensures:
  - Reproducibility
  - Consistent training/inference features
  - Traceable model versions

---

## **Explainability**

**SHAP (SHapley Additive Explanations)** is used to:
- Identify influential features
- Validate feature engineering decisions
- Interpret horizon-specific behavior

---

## **Automation**

The system is fully automated using **GitHub Actions**:

- **Hourly:** data ingestion and feature updates
- **Daily:** model retraining at **21:00 UTC**
- Secrets managed via GitHub repository settings

---

## **Frontend Application**

A **Streamlit dashboard** provides:
- Historical PM2.5 and AQI trends
- 24h / 48h / 72h forecasts
- AQI category visualization with color indicators
- Interactive plots for interpretation

---

## **Tech Stack**

| Component | Technology |
|--------|-----------|
| Data Source | Open-Meteo API |
| Feature Store | Hopsworks |
| Model Registry | Hopsworks |
| Models | Ridge, Tree-based Ensembles |
| Explainability | SHAP |
| Automation | GitHub Actions |
| Dashboard | Streamlit |
| Hosting | Streamlit Cloud |

---


## **Final Outcome**

This project delivers a **production-oriented air quality forecasting system** that integrates:

> data ingestion → feature engineering → multi-horizon forecasting → explainability → automation → visualization

into a **scalable, maintainable, and reproducible ML pipeline**.

---

