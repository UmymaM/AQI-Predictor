# **AQI Predictor**

## **Prediction Dashboard**:
[https://aqi-predictor-bwp.streamlit.app/](https://aqi-predictor-bwp.streamlit.app/)

## **Overview**

This project implements an **end-to-end machine learning system** to forecast **Air Quality Index (AQI)** by predicting **PM2.5 concentration levels** for the next **24, 48, and 72 hours** in **Bahawalpur, Pakistan**.
The system is built on a **100% serverless architecture** and trains *multiple regression models* to generate **short-term air quality forecasts**. Predicted PM2.5 values are subsequently converted into **standardized AQI categories** for user-friendly interpretation.

---

## **Key Objectives**

- Build a reliable **data ingestion pipeline** for weather and air-pollution data  
- Engineer meaningful **temporal and lag-based features** for time-series forecasting  
- Train and evaluate **multiple regression models** for short-term AQI prediction  
- Apply **SHAP analysis** to explain model predictions and feature importance  
- Store features and models using **Hopsworks Feature Store** and **Model Registry**  
- Automate data updates and retraining using **GitHub Actions**  
- Provide an interactive **Streamlit dashboard** for visualization and forecasts  

---

## **Problem Framing**

- **Problem Type:** Time-series regression  
- **Target Variable:** `PM2.5 concentration (µg/m³)`  
- **Forecast Horizons:** `+24h`, `+48h`, `+72h`  

### **Why PM2.5?**

**PM2.5** is the *most health-critical air pollutant* and serves as the **primary driver** for AQI calculations.
Rather than predicting AQI directly, the system predicts **PM2.5**, which is later converted into AQI using **standard PM2.5-to-AQI conversion formulas**.

---

## **System Architecture Overview**

### **1. Data Ingestion**

Hourly data is collected from:

- **Open-Meteo** — historical + current data

**Collected variables include:**

- PM2.5 concentration  
- Temperature  
- Humidity  
- Wind speed  
- Surface pressure  

---

### **2. Feature Engineering**

Raw data is transformed into **ML-ready features**, including:

- **Time-based features:** `hour`, `day_of_week`, `day_of_month`, `month`  
- **Lag features:** previous PM2.5 values (`1h`, `6h`, `24h`)  
- **Rolling statistics:** moving averages and short-term trends  
- **Change metrics:** hour-to-hour PM2.5 variation  
- **Future targets:** created using time shifts for each forecast horizon  

---

### **3. Feature Store (Hopsworks)**

All engineered features are stored in a **Hopsworks Feature Group**, enabling:

- Consistent feature reuse for **training and inference**  
- **Versioned feature management**  
- Improved **reproducibility and traceability**  

---

### **4. Model Training and Evaluation**

For each prediction horizon (`24h`, `48h`, `72h`), multiple models are trained:

- **Linear baseline:** Ridge Regression  
- **Tree-based ensembles:** Random Forest, Gradient Boosting  

**Evaluation metrics include:**

- `RMSE`  
- `MAE`  
- `R² Score`  

The **best-performing model** for each horizon is selected and versioned.

---

### **5. Explainability and Analysis**

**SHAP (SHapley Additive exPlanations)** is used to:

- Identify *influential features*  
- Validate feature relevance  
- Support model selection decisions  

Exploratory Data Analysis (**EDA**) and visualizations are performed throughout development to analyze **trends, seasonality, and anomalies**.

---

### **6. Model Registry and Automation**

Selected models are:

1. Saved in the **Hopsworks Model Registry**  
2. **Version-controlled** for traceability  

**GitHub Actions** automate:

- Periodic data ingestion  
- Feature rebuilding  
- Model retraining when sufficient new data is available  

---

### **7. Frontend Application**

A **Streamlit-based dashboard** provides:

- Historical **PM2.5 and AQI visualizations**  
- Short-term **air quality forecasts**  
- AQI category indicators with **color coding**  
- Model **explainability insights**  

---

## **Final Outcome**

The result is a **production-ready air quality forecasting system** that integrates:

> data ingestion → feature engineering → model training → explainability → automation → visualization

into a **scalable, maintainable, and reproducible pipeline**.
