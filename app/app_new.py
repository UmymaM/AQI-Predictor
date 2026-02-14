import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import joblib
import hopsworks
from dotenv import load_dotenv
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from zoneinfo import ZoneInfo

MODEL_NAME = "aqi_predictor_unified"
MODEL_VERSION = 1
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1
HORIZONS = [24, 48, 72]


# these features work fine for v1 of the model
BASE_FEATURES = [
    "pm25_lag1", "pm25_lag6", "pm25_lag24",
    "pm25_ma6", "pm25_ma24", "pm25_change_1hr",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "pressure_msl",
    "hour", "day_of_week", "day", "month",
]
# for later versions, add "pm10","carbon_monoxide","nitrogen_dioxide" and"sulphur_dioxide"
load_dotenv()

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Metric containers */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #555 !important;
    }
    
    /* Headers */
    h1 {
    
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h3 {
        color: #333 !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: 700;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 6px solid #667eea;
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Animation for cards */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .element-container {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)


def pm25_to_aqi(pm25):
    # converts pm25 to aqi
    if pm25 <= 12.0:
        return ((50 - 0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0
    elif pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    elif pm25 <= 250.4:
        return ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    else:
        return 500


def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "#00e400", "🟢", "Air quality is excellent"
    elif aqi <= 100:
        return "Moderate", "#ffff00", "🟡", "Acceptable air quality"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "🟠", "Sensitive groups should limit outdoor exposure"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000", "🔴", "Everyone should reduce outdoor activity"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97", "🟣", "Health alert! Avoid outdoor activity"
    else:
        return "Hazardous", "#7e0023", "🟤", "Emergency conditions - stay indoors!"


def create_gauge_chart(value, title, max_value=300):
    aqi = pm25_to_aqi(value)
    category, color, emoji, _ = get_aqi_info(aqi)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br><span style='font-size:0.7em'>{emoji} {category}</span>", 
               'font': {'size': 20, 'color': '#333'}},
        delta={'reference': 100, 'increasing': {'color': "#ff0000"}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 12], 'color': '#e8f5e9'},
                {'range': [12, 35.4], 'color': '#fff9c4'},
                {'range': [35.4, 55.4], 'color': '#ffe0b2'},
                {'range': [55.4, 150.4], 'color': '#ffcdd2'},
                {'range': [150.4, max_value], 'color': '#f3e5f5'}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 5},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig


def create_forecast_chart(current_pm25, predictions, current_time):
    # creates forecast timeline
    times = [current_time]
    values = [current_pm25]
    labels = ["Now"]
    
    for i, h in enumerate(HORIZONS):
        times.append(current_time + timedelta(hours=h))
        values.append(predictions[i])
        labels.append(f"+{h}h")
    
    # Get colors based on AQI
    colors = []
    for val in values:
        aqi = pm25_to_aqi(val)
        _, color, _, _ = get_aqi_info(aqi)
        colors.append(color)
    
    fig = go.Figure()
    
    # Add line
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode='lines+markers',
        name='PM2.5 Forecast',
        line=dict(color='#667eea', width=4, shape='spline'),
        marker=dict(
            size=20,
            color=colors,
            line=dict(color='white', width=3),
            symbol='circle'
        ),
        text=labels,
        hovertemplate='<b>%{text}</b><br>PM2.5: %{y:.1f} µg/m³<br>%{x|%b %d, %H:%M}<extra></extra>'
    ))
    
    # Add threshold zones
    fig.add_hrect(y0=0, y1=12, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=12, y1=35.4, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=35.4, y1=55.4, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=55.4, y1=150.4, fillcolor="red", opacity=0.1, line_width=0)
    
    # Add threshold lines
    fig.add_hline(y=12, line_dash="dash", line_color="green", opacity=0.6,
                  annotation_text="Good", annotation_position="right")
    fig.add_hline(y=35.4, line_dash="dash", line_color="orange", opacity=0.6,
                  annotation_text="Moderate", annotation_position="right")
    fig.add_hline(y=55.4, line_dash="dash", line_color="red", opacity=0.6,
                  annotation_text="Unhealthy", annotation_position="right")
    
    fig.update_layout(
        title={
            'text': "72-Hour PM2.5 Forecast",
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        xaxis_title="Time",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
    )
    
    return fig


def create_comparison_bar(current_pm25, predictions):
    #creates comparison bar chart
    categories = ["Current", "+24h", "+48h", "+72h"]
    values = [current_pm25] + list(predictions)
    
    colors = []
    for val in values:
        aqi = pm25_to_aqi(val)
        _, color, _, _ = get_aqi_info(aqi)
        colors.append(color)
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f"{v:.1f}" for v in values],
        textposition='inside',
        textfont=dict(size=14, color='#333', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
    ))
    
    fig.update_layout(
        title="PM2.5 Comparison",
        yaxis_title="PM2.5 (µg/m³)",
        template='plotly_white',
        height=450,
        font=dict(size=14),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


@st.cache_resource(show_spinner=False)
def get_hopsworks_project():
    return hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

@st.cache_resource(show_spinner=False, ttl=None)
def load_model():
    # load best model based on highest r2 scores
    project = get_hopsworks_project()
    mr = project.get_model_registry()
    all_models = mr.get_models(MODEL_NAME)
    
    if not all_models:
        return None, None, None, None
    
    # Find model with highest R² score
    best_model = None
    best_r2 = -float('inf')
    
    for model in all_models:
        r2 = model.training_metrics.get('r2', -float('inf'))
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    if best_model is None:
        return None, None, None, None
    
    # Download and load the best model
    model_dir = best_model.download()
    model_obj = joblib.load(os.path.join(model_dir, "model.pkl"))
    
    # Load feature names
    with open(os.path.join(model_dir, "features.json")) as f:
        feature_names = json.load(f)["feature_names"]
    
    # Load metrics
    metrics = best_model.training_metrics.copy()
    
    return model_obj, feature_names, metrics, best_model.version

def aqi_fact_box(current_aqi):
    if current_aqi <= 50:
        risk = "Minimal health risk for the general population."
    elif current_aqi <= 100:
        risk = "Acceptable, but unusually sensitive individuals may experience discomfort."
    elif current_aqi <= 150:
        risk = "Sensitive groups may experience respiratory symptoms."
    elif current_aqi <= 200:
        risk = "Increased likelihood of adverse health effects for everyone."
    else:
        risk = "Serious health effects possible. Emergency-level pollution."

    st.info(
        f"""
        **📘 AQI Insight**

        • **AQI (Air Quality Index)** converts pollutant concentration into a health-based scale (0–500).  
        • **PM2.5** particles are **~30× smaller than a human hair** and can penetrate deep into the lungs and bloodstream.  
        • Epidemiological studies show that **every +10 µg/m³ increase in PM2.5 raises mortality risk by ~6–8%**.  

        """
    )



@st.cache_data(show_spinner=False, ttl=300)
def load_latest_features():
    # Load latest features - cached for 5 minutes
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION )
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    return df.iloc[-1]  # latest row only


def main():
    # Header with animation
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.title("🌀 Air Quality Prediction Dashboard")
    st.markdown("### **Real-Time PM2.5 Forecasting for Bahawalpur**")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # sidebar
    with st.sidebar:
        st.markdown("## 📍 Location")
        st.markdown("**City:** Bahawalpur, Pakistan")
        st.markdown("**Coordinates:** 29.40°N, 71.68°E")
        st.markdown("---")
        st.markdown("## ⏰ Current Time")
        current_time = datetime.now(ZoneInfo("Asia/Karachi"))
        st.markdown(f"**{current_time.strftime('%I:%M %p')}**")
        st.markdown(f"{current_time.strftime('%B %d, %Y')}")
        st.markdown("---")
        st.markdown("## 🎯 AQI Scale")
        aqi_info = [
            ("Good", "0-50", "🟢"),
            ("Moderate", "51-100", "🟡"),
            ("Unhealthy (Sensitive)", "101-150", "🟠"),
            ("Unhealthy", "151-200", "🔴"),
            ("Very Unhealthy", "201-300", "🟣"),
            ("Hazardous", "301+", "🟤")]
        
        for category, range_val, emoji in aqi_info:
            st.markdown(f"{emoji} **{category}**: {range_val}")
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # prediction button
    predict_button = st.button("🔮 Generate 3-Day Forecast", use_container_width=True, type="primary")
    
    if predict_button:
        # Progress bar for loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load model
        status_text.text("⚡ Loading AI model...")
        progress_bar.progress(30)
        model, feature_names, metrics, model_version = load_model()
        
        # Load features
        status_text.text("📊 Fetching latest data...")
        progress_bar.progress(60)
        latest_row = load_latest_features()
        
        # Make predictions
        status_text.text("🧠 Generating predictions...")
        progress_bar.progress(90)
        
        rows = []
        for h in HORIZONS:
            row = [latest_row[f] for f in BASE_FEATURES]
            row.append(h)
            rows.append(row)
        
        X = np.array(rows)
        predictions = model.predict(X)
        
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        # current aqi
        st.markdown("## 📍 Current Air Quality")
        
        current_pm25 = float(latest_row['pm2_5'])
        current_aqi = pm25_to_aqi(current_pm25)
        category, color, emoji, description = get_aqi_info(current_aqi)

        
        # Alert box
        if current_aqi <= 50:
            st.success(f"{emoji} **{category}** - {description}")
        elif current_aqi <= 100:
            st.info(f"{emoji} **{category}** - {description}")
        elif current_aqi <= 150:
            st.warning(f"{emoji} **{category}** - {description}")
        else:
            st.error(f"{emoji} **{category}** - {description}")
        
        # Current metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("PM2.5", f"{current_pm25:.1f} µg/m³")
        
        with col2:
            st.metric("AQI", f"{current_aqi:.0f}")
        
        with col3:
            st.metric("Temperature", f"{latest_row['temperature_2m']:.1f}°C")
        
        with col4:
            st.metric("Humidity", f"{latest_row['relative_humidity_2m']:.0f}%")

        aqi_fact_box(current_aqi)    
        
        st.markdown("---")
        
        # forecast gauges
        st.markdown("## 🔮 72-Hour AQI Forecast")
        
        gauge_cols = st.columns(3)
        
        for i, h in enumerate(HORIZONS):
            with gauge_cols[i]:
                pred_time = latest_row['timestamp'] + timedelta(hours=h)
                fig = create_gauge_chart(predictions[i], f"+{h} Hours<br>{pred_time.strftime('%b %d, %H:00')} UTC")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(
        """
        ℹ️The **Δ (delta)** shown on each gauge represents how far the predicted air quality
        deviates from the **upper limit of a “Good” air quality level**.
        
        • **Reference value:** PM2.5 = **35.4 µg/m³** (AQI ≈ 100)  
        """
        )

        st.markdown("---")
        
        # charts
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            # Forecast timeline
            fig_timeline = create_forecast_chart(current_pm25, predictions, latest_row['timestamp'])
            st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
        
        with chart_col2:
            # Comparison bar
            fig_bar = create_comparison_bar(current_pm25, predictions)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        # forecast table
        st.markdown("## 📊 Detailed Forecast")
        
        forecast_data = []
        for i, h in enumerate(HORIZONS):
            pred_time = latest_row['timestamp'] + timedelta(hours=h)
            pred_pm25 = predictions[i]
            pred_aqi = pm25_to_aqi(pred_pm25)
            cat, _, emoji, _ = get_aqi_info(pred_aqi)
            
            forecast_data.append({
                "Forecast Time": pred_time.strftime("%b %d, %Y %H:%M"),
                "Horizon": f"+{h} hours",
                "PM2.5 (µg/m³)": f"{pred_pm25:.2f}",
                "AQI": f"{pred_aqi:.0f}",
                "Category": f"{emoji} {cat}",
                "Change": f"{pred_pm25 - current_pm25:+.1f}"
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # current conditions
        st.markdown("## 🌤️ Current Weather Conditions")
        
        weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
        
        with weather_col1:
            st.metric("🌡️ Temperature", f"{latest_row['temperature_2m']:.1f}°C")
            st.metric("💨 Wind Speed", f"{latest_row['wind_speed_10m']:.1f} m/s")
        
        with weather_col2:
            st.metric("💧 Humidity", f"{latest_row['relative_humidity_2m']:.0f}%")
            st.metric("🔽 Pressure", f"{latest_row['pressure_msl']:.0f} hPa")
        
        with weather_col3:
            st.metric("📊 PM2.5 (1h ago)", f"{latest_row['pm25_lag1']:.1f} µg/m³")
            st.metric("📊 PM2.5 (6h avg)", f"{latest_row['pm25_ma6']:.1f} µg/m³")
        
        with weather_col4:
            st.metric("📊 PM2.5 (24h avg)", f"{latest_row['pm25_ma24']:.1f} µg/m³")
            st.metric("📈 1h Change", f"{latest_row['pm25_change_1hr']:+.1f} µg/m³")
        
        st.markdown("---")
        
        # model info
        with st.expander("🤖 Model Information"):
            model_col1, model_col2 = st.columns(2)
            
            with model_col1:
                st.markdown("**Model Details:**")
                st.write(f"- **Name:** {MODEL_NAME}")
                st.write(f"- **Version:** {model_version}")
                st.write(f"- **Horizons:** 24h, 48h, 72h")
                st.write(f"- **RMSE:** {metrics.get('rmse', 'N/A'):.2f}")
            
            # with model_col2:
            #     st.markdown("**Performance Metrics:**")
            #     st.write(f"- **RMSE:** {metrics.get('rmse', 'N/A'):.2f}")
            #     st.write(f"- **MAE:** {metrics.get('mae', 'N/A'):.2f}")
            #     st.write(f"- **R² Score:** {metrics.get('r2', 'N/A'):.3f}")
        
        st.markdown("---")
        
        footer_col1, footer_col2 = st.columns(2)
    
        with footer_col1:
            st.markdown("**🕐 Last Updated**")
            st.write(latest_row['timestamp'].strftime("%Y-%m-%d %H:%M UTC"))
        
        with footer_col2:
            st.markdown("**⚡ Data Source**")
            st.write("Hopsworks Feature Store")
        
    else:
        # landing page before prediction
        st.markdown("""
            
        <div style='text-align: center; padding: 3rem; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
            <h2 margin-bottom: 1rem;'>Welcome to the Bahawalpur AQI Prediction Dashboard</h2>
            <p style='font-size: 1.2rem; color: #666; line-height: 1.8;'>
                Get AQI forecasts powered by machine learning.
                <br><br>
                Click the <strong>"Generate 3-Day Forecast"</strong> button above to see:
            </p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-top: 2rem;'>
                <div>
                    <h3 style='color: #667eea;'>📊 Real-Time Data</h3>
                    <p style='color: #888;'>Current PM2.5 & AQI levels</p>
                </div>
                <div>
                    <h3 style='color: #667eea;'>🔮 AI Predictions</h3>
                    <p style='color: #888;'>24h, 48h & 72h forecasts</p>
                </div>
                <div>
                    <h3 style='color: #667eea;'>📈 Visual Charts</h3>
                    <p style='color: #888;'>Interactive graphs & gauges</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()
