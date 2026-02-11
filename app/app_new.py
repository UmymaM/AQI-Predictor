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

MODEL_NAME = "aqi_predictor_unified"
MODEL_VERSION = 1
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1
HORIZONS = [24, 48, 72]

BASE_FEATURES = [
    "pm25_lag1", "pm25_lag6", "pm25_lag24",
    "pm25_ma6", "pm25_ma24", "pm25_change_1hr",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "pressure_msl",
    "hour", "day_of_week", "day", "month",
]

load_dotenv()

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS that works well in both modes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.2rem !important;
        margin-top: 0.5rem !important;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        padding: 1.5rem 1rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .element-container {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    [data-testid="stSidebar"] strong {
        color: white !important;
        font-weight: 600;
    }
    
    /* Info/Alert boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1.25rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Dataframe styling */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        padding: 1rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        font-weight: 700 !important;
        padding: 1rem !important;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }
    
    .streamlit-expanderContent {
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 1.5rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Welcome card */
    .welcome-card {
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid rgba(102, 126, 234, 0.2);
        background: rgba(102, 126, 234, 0.05);
    }
    
    .welcome-card h2 {
        color: #667eea;
        margin-bottom: 1.5rem;
        font-size: 2rem;
    }
    
    .welcome-card p {
        font-size: 1.15rem;
        line-height: 1.8;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 2rem;
        margin-top: 2.5rem;
    }
    
    .feature-item {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        line-height: 1.5;
        opacity: 0.7;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
        }
        
        .main-header p {
            font-size: 1rem !important;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def pm25_to_aqi(pm25):
    """Convert PM2.5 to AQI"""
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
    """Get AQI category information"""
    if aqi <= 50:
        return "Good", "#10b981", "🟢", "Air quality is excellent"
    elif aqi <= 100:
        return "Moderate", "#f59e0b", "🟡", "Acceptable air quality"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#f97316", "🟠", "Sensitive groups should limit outdoor exposure"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "🔴", "Everyone should reduce outdoor activity"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7", "🟣", "Health alert! Avoid outdoor activity"
    else:
        return "Hazardous", "#991b1b", "🟤", "Emergency conditions - stay indoors!"


def create_gauge_chart(value, title, max_value=300):
    """Create modern gauge chart that adapts to theme"""
    aqi = pm25_to_aqi(value)
    category, color, emoji, _ = get_aqi_info(aqi)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{title}</b><br><span style='font-size:0.85em; opacity:0.7'>{emoji} {category}</span>",
            'font': {'size': 16, 'family': 'Inter'}
        },
        number={'font': {'size': 48, 'family': 'Inter', 'weight': 900}, 'suffix': ''},
        gauge={
            'axis': {
                'range': [None, max_value],
                'tickwidth': 2,
                'tickfont': {'size': 12}
            },
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "rgba(100, 100, 100, 0.1)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [50, 100], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [100, 150], 'color': 'rgba(249, 115, 22, 0.15)'},
                {'range': [150, 200], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [200, max_value], 'color': 'rgba(168, 85, 247, 0.15)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': aqi
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )
    
    return fig


def create_forecast_chart(current_pm25, predictions, current_time):
    """Create forecast timeline chart"""
    times = [current_time]
    values = [current_pm25]
    labels = ["Now"]
    
    for i, h in enumerate(HORIZONS):
        times.append(current_time + timedelta(hours=h))
        values.append(predictions[i])
        labels.append(f"+{h}h")
    
    colors = []
    for val in values:
        aqi = pm25_to_aqi(val)
        _, color, _, _ = get_aqi_info(aqi)
        colors.append(color)
    
    fig = go.Figure()
    
    # Add shaded zones
    fig.add_hrect(y0=0, y1=12, fillcolor="#10b981", opacity=0.08, line_width=0, 
                  annotation_text="Good", annotation_position="inside top left",
                  annotation=dict(font=dict(size=11, color="#10b981")))
    fig.add_hrect(y0=12, y1=35.4, fillcolor="#f59e0b", opacity=0.08, line_width=0,
                  annotation_text="Moderate", annotation_position="inside top left",
                  annotation=dict(font=dict(size=11, color="#f59e0b")))
    fig.add_hrect(y0=35.4, y1=55.4, fillcolor="#f97316", opacity=0.08, line_width=0,
                  annotation_text="Unhealthy (Sensitive)", annotation_position="inside top left",
                  annotation=dict(font=dict(size=11, color="#f97316")))
    fig.add_hrect(y0=55.4, y1=150.4, fillcolor="#ef4444", opacity=0.08, line_width=0,
                  annotation_text="Unhealthy", annotation_position="inside top left",
                  annotation=dict(font=dict(size=11, color="#ef4444")))
    
    # Add line
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode='lines+markers',
        name='PM2.5 Forecast',
        line=dict(color='#667eea', width=4, shape='spline'),
        marker=dict(
            size=16,
            color=colors,
            line=dict(color='rgba(255,255,255,0.8)', width=3),
            symbol='circle'
        ),
        text=labels,
        hovertemplate='<b style="font-size:14px">%{text}</b><br>' +
                      'PM2.5: <b>%{y:.1f} µg/m³</b><br>' +
                      '%{x|%b %d, %I:%M %p}<extra></extra>',
    ))
    
    fig.update_layout(
        title={
            'text': "<b>72-Hour PM2.5 Forecast Timeline</b>",
            'font': {'size': 20, 'family': 'Inter'},
            'x': 0.02
        },
        xaxis_title="<b>Time</b>",
        yaxis_title="<b>PM2.5 Concentration (µg/m³)</b>",
        hovermode='x unified',
        height=420,
        font=dict(size=13, family='Inter'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    
    return fig


def create_comparison_bar(current_pm25, predictions):
    """Create comparison bar chart"""
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
            line=dict(color='rgba(255,255,255,0.3)', width=2),
        ),
        text=[f"<b>{v:.1f}</b>" for v in values],
        textposition='outside',
        textfont=dict(size=15, family='Inter', weight='bold'),
        hovertemplate='<b>%{x}</b><br>PM2.5: <b>%{y:.1f} µg/m³</b><extra></extra>',
    ))
    
    fig.update_layout(
        title={
            'text': "<b>PM2.5 Comparison</b>",
            'font': {'size': 18, 'family': 'Inter'},
            'x': 0.02
        },
        yaxis_title="<b>PM2.5 (µg/m³)</b>",
        height=380,
        font=dict(size=13, family='Inter'),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    
    return fig


# OPTIMIZED: Better caching strategy
@st.cache_resource(show_spinner=False)
def get_hopsworks_project():
    """Connect to Hopsworks - cached permanently"""
    return hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )


# OPTIMIZED: Load model once and cache forever
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model from Hopsworks - cached permanently"""
    with st.spinner("🔄 Loading AI model (one-time setup)..."):
        project = get_hopsworks_project()
        mr = project.get_model_registry()
        model = mr.get_model(MODEL_NAME, MODEL_VERSION)
        model_dir = model.download()
        model_obj = joblib.load(os.path.join(model_dir, "model.pkl"))

        with open(os.path.join(model_dir, "features.json")) as f:
            feature_names = json.load(f)["feature_names"]
        
        metrics = {}
        metrics_file = os.path.join(model_dir, "detailed_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)

        return model_obj, feature_names, model.training_metrics, model.version


# OPTIMIZED: Cache features for 5 minutes, fetch ONLY latest row
@st.cache_data(show_spinner=False, ttl=300)
def load_latest_features():
    """Load ONLY the latest feature row - cached for 5 minutes"""
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )
    
    # OPTIMIZED: Use limit=1 to fetch only latest row
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    # df = fg.read()
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#     df = df.sort_values("timestamp")
    return df.iloc[-1]


def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header animate-in">
        <h1>🌫️ Air Quality Prediction Dashboard</h1>
        <p>Real-Time PM2.5 Forecasting for Bahawalpur, Pakistan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📍 Location")
        st.markdown("**City:** Bahawalpur, Pakistan")
        st.markdown("**Coordinates:** 29.40°N, 71.68°E")
        
        st.markdown("## ⏰ Current Time")
        current_time = datetime.now()
        st.markdown(f"**{current_time.strftime('%I:%M %p')}**")
        st.markdown(f"{current_time.strftime('%B %d, %Y')}")
        
        st.markdown("## 🎯 AQI Scale")
        aqi_info = [
            ("Good", "0-50", "🟢"),
            ("Moderate", "51-100", "🟡"),
            ("Unhealthy (Sensitive)", "101-150", "🟠"),
            ("Unhealthy", "151-200", "🔴"),
            ("Very Unhealthy", "201-300", "🟣"),
            ("Hazardous", "301+", "🟤")
        ]
        
        for category, range_val, emoji in aqi_info:
            st.markdown(f"{emoji} **{category}**: {range_val}")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # OPTIMIZED: Load model on app startup (not on button click)
    # This happens once when the app starts
    try:
        model, feature_names, metrics, model_version = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure your Hopsworks credentials are correct in your .env file")
        return
    
    # Main prediction button
    predict_button = st.button("🔮 Generate 3-Day Forecast", use_container_width=True, type="primary")
    
    if predict_button:
        # OPTIMIZED: Much faster data loading
        with st.spinner("📊 Fetching latest data..."):
            try:
                latest_row = load_latest_features()
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                return
        
        # Make predictions
        with st.spinner("🧠 Generating predictions..."):
            rows = []
            for h in HORIZONS:
                row = [latest_row[f] for f in BASE_FEATURES]
                row.append(h)
                rows.append(row)
            
            X = np.array(rows)
            predictions = model.predict(X)
        
        # Current air quality
        st.markdown('<h2 class="section-header">📍 Current Air Quality</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        current_pm25 = float(latest_row['pm2_5'])
        current_aqi = pm25_to_aqi(current_pm25)
        category, color, emoji, description = get_aqi_info(current_aqi)
        
        # Alert box
        if current_aqi <= 50:
            st.success(f"### {emoji} **{category}**\n{description}")
        elif current_aqi <= 100:
            st.info(f"### {emoji} **{category}**\n{description}")
        elif current_aqi <= 150:
            st.warning(f"### {emoji} **{category}**\n{description}")
        else:
            st.error(f"### {emoji} **{category}**\n{description}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Current metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("PM2.5 Level", f"{current_pm25:.1f} µg/m³")
        
        with col2:
            st.metric("AQI Index", f"{current_aqi:.0f}")
        
        with col3:
            st.metric("Temperature", f"{latest_row['temperature_2m']:.1f}°C")
        
        with col4:
            st.metric("Humidity", f"{latest_row['relative_humidity_2m']:.0f}%")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Forecast section
        st.markdown('<h2 class="section-header">🔮 72-Hour AQI Forecast</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        gauge_cols = st.columns(3)
        
        for i, h in enumerate(HORIZONS):
            with gauge_cols[i]:
                pred_time = latest_row['timestamp'] + timedelta(hours=h)
                fig = create_gauge_chart(
                    predictions[i],
                    f"+{h} Hours<br>{pred_time.strftime('%b %d, %H:00')}"
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Charts section
        st.markdown('<h2 class="section-header">📈 Forecast Visualization</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            fig_timeline = create_forecast_chart(current_pm25, predictions, latest_row['timestamp'])
            st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
        
        with chart_col2:
            fig_bar = create_comparison_bar(current_pm25, predictions)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Detailed forecast table
        st.markdown('<h2 class="section-header">📊 Detailed Forecast Data</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        forecast_data = []
        for i, h in enumerate(HORIZONS):
            pred_time = latest_row['timestamp'] + timedelta(hours=h)
            pred_pm25 = predictions[i]
            pred_aqi = pm25_to_aqi(pred_pm25)
            cat, _, emoji, _ = get_aqi_info(pred_aqi)
            
            forecast_data.append({
                "📅 Forecast Time": pred_time.strftime("%b %d, %Y %H:%M"),
                "⏱️ Horizon": f"+{h} hours",
                "💨 PM2.5": f"{pred_pm25:.2f} µg/m³",
                "📊 AQI": f"{pred_aqi:.0f}",
                "🏷️ Category": f"{emoji} {cat}",
                "📈 Change": f"{pred_pm25 - current_pm25:+.1f} µg/m³"
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Weather conditions
        st.markdown('<h2 class="section-header">🌤️ Current Weather Conditions</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
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
            change = latest_row['pm25_change_1hr']
            st.metric("📈 1h Change", f"{change:+.1f} µg/m³", delta=f"{change:+.1f}")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Model information
        with st.expander("🤖 Model Information & Performance", expanded=False):
            model_col1, model_col2 = st.columns(2)
            
            with model_col1:
                st.markdown("**Model Details:**")
                st.write(f"- **Name:** {MODEL_NAME}")
                st.write(f"- **Version:** {model_version}")
                st.write(f"- **Type:** Multi-Horizon Regressor")
                st.write(f"- **Prediction Horizons:** 24h, 48h, 72h")
            
            with model_col2:
                st.markdown("**Performance Metrics:**")
                if metrics:
                    st.write(f"- **RMSE:** {metrics.get('rmse', 'N/A'):.2f}")
                    st.write(f"- **MAE:** {metrics.get('mae', 'N/A'):.2f}")
                    st.write(f"- **R² Score:** {metrics.get('r2', 'N/A'):.3f}")
                else:
                    st.write("Metrics not available")
        
        # Footer
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.markdown("**🕐 Last Updated**")
            st.write(latest_row['timestamp'].strftime("%Y-%m-%d %H:%M UTC"))
        
        with footer_col2:
            st.markdown("**⚡ Data Source**")
            st.write("Hopsworks Feature Store")
        
        with footer_col3:
            st.markdown("**🔄 Refresh Rate**")
            st.write("5 minutes (cached)")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="welcome-card">
            <h2>Welcome to the AQI Prediction Dashboard</h2>
            <p>
                Get accurate air quality forecasts powered by machine learning.
                <br>
                Click the <strong>"Generate 3-Day Forecast"</strong> button above to begin.
            </p>
            
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()