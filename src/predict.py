import os
import json
import joblib
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from zoneinfo import ZoneInfo

load_dotenv()

MODELS_DIR = "models/latest"
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1
HORIZONS = [24, 48, 72]

def get_hopsworks_project():
    """Connect to Hopsworks project."""
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not project_name or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT or HOPSWORKS_API_KEY not set in .env")
    
    print(f"Connecting to Hopsworks project: {project_name}")
    return hopsworks.login(project=project_name, api_key_value=api_key)

def load_latest_features():
    # load recent features from hopsworks
    print("\n=== Loading latest features from Hopsworks ===")
    
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        
        # Read all data and get the latest row
        df = fg.read()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp')
        
        # Get the most recent entry
        latest = df.iloc[[-1]].copy()  # Keep as DataFrame with one row
        
        print(f"Latest data timestamp: {latest['timestamp'].iloc[0]}")
        print(f"Current PM2.5: {latest['pm2_5'].iloc[0]:.2f} µg/m³")
        
        return latest
        
    except Exception as e:
        raise RuntimeError(f"Error loading features from Hopsworks: {e}")

def load_trained_models(project):

    # load the latest version of the best performing model from model registry
    print("\n=== Loading trained models ===")
    try:
        mr=project.get_model_registry()
        models={}
        for h in HORIZONS:
            model_name=f"aqi_predictor_t{h}h"
            try:
                print(f"Loading model: {model_name}")

                model_list = mr.get_models(name=model_name)
                if not model_list:
                    print(f"  ⚠️  No models found for {model_name}")
                    continue
                py_model = max(model_list, key=lambda x: x.version)
                # py_model=mr.get_model(name=model_name,version=5)

                print(f"Version: {py_model.version}")
                print(f"Created: {py_model.created}")
        
                model_dir = py_model.download()
                
                # Load the actual model object
                model_file = os.path.join(model_dir, "model.pkl")
                model = joblib.load(model_file)
                
                models[h] = (model, py_model.name)
                # models[h] = {
                #      "sk_model": model,
                #         "hs_model": py_model,   # keep metadata
                #     }
                print(f"  ✓ Loaded successfully, Version={py_model.version}")
                
            except Exception as e:
                print(f"  ⚠️  Failed to load {model_name}: {e}")
                continue 

        if not models:
            raise RuntimeError("No models could be loaded from registry!")
        
        print(f"\n✓ Loaded {len(models)} models from Model Registry")
        
        feature_names = ["pm25_lag1","pm25_lag6","pm25_lag24","pm25_ma6","pm25_ma24",
            "pm25_change_1hr","temperature_2m","relative_humidity_2m","wind_speed_10m","pressure_msl",
            "hour","day_of_week", "month",]
        
        print(f"\nUsing {len(feature_names)} features")
        
        return models, feature_names       

    except Exception as e:
        print(f"Failed to load {model_name} from the registry ")    


def make_predictions(latest_features, models, feature_names):
    # makes predictions for each horizon
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)
    
    # Prepare features
    X = latest_features[feature_names].values
    
    print(f"\nInput features shape: {X.shape}")
    print(f"Timestamp: {latest_features['timestamp'].iloc[0]}")
    
    predictions = {}
    
    for h in HORIZONS:
        if h not in models:
            print(f"\n⚠️  Skipping {h}h forecast - no model available")
            continue
        model, model_name = models[h]
        
        # Make prediction
        pred = model.predict(X)[0]
        
        # Calculate prediction timestamp
        pred_time = latest_features['timestamp'].iloc[0] + timedelta(hours=h)
        
        predictions[h] = {
            'horizon_hours': h,
            'predicted_pm25': float(pred),
            'prediction_time': pred_time.isoformat(),
            'model_used': model_name,
            'current_pm25': float(latest_features['pm2_5'].iloc[0])
        }
        print(f"\n📊 {h}-hour forecast:")
        print(f"   Prediction time: {pred_time.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"   Predicted PM2.5: {pred:.2f} µg/m³")
        print(f"   Current PM2.5:   {latest_features['pm2_5'].iloc[0]:.2f} µg/m³")
        print(f"   Change:          {pred - latest_features['pm2_5'].iloc[0]:+.2f} µg/m³")
        print(f"   Model:           {model_name.upper()}")
    
    return predictions

def pm25_to_aqi(pm25):
    
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + i_lo
            return round(aqi)

    return None  # PM2.5 out of range



def classify_aqi(aqi):
    # classifies pm2.5 into aqi values
    if aqi <= 12:
        return "Good", "🟢", "Air quality is satisfactory"
    elif aqi <= 35.4:
        return "Moderate", "🟡", "Air quality is acceptable"
    elif aqi <= 55.4:
        return "Unhealthy for Sensitive Groups", "🟠", "Sensitive groups may experience health effects"
    elif aqi <= 150.4:
        return "Unhealthy", "🔴", "Everyone may begin to experience health effects"
    elif aqi <= 250.4:
        return "Very Unhealthy", "🟣", "Health alert: everyone may experience serious effects"
    else:
        return "Hazardous", "🟤", "Health warning of emergency conditions"


def print_summary(predictions, latest_features):

    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    
    current_pm25 = latest_features['pm2_5'].iloc[0]
    current_time = latest_features['timestamp'].iloc[0]
    
    print(f"\n📍 Location: Bahawalpur, Pakistan")
    print(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"🌫️  Current PM2.5: {current_pm25:.2f} µg/m³")
    
    aqi_from_formula=pm25_to_aqi(current_pm25)
    category, emoji, desc = classify_aqi(aqi_from_formula)
    print(f"   AQI Category: {emoji} {category}")
    print(f"   {desc}")
    
    print("\n" + "-"*70)
    print("FORECASTS")
    print("-"*70)
    
    for h in sorted(predictions.keys()):
        pred = predictions[h]
        pred_pm25 = pred['predicted_pm25']
        pred_time = datetime.fromisoformat(pred['prediction_time'])
        

        aqi_from_formula=pm25_to_aqi(pred_pm25)
        category, emoji, desc = classify_aqi(aqi_from_formula)
        change = pred_pm25 - current_pm25
        
        print(f"\n⏰ {h}-Hour Forecast ({pred_time.strftime('%Y-%m-%d %H:%M')})")
        print(f"   Predicted PM2.5: {pred_pm25:.2f} µg/m³ ({change:+.2f})")
        print(f"   AQI Category: {emoji} {category}")
        print(f"   Model: {pred['model_used'].upper()}")


def save_predictions(predictions):
    """Save predictions to JSON file."""
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d_%H%M")
    output_file = output_dir / f"predictions_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'predictions': predictions
        }, f, indent=2)
    
    print(f"\n💾 Predictions saved to: {output_file}")
    
    # Also save as latest
    latest_file = output_dir / "latest_predictions.json"
    with open(latest_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'predictions': predictions
        }, f, indent=2)
    
    print(f"💾 Also saved as: {latest_file}")

def run_prediction_pipeline():
    project = get_hopsworks_project()
    latest_features = load_latest_features()
    models, feature_names = load_trained_models(project)
    predictions = make_predictions(latest_features, models, feature_names)
    return latest_features, predictions, models


def main():
    # Prediction pipeline
    try:
        project=get_hopsworks_project()
        # Load latest features from Hopsworks
        latest_features = load_latest_features()
        
        # Load trained models
        models, feature_names = load_trained_models(project)
        
        if not models:
            print("\n❌ No models available for prediction!")
            return 1
        
        # Make predictions
        predictions = make_predictions(latest_features, models, feature_names)
        
        # Print summary
        print_summary(predictions, latest_features)
        
        # Save predictions
        save_predictions(predictions)
        
        print("\n" + "="*70)
        print("✅ Predictions completed successfully!")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())