import os
import json
import shutil
from datetime import datetime
from math import sqrt
from typing import List, Dict, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Horizons we want to predict (in hours)
HORIZONS = [24, 48, 72]

# Feature names (must match what's in your feature store)
FEATURE_NAMES: List[str] = [
    # "pm2_5",              # Current PM2.5 (your column name)
    "pm25_lag1",          # Lag 1 hour
    "pm25_lag6",          # Lag 6 hours
    "pm25_lag24",         # Lag 24 hours
    "pm25_ma6",           # Moving average 6 hours
    "pm25_ma24",          # Moving average 24 hours
    "pm25_change_1hr",    # 1-hour change
    "temperature_2m",     # Temperature
    "relative_humidity_2m",  # Humidity
    "wind_speed_10m",     # Wind speed
    "pressure_msl",       # Pressure 
    "hour",               # Hour of day
    "day_of_week",        # Day of week 
    "month",              # Month
]


def get_hopsworks_project():
    """Connect to Hopsworks project."""
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not project_name or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT or HOPSWORKS_API_KEY not set in .env")
    
    print(f"Connecting to Hopsworks project: {project_name}")
    return hopsworks.login(project=project_name, api_key_value=api_key)


def load_features_from_hopsworks() -> pd.DataFrame:

    print("\n=== Loading features from Hopsworks ===")
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        # Read all data
        df = fg.read()
        print(f"Loaded {len(df)} rows from feature store")
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error loading features from Hopsworks: {e}")


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    
    print("\n=== Creating target variables ===")
    df = df.copy()
    for h in HORIZONS:
        target_col = f"pm25_tplus_{h}"
        # Shift pm2_5 backwards by h hours to create future target
        df[target_col] = df['pm2_5'].shift(-h)  # Using your column name
        print(f"Created target: {target_col}")
    
    # Drop rows where any target is NaN
    initial_len = len(df)
    df = df.dropna(subset=[f"pm25_tplus_{h}" for h in HORIZONS])
    print(f"Dropped {initial_len - len(df)} rows with NaN targets")
    return df


def prepare_data(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:

    # Prepare features (X) and target (y) for training.
    # Check if all required columns exist
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")
    # Drop rows with NaN in features or target
    required_cols = FEATURE_NAMES + [target_col]
    df_clean = df[required_cols].dropna()
    
    X = df_clean[FEATURE_NAMES].values
    y = df_clean[target_col].values
    
    return X, y


def get_model_candidates() -> Dict:

    return {
        "ridge": Ridge(
            alpha=1.0,
            random_state=RANDOM_STATE
        ),
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
    }

def train_for_horizon(df: pd.DataFrame, target_col: str) -> Tuple[str, Dict, object]:
    
    # Train all candidate models for a specific horizon and return the best.

    print(f"\n{'='*60}")
    print(f"Training models for: {target_col}")
    print(f"{'='*60}")
    
    # Prepare data
    X, y = prepare_data(df, target_col)
    
    if len(X) < 200:
        print(f"WARNING: Only {len(X)} samples for {target_col}. Results may be noisy.")
    
    print(f"Training samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    
    # Time-series split (no shuffle to preserve temporal order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False  # Important for time series!
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Get model candidates
    candidates = get_model_candidates()
    
    best_name = None
    best_model = None
    best_rmse = float('inf')
    metrics: Dict[str, Dict[str, float]] = {}
    
    # Train and evaluate each model
    for name, model in candidates.items():
        print(f"\nTraining {name.upper()}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics[name] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Track best model
        if rmse < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse
    
    print(f"\n🏆 Best model: {best_name.upper()} (RMSE: {best_rmse:.4f})")
    
    return best_name, metrics, best_model


def save_to_model_registry(project, model_obj, model_path: str, model_name: str, horizon: int, metrics: Dict):

    # Save trained model to Hopsworks Model Registry.
    
    print(f"\n{'='*60}")
    print(f"Uploading model to Hopsworks Model Registry...")
    print(f"{'='*60}")
    
    try:
        # Get model registry (following Hopsworks docs pattern)
        print("1. Getting model registry...")
        mr = project.get_model_registry()
        print("   ✓ Model registry accessed")
        
        # Model name in registry
        registry_model_name = f"aqi_predictor_t{horizon}h"
        print(f"2. Model name: {registry_model_name}")
        
        # Create model directory structure
        print("3. Creating model directory...")
        import tempfile
        import shutil
        
        model_dir = tempfile.mkdtemp()
        
        # Save the actual model object to the directory
        model_file_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model_obj, model_file_path)
        print(f"   ✓ Model saved to: {model_file_path}")
        
        # Create input example (first row of features for schema)
        print("4. Creating input example...")
        
        # Create model in registry using sklearn (since all models are sklearn-compatible)
        print("5. Creating model in Hopsworks...")
        
        py_model = mr.sklearn.create_model(
            name=registry_model_name,
            metrics=metrics,
            description=f"PM2.5 {horizon}h forecast. Algorithm: {model_name}. RMSE: {metrics['rmse']:.2f}"
        )
        print("   ✓ Model created")
        
        # Save model directory to registry
        print("6. Uploading to registry...")
        py_model.save(model_dir)
        print("   ✓ Upload complete!")
        
        # Cleanup
        shutil.rmtree(model_dir)
        print("7. Cleaned up temp files")
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: '{registry_model_name}' in Model Registry!")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        print(f"   R²:   {metrics['r2']:.4f}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR uploading to Model Registry")
        print(f"Error: {str(e)}")
        print(f"{'='*60}\n")
        
        import traceback
        traceback.print_exc()
        
        return False


def train_and_evaluate() -> Tuple[Dict, str]:
    """
    Main training pipeline:
    1. Load features from Hopsworks
    2. Create target variables
    3. Train models for each horizon
    4. Save best models locally and to Hopsworks Model Registry
    
    Returns:
        Tuple of (overall_report, status)
    """
    print("\n" + "="*70)
    print("AQI PREDICTION MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Create version timestamp (Pakistan time)
    version = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d_%H%M")
    outdir = os.path.join(MODELS_DIR, version)
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\nVersion: {version}")
    print(f"Output directory: {outdir}")
    
    # Get Hopsworks project for model registry
    project = get_hopsworks_project()
    
    # Load features from Hopsworks
    df = load_features_from_hopsworks()
    
    # Create target variables
    df = create_target_variables(df)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize report
    overall_report = {
        "version": version,
        "timestamp": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "feature_names": FEATURE_NAMES,
        "total_samples": len(df),
        "date_range": {
            "start": str(df['timestamp'].min()),
            "end": str(df['timestamp'].max())
        },
        "horizons": {}
    }
    model_version={}
    # Train models for each horizon
    for h in HORIZONS:
        target_col = f"pm25_tplus_{h}"
        
        # Train and get best model
        best_name, metrics, best_model = train_for_horizon(df, target_col)
        
        # Add to report
        overall_report["horizons"][f"h{h}"] = {
            "target": target_col,
            "best_model": best_name,
            "metrics": metrics
        }
        
        # Save best model locally
        model_filename = f"{best_name}_tplus{h}.joblib"
        model_path = os.path.join(outdir, model_filename)
        joblib.dump(best_model, model_path)
        print(f"\n✅ Saved locally: {model_path}")
        
        # Save to Hopsworks Model Registry
        best_metrics = metrics[best_name]
        success = save_to_model_registry(
            project=project,
            model_obj=best_model,
            model_path=model_path,
            model_name=best_name,
            horizon=h,
            metrics=best_metrics
        )
        
        if not success:
            print(f"⚠️  Model saved locally but not uploaded to Hopsworks registry")
    
    # Save metadata files
    print("\n" + "="*60)
    print("Saving metadata files...")
    
    # Save feature names
    features_path = os.path.join(outdir, "features.json")
    with open(features_path, "w") as f:
        json.dump({"feature_names": FEATURE_NAMES}, f, indent=2)
    print(f"✅ Saved: {features_path}")
    
    # Save full report
    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w") as f:
        json.dump(overall_report, f, indent=2)
    print(f"✅ Saved: {report_path}")
    
    # Copy to 'latest' directory
    latest_dir = os.path.join(MODELS_DIR, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(outdir, latest_dir)
    print(f"✅ Copied to: {latest_dir}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for h in HORIZONS:
        h_key = f"h{h}"
        h_data = overall_report["horizons"][h_key]
        best = h_data["best_model"]
        best_metrics = h_data["metrics"][best]
        
        print(f"\n📊 {h}-hour forecast:")
        print(f"   Best Model: {best.upper()}")
        print(f"   RMSE: {best_metrics['rmse']:.4f}")
        print(f"   MAE:  {best_metrics['mae']:.4f}")
        print(f"   R²:   {best_metrics['r2']:.4f}")
    
    print("\n" + "="*70)
    print("✅ Training pipeline completed successfully!")
    print("="*70)
    
    return overall_report, "success"


def main():
    """Main entry point."""
    try:
        report, status = train_and_evaluate()
        
        # Print final report
        print("\n📋 Full Report:")
        print(json.dumps(report, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


