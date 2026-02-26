import os
import json
import shutil
from datetime import datetime
from math import sqrt
from typing import List, Dict, Tuple
import warnings
import joblib
import tempfile
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor  
from catboost import CatBoostRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
load_dotenv()

# Configuration
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
HORIZONS = [24, 48, 72]

BASE_FEATURES: List[str] = [
    "pm25_lag1", "pm25_lag6", "pm25_lag24", "pm25_ma6", "pm25_ma24", "pm25_change_1hr",    
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pressure_msl",
    "hour", "day_of_week", "day", "month"
]


def get_hopsworks_project():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),  # Replace with your project name
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY")  # Get from Hopsworks UI > Account Settings > API Keys
    )
    return project


def load_features_from_hopsworks() -> pd.DataFrame:
    print("Loading features from Hopsworks")
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    df = fg.read()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df)} rows")
    return df


def create_unified_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Create unified dataset with horizon as a feature
    print("Creating unified dataset")
    rows = []
    for h in HORIZONS:
        df_horizon = df.copy()
        df_horizon['target'] = df_horizon['pm2_5'].shift(-h)
        df_horizon['horizon'] = h
        df_horizon = df_horizon.dropna(subset=['target'])
        df_horizon = df_horizon[BASE_FEATURES + ['horizon', 'target']]
        rows.append(df_horizon)
    
    unified_df = pd.concat(rows, ignore_index=True)
    print(f"Created {len(unified_df)} samples across {len(HORIZONS)} horizons")
    return unified_df


def get_model_candidates() -> Dict:
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "lasso": Lasso(alpha=0.5, random_state=RANDOM_STATE, max_iter=2000),
        
        "lightgbm": LGBMRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=3, num_leaves=15,
            min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=0.3, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        ),
        
        "catboost": CatBoostRegressor(
            iterations=120, learning_rate=0.1, depth=3, l2_leaf_reg=10,
            subsample=0.7, random_state=RANDOM_STATE, verbose=0, thread_count=-1
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.1, max_depth=3,
            min_samples_split=25, min_samples_leaf=12, subsample=0.7,
            max_features='sqrt', random_state=RANDOM_STATE
        ),
        
        "xgboost": XGBRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=3, min_child_weight=8,
            subsample=0.7, colsample_bytree=0.7, gamma=0.3, reg_alpha=0.3,
            reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ),
    }

def train_unified_model(df: pd.DataFrame) -> Tuple[str, object, Dict, List[str]]:
    print("Training models")
    
    feature_cols = BASE_FEATURES + ['horizon']
    X = df[feature_cols].values
    y = df['target'].values
    
    # Time-series split (chronological)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    candidates = get_model_candidates()
    best_name = None
    best_model = None
    best_rmse = float('inf')
    all_metrics = {}
    
    for name, model in candidates.items():
        print(f"\nTraining {name.upper()}...", end=" ")
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        gap = r2_train - r2_test
        
        print(f"R²={r2_test:.4f}, RMSE={rmse_test:.2f}, Gap={gap:.4f}")
        
        # Calculate per-horizon metrics
        horizon_metrics = {}
        for h in HORIZONS:
            horizon_mask = X_test[:, -1] == h
            if horizon_mask.sum() > 0:
                y_test_h = y_test[horizon_mask]
                y_pred_h = y_pred_test[horizon_mask]
                
                horizon_metrics[f"{h}h"] = {
                    "r2": float(r2_score(y_test_h, y_pred_h)),
                    "rmse": float(sqrt(mean_squared_error(y_test_h, y_pred_h))),
                    "mae": float(mean_absolute_error(y_test_h, y_pred_h)),
                    "samples": int(horizon_mask.sum())
                }
        
        # Store metrics
        all_metrics[name] = {
            "overall": {
                "r2_train": float(r2_train),
                "r2": float(r2_test),
                "rmse": float(rmse_test),
                "mae": float(mae_test)
            },
            "per_horizon": horizon_metrics
        }
        
        # Track best model
        if rmse_test < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse_test

    return best_name, best_model, all_metrics, feature_cols


def save_unified_model(project, model_obj, model_name: str, metrics: Dict, 
                      feature_names: List[str], all_metrics: Dict):
    print("\nSaving model to Hopsworks", end=" ")
    
    model_dir = tempfile.mkdtemp()
    
    try:
        # Save artifacts
        joblib.dump(model_obj, os.path.join(model_dir, "model.pkl"))
        
        with open(os.path.join(model_dir, "features.json"), 'w') as f:
            json.dump({"feature_names": feature_names}, f, indent=2)
        
        with open(os.path.join(model_dir, "detailed_metrics.json"), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Upload to Hopsworks
        mr = project.get_model_registry()
        py_model = mr.sklearn.create_model(
            name="aqi_predictor_unified",
            metrics=metrics["overall"],
            description=(
                f"PM2.5 predictor ({', '.join(map(str, HORIZONS))}h horizons). "
                f"Algorithm: {model_name}. "
                f"RMSE: {metrics['overall']['rmse']:.2f}, R²: {metrics['overall']['r2']:.3f}"
            )
        )
        py_model.save(model_dir)
        shutil.rmtree(model_dir)
        
        return True
        
    except Exception as e:
        print(f"Failed: {e}")
        shutil.rmtree(model_dir, ignore_errors=True)
        return False


def train_and_evaluate() -> Tuple[Dict, str]:
    # Main Training Pipeline
    version = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d_%H%M")
    
    # Load and train
    project = get_hopsworks_project()
    df = load_features_from_hopsworks()
    unified_df = create_unified_dataset(df)
    best_name, best_model, all_metrics, feature_names = train_unified_model(unified_df)
    
    # Save to Hopsworks
    save_unified_model(project, best_model, best_name, all_metrics[best_name], 
                      feature_names, all_metrics)
    
    # Create report
    report = {
        "version": version,
        "timestamp": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "best_algorithm": best_name,
        "metrics": all_metrics[best_name],
        "all_models": {k: v["overall"] for k, v in all_metrics.items()}}
    
    return report, "success"


def main():
    try:
        print("Training Pipeline")
        report, status = train_and_evaluate()
        
        print(f"Training ")
        print(f"Best Model: {report['best_algorithm'].upper()}")
        print(f"Test R²: {report['metrics']['overall']['r2']:.4f}")
        print(f"Test RMSE: {report['metrics']['overall']['rmse']:.2f} µg/m³")
    
        return 0
        
    except Exception as e:
        print("Training Failed.")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())