# src/train_models.py

"""
Purpose:
  - Train models to forecast PM2.5 at 3 future horizons:
      +24 hours, +48 hours, +72 hours
  - For each horizon, try 4 models (Ridge, RandomForest, GradientBoosting, XGBoost) and pick the best
  - Save the best model per horizon with versioning and a report

Outputs:
  models/
    - <version>/
        - <best_model>_tplus24.joblib
        - <best_model>_tplus48.joblib
        - <best_model>_tplus72.joblib
        - features.json
        - report.json
    - latest/   (copy of the latest run)

Run:
  python train.py
"""

import os
import json
import glob
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
    "pm2_5",              # Current PM2.5 (your column name)
    "pm25_lag1",          # Lag 1 hour
    "pm25_lag6",          # Lag 6 hours
    "pm25_lag24",         # Lag 24 hours
    "pm25_ma6",           # Moving average 6 hours
    "pm25_ma24",          # Moving average 24 hours
    "pm25_change_1hr",    # 1-hour change
    "temperature_2m",     # Temperature
    "relative_humidity_2m",  # Humidity
    "wind_speed_10m",     # Wind speed
    "pressure_msl",       # Pressure (your column name)
    "hour",               # Hour of day
    "day_of_week",        # Day of week (your column name)
    "month",              # Month
]


def get_hopsworks_project():
    """Connect to Hopsworks project."""
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")

    # project = hopsworks.login()
    # fs = project.get_feature_store(name='aqi_prediction_bwp_featurestore')
    # fg = fs.get_feature_group('aqi_features_hourly', version=1)
    
    if not project_name or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT or HOPSWORKS_API_KEY not set in .env")
    
    print(f"Connecting to Hopsworks project: {project_name}")
    return hopsworks.login(project=project_name, api_key_value=api_key)


def load_features_from_hopsworks() -> pd.DataFrame:
    """
    Load features from Hopsworks Feature Store.
    
    Returns:
        DataFrame with all features from the feature group
    """
    print("\n=== Loading features from Hopsworks ===")
    
    try:
        project = hopsworks.login()
        fs = project.get_feature_store(name='aqi_prediction_bwp_featurestore')
        
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
    """
    Create target variables for each forecast horizon.
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with added target columns
    """
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
    """
    Prepare features (X) and target (y) for training.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
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
    """
    Define model candidates for training.
    
    Returns:
        Dictionary of model name -> model instance
    """
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
    """
    Train all candidate models for a specific horizon and return the best.
    
    Args:
        df: DataFrame with features and targets
        target_col: Target column name (e.g., 'pm25_tplus_24')
        
    Returns:
        Tuple of (best_model_name, all_metrics, best_model_object)
    """
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


def train_and_evaluate() -> Tuple[Dict, str]:
    """
    Main training pipeline:
    1. Load features from Hopsworks
    2. Create target variables
    3. Train models for each horizon
    4. Save best models and reports
    
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
        
        # Save best model
        model_filename = f"{best_name}_tplus{h}.joblib"
        model_path = os.path.join(outdir, model_filename)
        joblib.dump(best_model, model_path)
        print(f"\n✅ Saved: {model_path}")
    
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





# ***************************************************************************************************************

# # src/train_models.py

# import os
# import json
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Tuple

# import numpy as np
# import pandas as pd
# import joblib
# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import xgboost as xgb
# import lightgbm as lgb

# import hopsworks
# from dotenv import load_dotenv
# load_dotenv()

# TARGET_COL = "pm2_5"
# HORIZON = 72  # 72-hour forecast
# MODELS_DIR = Path("models")
# MODELS_DIR.mkdir(exist_ok=True)


# class ModelTrainer:
#     def __init__(self, model_name="aqi_model"):
#         self.model_name = model_name
#         self.best_model = None
#         self.best_model_name = None
#         self.model_scores = {}
#         # self.client = hopsworks.login()
#         self.project = hopsworks.login()
#         self.mr = self.project.get_model_registry()
#         # self.project = self.client.get_project("AQI_Predictor")
#         # self.mr = self.project.get_model_registry()
#         self.models = {}
#         self._initialize_models()

#     def _initialize_models(self):
#         # Wrap tree-based models with MultiOutputRegressor for multi-step forecasting
#         self.models["random_forest"] = MultiOutputRegressor(RandomForestRegressor(
#             n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
#         ))
#         self.models["xgboost"] = MultiOutputRegressor(xgb.XGBRegressor(
#             n_estimators=300, max_depth=6, learning_rate=0.05,
#             subsample=0.8, colsample_bytree=0.8,
#             objective="reg:squarederror", random_state=42
#         ))
#         self.models["lightgbm"] = MultiOutputRegressor(lgb.LGBMRegressor(
#             n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42
#         ))
#         print("Models initialized:", list(self.models.keys()))

#     def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
#         """
#         Convert dataset into X (features) and y (multi-step pm2_5)
#         """
#         df = df.sort_values("timestamp").reset_index(drop=True)
#         feature_cols = [c for c in df.columns if c not in ["timestamp", TARGET_COL]]
#         X, y = [], []

#         for i in range(len(df) - HORIZON):
#             X.append(df[feature_cols].iloc[i].values)
#             y.append(df[TARGET_COL].iloc[i + 1:i + 1 + HORIZON].values)

#         X = np.array(X)
#         y = np.array(y)
#         return X, y, feature_cols

#     def train_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
#         X, y, feature_cols = self.prepare_data(df)

#         # Sequential train/validation split
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=0.2, shuffle=False
#         )

#         for name, model in self.models.items():
#             print(f"\nTraining {name}...")
#             model.fit(X_train, y_train)
#             preds = model.predict(X_val)
#             rmse = np.sqrt(mean_squared_error(y_val, preds))
#             mae = mean_absolute_error(y_val, preds)
#             r2 = r2_score(y_val, preds)
#             self.model_scores[name] = {"rmse": rmse, "mae": mae, "r2": r2}
#             print(f"{name} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

#         self.select_best_model()
#         self.save_best_model(feature_cols)
#         self.save_training_results()
#         return self.model_scores

#     def select_best_model(self):
#         best_rmse = float("inf")
#         best_name = None
#         for name, scores in self.model_scores.items():
#             if scores["rmse"] < best_rmse:
#                 best_rmse = scores["rmse"]
#                 best_name = name
#         self.best_model_name = best_name
#         self.best_model = self.models[best_name]
#         print(f"\nBest model: {best_name} (RMSE: {best_rmse:.2f})")

#     def save_training_results(self):
#         results = {
#             "timestamp": datetime.now().isoformat(),
#             "model_scores": self.model_scores,
#             "best_model": self.best_model_name
#         }
#         with open(MODELS_DIR / "training_results.json", "w") as f:
#             json.dump(results, f, indent=2)
#         print("Training results saved.")

#     def save_best_model(self, feature_cols):
#         if self.best_model is None:
#             raise ValueError("No model trained yet.")

#         model_path = MODELS_DIR / f"{self.model_name}.pkl"
#         joblib.dump(self.best_model, model_path)
#         print(f"Saved best model locally: {model_path}")

#         metadata = {"features": feature_cols, "metrics": self.model_scores[self.best_model_name]}
#         metadata_path = MODELS_DIR / f"{self.model_name}_metadata.json"
#         with open(metadata_path, "w") as f:
#             json.dump(metadata, f)
#         print(f"Saved metadata locally: {metadata_path}")

#         # Hopsworks registration
#         try:
#             existing_model = None
#             for m in self.mr.get_models():
#                 if m.name == self.model_name:
#                     existing_model = m
#                     break

#             if existing_model:
#                 version = existing_model.create_model_version(
#                     model=str(model_path),
#                     metrics=self.model_scores[self.best_model_name],
#                     description="Incremental training"
#                 )
#                 print(f"Registered new version: {version.version}")
#             else:
#                 new_model = self.mr.create_model(
#                     name=self.model_name,
#                     description="AQI 72-hour forecast",
#                     metrics=self.model_scores[self.best_model_name]
#                 )
#                 version = new_model.create_model_version(
#                     model=str(model_path),
#                     metrics=self.model_scores[self.best_model_name],
#                     description="Initial version"
#                 )
#                 print(f"Registered model in Hopsworks with version: {version.version}")
#         except Exception as e:
#             print("Failed to register model:", e)


# # ------------------ Example usage ------------------ #
# if __name__ == "__main__":
#     from data_ingestion import fetch_recent
#     from features import build_features

#     raw_df = fetch_recent()
#     feature_df = build_features(raw_df)

#     trainer = ModelTrainer()
#     scores = trainer.train_models(feature_df)
#     print("\nModel training completed. Scores:", scores)



# **********************************************************************************************

# """
# model_trainer.py

# Trains multiple ML models (RandomForest, XGBoost, LSTM) for AQI prediction.
# Evaluates, selects best model (based on RMSE), and registers it in Hopsworks Model Registry.
# Supports SHAP/LIME explainability.
# """

# import os
# import json
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Optional, Tuple, Any

# import numpy as np
# import pandas as pd
# import joblib
# import warnings
# warnings.filterwarnings("ignore")

# # ML
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import xgboost as xgb
# import lightgbm as lgb
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# # Explainability
# import shap
# import lime
# import lime.lime_tabular

# # Hopsworks
# import hopsworks
# from dotenv import load_dotenv

# load_dotenv()

# TARGET_COL = "pm2_5"
# MODELS_DIR = Path("models")
# MODELS_DIR.mkdir(exist_ok=True)

# class ModelTrainer:
#     def __init__(self, model_name="aqi_model"):
#         self.model_name = model_name
#         self.best_model = None
#         self.best_model_name = None
#         self.best_metrics = {}
#         self.client = hopsworks.login()
#         self.project = self.client.get_project("AQI_Predictor")  # replace with your project name
#         self.mr = self.project.get_model_registry()
#         self.models = {}  # dict to hold initialized models

#         # Initialize models
#         self._initialize_models()

#     # ------------------ 1. Initialize models ------------------ #
#     def _initialize_models(self):
#         self.models["random_forest"] = RandomForestRegressor(
#             n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
#         )
#         self.models["xgboost"] = xgb.XGBRegressor(
#             n_estimators=300, max_depth=6, learning_rate=0.05,
#             subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
#             random_state=42
#         )
#         self.models["lightgbm"] = lgb.LGBMRegressor(
#             n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42
#         )
#         print("Models initialized:", list(self.models.keys()))

#     # ------------------ 2. Prepare data ------------------ #
#     def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
#         df = df.sort_values("timestamp").reset_index(drop=True)
#         feature_cols = [c for c in df.columns if c not in ["timestamp", TARGET_COL]]
#         X = df[feature_cols].values
#         y = df[TARGET_COL].values
#         return X, y, feature_cols

#     # ------------------ 3. Create LSTM Model ------------------ #
#     def create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
#         model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=input_shape),
#             Dropout(0.2),
#             LSTM(50, return_sequences=False),
#             Dropout(0.2),
#             Dense(25),
#             Dense(1)
#         ])
#         model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
#         return model

#     # ------------------ 4. Train models ------------------ #
#     def train_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
#         X, y, feature_cols = self.prepare_data(df)

#         # Train-test split (sequential)
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=0.2, shuffle=False
#         )

#         for name, model in self.models.items():
#             print(f"\nTraining {name}...")
#             model.fit(X_train, y_train)
#             preds = model.predict(X_val)
#             rmse = np.sqrt(mean_squared_error(y_val, preds))
#             mae = mean_absolute_error(y_val, preds)
#             r2 = r2_score(y_val, preds)
#             self.model_scores[name] = {"rmse": rmse, "mae": mae, "r2": r2}
#             print(f"{name} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

#         self.select_best_model()
#         self.save_best_model(feature_cols)
#         self.save_training_results()
#         return self.model_scores

#     # ------------------ 5. Select best model ------------------ #
#     def select_best_model(self):
#         best_rmse = float("inf")
#         best_name = None
#         for name, scores in self.model_scores.items():
#             if scores["rmse"] < best_rmse:
#                 best_rmse = scores["rmse"]
#                 best_name = name
#         self.best_model_name = best_name
#         self.best_model = self.models[best_name]
#         print(f"\nBest model: {best_name} (RMSE: {best_rmse:.2f})")

#     # ------------------ 6. Save training results ------------------ #
#     def save_training_results(self):
#         results = {
#             "timestamp": datetime.now().isoformat(),
#             "model_scores": self.model_scores,
#             "best_model": self.best_model_name
#         }
#         with open(MODELS_DIR / "training_results.json", "w") as f:
#             json.dump(results, f, indent=2)
#         print("Training results saved.")

#     # ------------------ 7. Save best model ------------------ #
#     # def save_best_model(self, feature_cols: list):
#     #     if not self.best_model:
#     #         raise ValueError("No best model selected")

#     #     mr_model = self.mr.log_model(
#     #         model=self.best_model,
#     #         name="aqi_pm25_forecaster",
#     #         metrics=self.model_scores[self.best_model_name],
#     #         model_type="sklearn",
#     #         description=f"{self.best_model_name} predicting next hour AQI",
#     #         training_data={"feature_group": self.fg_name, "version": self.fg_version}
#     #     )
#     #     mr_model.set_production()
#     #     print(f"Best model {self.best_model_name} registered in Hopsworks.")

#     def save_best_model(self, feature_cols):
#         """
#         Save the best model and metadata to Hopsworks Model Registry
#         """
#         if self.best_model is None:
#             raise ValueError("No model has been trained yet.")

#         # save locally first
#         model_dir = "models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
#         joblib.dump(self.best_model, model_path)
#         print(f"Saved best model locally at {model_path}")

#         # save metadata
#         metadata = {
#             "features": feature_cols,
#             "metrics": self.best_metrics
#         }
#         metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.json")
#         with open(metadata_path, "w") as f:
#             json.dump(metadata, f)
#         print(f"Saved metadata locally at {metadata_path}")

#         # Register model in Hopsworks Model Registry
#         try:
#             # Check if model exists in registry
#             existing_model = None
#             for m in self.mr.get_models():
#                 if m.name == self.model_name:
#                     existing_model = m
#                     break

#             if existing_model:
#                 # Create a new version
#                 model_version = existing_model.create_model_version(
#                     model=model_path,
#                     metrics=self.best_metrics,
#                     description="New version from incremental training"
#                 )
#                 print(f"Registered new version in Hopsworks: {model_version.version}")
#             else:
#                 # Create new model
#                 new_model = self.mr.create_model(
#                     name=self.model_name,
#                     description="AQI prediction model",
#                     metrics=self.best_metrics
#                 )
#                 model_version = new_model.create_model_version(
#                     model=model_path,
#                     metrics=self.best_metrics,
#                     description="Initial version from training"
#                 )
#                 print(f"Registered model in Hopsworks with version: {model_version.version}")

#         except Exception as e:
#             print("Failed to register model in Hopsworks:", e)


#     # ------------------ 8. Load model ------------------ #
#     def load_model(self, version=None):
#         """
#         Load model from Hopsworks registry
#         """
#         try:
#             model = self.mr.get_model(self.model_name)
#             if version is None:
#                 version = model.get_latest_version().version
#             model_path = model.get_version(version).download()
#             self.best_model = joblib.load(model_path)
#             print(f"Loaded model {self.model_name} version {version} from Hopsworks")
#         except Exception as e:
#             print("Failed to load model:", e)

#     # ------------------ 9. Explain predictions ------------------ #
#     def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
#         if self.best_model_name is None:
#             raise ValueError("No model trained yet")

#         model = self.best_model
#         # SHAP
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X)

#         # LIME
#         lime_explainer = lime.lime_tabular.LimeTabularExplainer(
#             X.values,
#             feature_names=X.columns.tolist(),
#             class_names=[TARGET_COL],
#             mode="regression"
#         )
#         lime_exp = lime_explainer.explain_instance(
#             X.iloc[0].values,
#             model.predict,
#             num_features=len(X.columns)
#         )

#         return {"shap_values": shap_values.tolist(), "lime_explanation": lime_exp.as_list()}


# # ------------------ Example Usage ------------------ #
# if __name__ == "__main__":
#     from data_ingestion import fetch_recent
#     from features import build_features

#     # Fetch recent data from Hopsworks or API
#     raw_df = fetch_recent()
#     feature_df = build_features(raw_df)

#     trainer = ModelTrainer(fg_name="aqi_features_hourly", fg_version=1)
#     scores = trainer.train_models(feature_df)
#     print("\nModel training completed. Scores:", scores)


# *******************************************************************************************************************

# import os
# import hopsworks
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from dotenv import load_dotenv
# import joblib
# import numpy as np

# load_dotenv()

# FEATURE_GROUP_NAME = "aqi_features_hourly"
# FEATURE_GROUP_VERSION = 1
# MODEL_REGISTRY_NAME = "aqi_predictor_model"

# HORIZON_HOURS = 72  # next 3 days, hourly predictions

# def fetch_features():
#     project = hopsworks.login(
#         project=os.getenv("HOPSWORKS_PROJECT"),
#         api_key_value=os.getenv("HOPSWORKS_API_KEY")
#     )
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(
#         name=FEATURE_GROUP_NAME,
#         version=FEATURE_GROUP_VERSION
#     )
#     df = fg.read()
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     return df

# def create_targets(df):
#     """Create multi-output targets for the next 72 hours (3 days)."""
#     df = df.copy()
#     for h in range(1, HORIZON_HOURS + 1):
#         df[f"t_plus_{h}"] = df["pm2_5"].shift(-h)
#     df = df.dropna().reset_index(drop=True)
#     return df

# def train_multioutput_model(X, y):
#     base_model = RandomForestRegressor(
#         n_estimators=100,
#         max_depth=10,
#         random_state=42
#     )
#     model = MultiOutputRegressor(base_model)
#     model.fit(X, y)
#     return model

# def evaluate_model(model, X_test, y_test):
#     """Evaluate multi-output model and return mean metrics over all horizons."""
#     preds = model.predict(X_test)
#     mae = mean_absolute_error(y_test, preds)
#     rmse = mean_squared_error(y_test, preds, squared=False)
#     return {"mae": mae, "rmse": rmse}

# def main():
#     df = fetch_features()
#     df = create_targets(df)
    
#     feature_cols = df.columns.difference(["timestamp"] + [f"t_plus_{h}" for h in range(1, HORIZON_HOURS+1)])
#     X = df[feature_cols]
    
#     # Targets: all 72 future hours
#     target_cols = [f"t_plus_{h}" for h in range(1, HORIZON_HOURS+1)]
#     y = df[target_cols]
    
#     # Train 3 separate models
#     models = {}
#     metrics_all = {}
    
#     for i in range(1, 4):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#         model = train_multioutput_model(X_train, y_train)
#         metrics = evaluate_model(model, X_test, y_test)
#         print(f"Model {i} metrics: {metrics}")
        
#         # Save locally
#         os.makedirs("models", exist_ok=True)
#         path = f"models/model_{i}_multioutput_rf.pkl"
#         joblib.dump(model, path)
#         print(f"Saved model {i} at {path}")
        
#         models[i] = model
#         metrics_all[i] = metrics
    
#     # Select the best model (lowest MAE)
#     best_model_id = min(metrics_all, key=lambda k: metrics_all[k]["mae"])
#     print(f"Best model is Model {best_model_id} with metrics {metrics_all[best_model_id]}")
    
#     # Push the best model to Hopsworks Model Registry
#     project = hopsworks.login(
#         project=os.getenv("HOPSWORKS_PROJECT"),
#         api_key_value=os.getenv("HOPSWORKS_API_KEY")
#     )
#     mr = project.get_model_registry()
    
#     best_model = models[best_model_id]
#     model_file_path = f"models/model_{best_model_id}_multioutput_rf.pkl"
#     mr_model = mr.python.create_model(
#         name=MODEL_REGISTRY_NAME,
#         metrics=metrics_all[best_model_id],
#         description=f"RandomForest multi-output model predicting next 72 hours AQI",
#         model_file=model_file_path
#     )
#     mr_model.save()
#     print(f"Best model pushed to Hopsworks Model Registry: {MODEL_REGISTRY_NAME}")

# if __name__ == "__main__":
#     main()
