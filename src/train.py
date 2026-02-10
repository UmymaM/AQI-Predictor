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
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor  
from catboost import CatBoostRegressor  
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

load_dotenv()

# Configuration
FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 1
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TEST_RATIO = 0.2
HORIZONS = [24, 48, 72]

BASE_FEATURES: List[str] = [
    "pm25_lag1","pm25_lag6","pm25_lag24","pm25_ma6","pm25_ma24","pm25_change_1hr",    
    "temperature_2m","relative_humidity_2m", "wind_speed_10m","pressure_msl",
     "pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide",
    "hour","day_of_week", "day","month",]

def get_hopsworks_project():
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project_name or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT or HOPSWORKS_API_KEY not set in .env")
    print(f"Connecting to Hopsworks project: {project_name}")
    return hopsworks.login(project=project_name, api_key_value=api_key)


def load_features_from_hopsworks() -> pd.DataFrame:
    print(" Loading features from Hopsworks ")
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION)
        df = fg.read()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error loading features from Hopsworks: {e}")


def create_unified_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Adding horizons as features")
    rows = []
    for h in HORIZONS:
        # Create target for this horizon
        df_horizon = df.copy()
        df_horizon['target'] = df_horizon['pm2_5'].shift(-h)
        df_horizon['horizon'] = h  # Adding horizon as feature
        
        df_horizon = df_horizon.dropna(subset=['target'])
        feature_cols = BASE_FEATURES + ['horizon', 'target']
        df_horizon = df_horizon[feature_cols]
        rows.append(df_horizon)
        print(f"  {h}h horizon: {len(df_horizon)} samples")
    
    # Combine all horizons into one dataset
    unified_df = pd.concat(rows, ignore_index=True)
    return unified_df

def horizon_aware_split(df: pd.DataFrame):
    train_parts, test_parts = [], []
    for h in HORIZONS:
        df_h = df[df["horizon"] == h]
        split_idx = int(len(df_h) * (1 - TEST_RATIO))

        train_parts.append(df_h.iloc[:split_idx])
        test_parts.append(df_h.iloc[split_idx:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df



def get_model_candidates() -> Dict:
    return {
        "ridge": Ridge( alpha=1.0,random_state=RANDOM_STATE),

        "lasso": Lasso(alpha=0.5,random_state=RANDOM_STATE,max_iter=2000),

        "rf": RandomForestRegressor(n_estimators=300,max_depth=15,min_samples_split=5,
            min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1),

        "gbr": GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,
            max_depth=5,random_state=RANDOM_STATE),

        "xgboost": XGBRegressor(n_estimators=300,learning_rate=0.1,max_depth=5,
            subsample=0.8,colsample_bytree=0.8,random_state=RANDOM_STATE,
            n_jobs=-1,verbosity=0),
        
        "lightgbm": LGBMRegressor(n_estimators=300,learning_rate=0.1,max_depth=5,
            num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=RANDOM_STATE,
            n_jobs=-1,verbose=-1),
        
        "catboost": CatBoostRegressor(iterations=300,learning_rate=0.1,depth=5,
            subsample=0.8,random_state=RANDOM_STATE,verbose=0,thread_count=-1),
        
        "extratrees": ExtraTreesRegressor(n_estimators=300,max_depth=15,min_samples_split=5,
            min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1)
    }


def train_unified_model(df: pd.DataFrame) -> Tuple[str, object, Dict, List[str]]:
    print(f"Training models for all horizons :D")
    
    feature_cols = BASE_FEATURES + ['horizon']
    X = df[feature_cols].values
    y = df['target'].values
    
    # Time-series split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False  
    )
    print(f"  Train size: {len(X_train):,} samples \n Test size: {len(X_test):,} samples")
    candidates = get_model_candidates()
    best_name = None
    best_model = None
    best_rmse = float('inf')
    all_metrics = {}
    # Train and evaluate each model
    for name, model in candidates.items():
        print(f"\n{'='*70}")
        print(f"Training: {name.upper()}")
        print(f"{'='*70}")
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Overall metrics
        rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        print(f"  Train RMSE: {rmse_train:.4f}")
        print(f"  Test RMSE:  {rmse_test:.4f}")
        print(f"  Test MAE:   {mae_test:.4f}")
        print(f"  Test R²:    {r2_test:.4f}")
        # per horizon metrics
        horizon_metrics = {}
        print(f"\n  Per-Horizon Performance:")
        for h in HORIZONS:
            horizon_mask = X_test[:, -1] == h  # Last feature is 'horizon'
            if horizon_mask.sum() > 0:
                y_test_h = y_test[horizon_mask]
                y_pred_h = y_pred_test[horizon_mask]
                
                rmse_h = sqrt(mean_squared_error(y_test_h, y_pred_h))
                mae_h = mean_absolute_error(y_test_h, y_pred_h)
                r2_h = r2_score(y_test_h, y_pred_h)
                
                horizon_metrics[f"{h}h"] = {
                    "rmse": float(rmse_h),
                    "mae": float(mae_h),
                    "r2": float(r2_h),
                    "samples": int(horizon_mask.sum())}
                print(f"    {h}h: RMSE={rmse_h:.4f}, MAE={mae_h:.4f}, R²={r2_h:.4f}")
        
        # Store metrics
        all_metrics[name] = {
            "overall": {
                "rmse_train": float(rmse_train), "rmse": float(rmse_test),
                "mae": float(mae_test),"r2": float(r2_test)},
            "per_horizon": horizon_metrics}
        
        # Track best model
        if rmse_test < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse_test
            print(f"New best model!")

    print(f"  Best model: {best_name.upper()}")
    print(f"  Test RMSE: {best_rmse:.4f}")
    print(f"  Test MAE:  {all_metrics[best_name]['overall']['mae']:.4f}")
    print(f"  Test R²:   {all_metrics[best_name]['overall']['r2']:.4f}")

    return best_name, best_model, all_metrics, feature_cols



def perform_shap_analysis(model, X_train: np.ndarray,  feature_names: List[str],
    model_name: str, outdir: str = "shap_plots"):
    try:
        os.makedirs(outdir, exist_ok=True)
        print(f"SHAP ANALYSIS")
        # setting sample size to 200 as shap is slower on larger datasets
        sample_size = min(200, len(X_train))
        indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_shap = X_train[indices]

        print("SHAP Explainer")
        if model_name in ['rf', 'gbr', 'xgboost', 'lightgbm', 'extratrees']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_shap)

        shap_values = explainer(X_shap)
    
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values,  X_shap, feature_names=feature_names,show=False)
        summary_file = os.path.join(outdir, "shap_summary_unified.png")
        plt.savefig(summary_file, bbox_inches='tight', dpi=150)
        plt.close()
        plt.figure(figsize=(10, 8))
        shap.summary_plot( shap_values,X_shap,feature_names=feature_names,
            plot_type="bar",show=False)
        importance_file = os.path.join(outdir, "shap_importance_unified.png")
        plt.savefig(importance_file, bbox_inches='tight', dpi=150)
        plt.close()
        return True
        
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
        return False
    


def save_unified_model(project, model_obj, model_name: str,metrics: Dict, 
    feature_names: List[str],all_metrics: Dict,registry_name: str = "aqi_predictor_unified"):

    print("Saving model to hospworks.")
    try:
        mr = project.get_model_registry()
        import tempfile
        # creating a temp directory bcs hopsworks uploads a dir containing feature schema,
        # metrics,etc and not a single file
        model_dir = tempfile.mkdtemp()
        # 1. Save model
        model_file = os.path.join(model_dir, "model.pkl")
        joblib.dump(model_obj, model_file)
        print(f"Model saved to temp file: {model_file}")
        #2.  Save feature names
        features_file = os.path.join(model_dir, "features.json")
        with open(features_file, 'w') as f:
            json.dump({"feature_names": feature_names}, f, indent=2)
        print(f"Features saved to temp file: {features_file}")
        #3. Save detailed metrics
        metrics_file = os.path.join(model_dir, "detailed_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to temp file: {metrics_file}")
        # Create model in registry
        py_model = mr.sklearn.create_model(
            name=registry_name,
            metrics=metrics["overall"],
            description=(
                f"PM2.5 predictor for horizons {HORIZONS}. "
                f"Algorithm: {model_name}. "
                f"RMSE: {metrics['overall']['rmse']:.2f}, "
                f"R²: {metrics['overall']['r2']:.3f}" ) )
        
        # saving model in registry
        py_model.save(model_dir)
        # Cleanup
        shutil.rmtree(model_dir)
        print("Model uploaded to hopsworks!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def train_and_evaluate() -> Tuple[Dict, str]:
    # training pipeline
    # create version timestamp
    version = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d_%H%M")
    # creating a version specific folder e.g. 2026-01-01
    outdir = os.path.join(MODELS_DIR, version)
    os.makedirs(outdir, exist_ok=True)
    
    #connecting to hopsworks + loading features
    project = get_hopsworks_project()
    df = load_features_from_hopsworks()

    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # adding horizons as features + training model
    unified_df = create_unified_dataset(df)
    best_name, best_model, all_metrics, feature_names = train_unified_model(unified_df)
    
    # Save model locally
    local_path = os.path.join(outdir, "aqi_model.pkl")
    joblib.dump(best_model, local_path)
    print("Model saved locally")
    
    # Save metrics + feature names locally
    all_metrics_path = os.path.join(outdir, "all_models_metrics.json")
    with open(all_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    features_path = os.path.join(outdir, "features.json")
    with open(features_path, 'w') as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    
    # commenting shap as ive generated plots once
    # SHAP analysis
    # X_train = unified_df[feature_names].values
    # perform_shap_analysis(
    #     best_model, 
    #     X_train, 
    #     feature_names, 
    #     best_name,
    #     outdir=os.path.join(outdir, "shap_plots")
    # )
    
    # saving model to hopsworks
    save_unified_model(project, best_model, best_name,all_metrics[best_name], 
        feature_names, all_metrics)
    
    # creating report
    overall_report = {
        "version": version,
        "timestamp": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "model_type": "unified",
        "best_algorithm": best_name,
        "feature_names": feature_names,
        "total_samples": len(unified_df),
        "horizons": HORIZONS,
        "metrics": all_metrics[best_name],
        "all_models": {k: v["overall"] for k, v in all_metrics.items()}}
    
    report_path = os.path.join(outdir, "report.json")
    with open(report_path, 'w') as f:
        json.dump(overall_report, f, indent=2)

    # Copy to 'latest' directory
    latest_dir = os.path.join(MODELS_DIR, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(outdir, latest_dir)
    return overall_report, "success"

def main():
    try:
        report, status = train_and_evaluate()
        print("Training Complete!")
        print(f"Status: {status}")
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    exit(main())