"""
XGBoost-based Traffic Congestion Predictor.
Uses engineered tabular features (weather, time, sensor, GPS aggregates).
Can be used standalone or as an ensemble component alongside STGCN-LSTM.
"""
 
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, Optional
from collections import defaultdict
 
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
logger = logging.getLogger(__name__)
 
 
# -----------------------------------------------------------------------------
# Column Mapping  (actual CSV columns -> internal names)
# -----------------------------------------------------------------------------
 
COLUMN_MAP = {
    "total_vehicles":    "vehicle_count",
    "avg_speed_camera":  "avg_speed",
    "avg_speed_sensor":  "avg_speed",
    "congestion_index":  "congestion_score",
    "congestion_camera": "congestion_raw_camera",
    "congestion_sensor": "congestion_raw_sensor",
    "sensor_flow_veh_hr":"sensor_flow",
    "avg_occupancy":     "occupancy",
    "weather_severity":  "weather_severity",
    "weather_category":  "weather_condition",
    "incident_count":    "incident_count",
}
 
 
def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    dst_to_srcs = defaultdict(list)
    for src, dst in COLUMN_MAP.items():
        if src in df.columns:
            dst_to_srcs[dst].append(src)
 
    rename_single = {}
    for dst, srcs in dst_to_srcs.items():
        if dst in df.columns:
            # destination already exists — drop only sources that differ from dst
            df = df.drop(columns=[s for s in srcs if s in df.columns and s != dst], errors="ignore")
        elif len(srcs) == 1:
            rename_single[srcs[0]] = dst
        else:
            logger.info(f"Averaging {srcs} -> '{dst}'")
            df[dst] = df[srcs].mean(axis=1)
            df = df.drop(columns=srcs)
 
    if rename_single:
        logger.info(f"Remapping columns: {rename_single}")
        df = df.rename(columns=rename_single)
 
    return df
 
 
# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------
 
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    df = _remap_columns(df)
 
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "hour" not in df.columns:
            df["hour"] = df["timestamp"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["timestamp"].dt.dayofweek
        if "month" not in df.columns:
            df["month"] = df["timestamp"].dt.month
        if "is_peak_hour" not in df.columns:
            df["is_peak_hour"] = df["hour"].apply(
                lambda h: 1 if h in range(8, 11) or h in range(17, 21) else 0
            )
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
 
    if "segment_id" not in df.columns:
        df["segment_id"] = "default"
 
    defaults = {
        "vehicle_count":    df.get("sensor_flow", pd.Series([0])).median(),
        "avg_speed":        30.0,
        "temperature":      30.0,
        "rainfall":         0.0,
        "humidity":         70.0,
        "wind_speed":       0.0,
        "congestion_score": None,
        "incident_count":   0,
        "occupancy":        0.0,
        "sensor_flow":      0.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            if default is not None:
                logger.warning(f"Column '{col}' not found - filling with default {default}")
                df[col] = default
            else:
                raise ValueError(
                    f"Target column '{col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
 
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
 
    sort_cols = ["segment_id", "timestamp"] if "timestamp" in df.columns else ["segment_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
 
    for lag in [1, 2, 3, 6, 12]:
        df[f"congestion_lag_{lag}"] = (
            df.groupby("segment_id")["congestion_score"].shift(lag)
        )
        df[f"speed_lag_{lag}"] = (
            df.groupby("segment_id")["avg_speed"].shift(lag)
        )
 
    for win in [3, 6]:
        df[f"congestion_rollmean_{win}"] = (
            df.groupby("segment_id")["congestion_score"]
            .transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        )
        df[f"speed_rollstd_{win}"] = (
            df.groupby("segment_id")["avg_speed"]
            .transform(lambda x: x.shift(1).rolling(win, min_periods=1).std().fillna(0))
        )
 
    df["rain_x_vehicles"] = df["rainfall"] * df["vehicle_count"]
    df["temp_x_hour"]     = df["temperature"] * df["hour"]
 
    if "weather_severity" not in df.columns:
        df["weather_severity"] = 0
 
    if "weather_condition" in df.columns:
        le = LabelEncoder()
        df["weather_encoded"] = le.fit_transform(df["weather_condition"].astype(str))
    else:
        df["weather_encoded"] = 0
 
    if df["segment_id"].dtype == object:
        df["segment_hash"] = df["segment_id"].apply(lambda x: hash(x) % 10_000)
    else:
        df["segment_hash"] = df["segment_id"]
 
    lag_cols = [c for c in df.columns if "lag_" in c or "rollmean_" in c or "rollstd_" in c]
    df[lag_cols] = df[lag_cols].bfill().ffill().fillna(0)
 
    nan_counts = df.isnull().sum()
    remaining_nans = nan_counts[nan_counts > 0]
    if len(remaining_nans):
        logger.info(f"NaN counts before final drop:\n{remaining_nans}")
    df = df.dropna(subset=[TARGET_COL])
 
    return df
 
 
FEATURE_COLS = [
    "vehicle_count", "avg_speed", "occupancy", "sensor_flow", "incident_count",
    "weather_severity", "weather_encoded",
    "rainfall", "humidity", "wind_speed", "temperature",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_peak_hour", "is_weekend",
    "congestion_lag_1", "congestion_lag_2", "congestion_lag_3",
    "congestion_lag_6", "congestion_lag_12",
    "speed_lag_1", "speed_lag_2", "speed_lag_3",
    "congestion_rollmean_3", "congestion_rollmean_6",
    "speed_rollstd_3", "speed_rollstd_6",
    "rain_x_vehicles", "temp_x_hour", "segment_hash",
]
TARGET_COL = "congestion_score"
 
 
# -----------------------------------------------------------------------------
# XGBoost Trainer
# -----------------------------------------------------------------------------
 
class XGBoostTrafficModel:
    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_cols = FEATURE_COLS
 
    def _get_available_features(self, df: pd.DataFrame):
        available = [c for c in self.feature_cols if c in df.columns]
        missing   = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing features (will skip): {missing}")
        return available
 
    def train(self, df: pd.DataFrame, tune_hyperparams: bool = False, test_size: float = 0.2) -> Dict:
        df = build_features(df)
        feats = self._get_available_features(df)
 
        logger.info(f"Training with {len(feats)} features on {len(df)} samples")
 
        X = df[feats].values
        y = df[TARGET_COL].values
 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
 
        if tune_hyperparams:
            self.model = self._tune(X_train, y_train)
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                early_stopping_rounds=30,
                eval_metric="rmse",
                verbosity=1,
            )
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
 
        metrics = self._evaluate(X_train, y_train, X_val, y_val, feats)
        logger.info(
            f"Validation RMSE: {metrics['val_rmse']:.4f}  "
            f"MAE: {metrics['val_mae']:.4f}  "
            f"R2: {metrics['val_r2']:.4f}"
        )
        return metrics
 
    def _tune(self, X_train, y_train) -> xgb.XGBRegressor:
        param_grid = {
            "n_estimators":     [300, 500, 700],
            "max_depth":        [4, 6, 8],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        base   = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42)
        search = RandomizedSearchCV(base, param_grid, n_iter=20, cv=3,
                                    scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        logger.info(f"Best params: {search.best_params_}")
        return search.best_estimator_
 
    def _evaluate(self, X_tr, y_tr, X_val, y_val, feats) -> Dict:
        y_tr_pred  = self.model.predict(X_tr)
        y_val_pred = self.model.predict(X_val)
        importances = dict(zip(feats, self.model.feature_importances_))
        top_feats   = sorted(importances.items(), key=lambda x: -x[1])[:10]
        return {
            "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
            "val_rmse":   float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            "val_mae":    float(mean_absolute_error(y_val, y_val_pred)),
            "val_r2":     float(r2_score(y_val, y_val_pred)),
            "top_features": top_feats,
        }
 
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = build_features(df)
        feats = self._get_available_features(df)
        return self.model.predict(df[feats].values)
 
    def save(self, name: str = "xgb_traffic"):
        path = os.path.join(self.model_dir, f"{name}.json")
        self.model.save_model(path)
        logger.info(f"XGBoost model saved -> {path}")
 
    def load(self, name: str = "xgb_traffic"):
        path = os.path.join(self.model_dir, f"{name}.json")
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        logger.info(f"XGBoost model loaded <- {path}")
 