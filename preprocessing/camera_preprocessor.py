"""
camera_preprocessor.py
----------------------
Loads, cleans, and feature-engineers raw camera CSV data.
Handles ByteTrack columns (active_tracks, flow_rate_veh_hr) gracefully
whether or not they exist in the source file.
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
 
 
def load_camera_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df
 
 
def clean_camera_data(df: pd.DataFrame) -> pd.DataFrame:
    # Clip vehicle counts to non-negative
    vehicle_cols = ['cars', 'motorcycles', 'buses', 'trucks']
    for col in vehicle_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
 
    # Recalculate total
    present_vehicle_cols = [c for c in vehicle_cols if c in df.columns]
    if present_vehicle_cols:
        df['total_vehicles'] = df[present_vehicle_cols].sum(axis=1)
 
    # Fill missing avg_speed_kmh using interpolation
    if 'avg_speed_kmh' in df.columns:
        if df['avg_speed_kmh'].isna().any():
            df['avg_speed_kmh'] = df['avg_speed_kmh'].interpolate(method='linear')
        # Clip speed to realistic range (5–80 km/h for Kolkata)
        df['avg_speed_kmh'] = df['avg_speed_kmh'].clip(lower=5, upper=80)
 
    # ── ByteTrack: active_tracks ──────────────────────────────────────────────
    if 'active_tracks' not in df.columns:
        # Fall back to total_vehicles if ByteTrack data is absent
        df['active_tracks'] = df.get('total_vehicles', 0)
    else:
        df['active_tracks'] = (
            df['active_tracks']
            .clip(lower=0)
            .fillna(df.get('total_vehicles', 0))
        )
 
    # ── ByteTrack: flow_rate_veh_hr ───────────────────────────────────────────
    if 'flow_rate_veh_hr' not in df.columns:
        # Derive from total_vehicles (per-minute count → hourly estimate)
        df['flow_rate_veh_hr'] = df.get('total_vehicles', 0) * 60.0
    else:
        df['flow_rate_veh_hr'] = (
            df['flow_rate_veh_hr']
            .clip(lower=0)
            .interpolate(method='linear')
            .fillna(0.0)
        )
 
    # Drop duplicate timestamps per camera
    df = df.drop_duplicates(subset=['camera_id', 'timestamp'])
 
    return df.reset_index(drop=True)
 
 
def engineer_camera_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Time-based features
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek      # 0=Monday
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
 
    # Kolkata peak hours: morning 8-10, evening 17-19
    df['is_morning_peak'] = df['hour'].between(8, 10).astype(int)
    df['is_evening_peak'] = df['hour'].between(17, 19).astype(int)
    df['is_peak_hour']    = (
        (df['is_morning_peak'] == 1) | (df['is_evening_peak'] == 1)
    ).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
 
    # Vehicle mix ratios
    if {'buses', 'trucks', 'total_vehicles'}.issubset(df.columns):
        df['heavy_vehicle_ratio'] = (
            (df['buses'] + df['trucks']) / df['total_vehicles'].replace(0, np.nan)
        ).fillna(0)
 
    # Congestion index: higher total + lower speed = more congested (0–1 scale)
    if 'congestion_index' not in df.columns:
        max_speed    = df['avg_speed_kmh'].max()
        max_vehicles = df['total_vehicles'].max()
        df['congestion_index'] = (
            0.6 * (1 - df['avg_speed_kmh'] / max_speed) +
            0.4 * (df['total_vehicles'] / max_vehicles)
        ).round(4)
 
    # Congestion label
    df['congestion_level'] = pd.cut(
        df['congestion_index'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
 
    return df
 
 
def preprocess_camera(raw_path: str, output_path: str) -> pd.DataFrame:
    print(f"[Camera] Loading from {raw_path}")
    df = load_camera_data(raw_path)
    print(f"[Camera] Loaded {len(df)} rows, {df['camera_id'].nunique()} cameras")
 
    df = clean_camera_data(df)
    print(f"[Camera] After cleaning: {len(df)} rows")
 
    df = engineer_camera_features(df)
    print(f"[Camera] Features engineered: {list(df.columns)}")
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Camera] Saved to {output_path}")
 
    return df
 
 
if __name__ == "__main__":
    preprocess_camera(
        raw_path="data/raw/camera/vehicle_counts.csv",
        output_path="data/preprocessed/camera/vehicle_counts_processed.csv"
    )