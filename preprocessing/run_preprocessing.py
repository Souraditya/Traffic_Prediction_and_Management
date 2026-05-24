"""
Master Preprocessing Orchestrator
Runs all data source preprocessors and produces a merged dataset
ready for model training.
 
Usage:
    python preprocessing/run_preprocessing.py
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
 
from preprocessing.camera_preprocessor  import preprocess_camera
from preprocessing.sensor_preprocessor  import preprocess_sensor
from preprocessing.weather_preprocessor import preprocess_weather
from preprocessing.gps_preprocessor     import preprocess_gps
 
 
RAW_BASE  = Path("data/raw")
PROC_BASE = Path("data/processed")
 
 
def merge_datasets(
    camera_df:  pd.DataFrame,
    sensor_df:  pd.DataFrame,
    weather_df: pd.DataFrame,
    gps_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all processed datasets on timestamp (rounded to nearest minute).
    Camera and sensor data are aggregated per minute across all devices.
    Weather and GPS data are joined by nearest timestamp.
    """
    print("\n[Merge] Starting dataset merge...")
 
    # Round all timestamps to minute
    for df in [camera_df, sensor_df, weather_df, gps_df]:
        df['timestamp'] = df['timestamp'].dt.round('min')
 
    # Aggregate camera data per minute
    camera_agg = camera_df.groupby('timestamp').agg(
        total_vehicles    = ('total',           'sum'),
        avg_speed_camera  = ('avg_speed_kmh',   'mean'),
        congestion_camera = ('congestion_index', 'mean'),
        incident_count    = ('incident_flag',    'sum'),
    ).reset_index()
 
    # Aggregate sensor data per minute
    sensor_agg = sensor_df.groupby('timestamp').agg(
        sensor_flow_veh_hr   = ('flow_veh_per_hr', 'sum'),
        avg_speed_sensor     = ('speed_kmh',        'mean'),
        avg_occupancy        = ('occupancy_pct',    'mean'),
        congestion_sensor    = ('congestion_index', 'mean'),
    ).reset_index()
 
    # Weather: keep only useful columns
    weather_cols = ['timestamp', 'weather_severity', 'weather_category']
    weather_cols = [c for c in weather_cols if c in weather_df.columns]
    weather_agg  = weather_df[weather_cols].drop_duplicates('timestamp')
 
    # GPS: aggregate per minute
    gps_speed_col = 'speed_kmh' if 'speed_kmh' in gps_df.columns else None
    gps_tti_col   = 'travel_time_index' if 'travel_time_index' in gps_df.columns else None
    gps_agg_cols  = ['timestamp'] + [c for c in [gps_speed_col, gps_tti_col] if c]
    if len(gps_agg_cols) > 1:
        gps_agg = gps_df[gps_agg_cols].groupby('timestamp').mean().reset_index()
        if gps_speed_col:
            gps_agg.rename(columns={gps_speed_col: 'avg_speed_gps'}, inplace=True)
    else:
        gps_agg = None
 
    # Merge all on timestamp
    merged = camera_agg.merge(sensor_agg,  on='timestamp', how='outer')
    merged = merged.merge(weather_agg,     on='timestamp', how='left')
    if gps_agg is not None:
        merged = merged.merge(gps_agg,     on='timestamp', how='left')
 
    # Forward fill weather (changes slowly)
    weather_fill_cols = [c for c in ['weather_severity', 'weather_category'] if c in merged.columns]
    merged[weather_fill_cols] = merged[weather_fill_cols].ffill()
 
    # Sort by time
    merged = merged.sort_values('timestamp').reset_index(drop=True)
 
    # Final congestion label from average of available indices
    ci_cols = [c for c in ['congestion_camera', 'congestion_sensor'] if c in merged.columns]
    merged['congestion_index'] = merged[ci_cols].mean(axis=1).round(4)
    merged['congestion_level'] = pd.cut(
        merged['congestion_index'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
 
    print(f"[Merge] Merged dataset shape: {merged.shape}")
    print(f"[Merge] Columns: {list(merged.columns)}")
    return merged
 
 
def run_all():
    print("=" * 60)
    print("  Traffic Management — Preprocessing Pipeline")
    print("=" * 60)
 
    # Run each preprocessor
    camera_df  = preprocess_camera(
        raw_path    = str(RAW_BASE / "camera/vehicle_counts.csv"),
        output_path = str(PROC_BASE / "camera/vehicle_counts_processed.csv"),
    )
    sensor_df  = preprocess_sensor(
        raw_path    = str(RAW_BASE / "sensor/sensor_readings.csv"),
        output_path = str(PROC_BASE / "sensor/sensor_readings_processed.csv"),
    )
    weather_df = preprocess_weather(
        raw_path    = str(RAW_BASE / "weather/weather.csv"),
        output_path = str(PROC_BASE / "weather/weather_processed.csv"),
    )
    gps_df     = preprocess_gps(
        raw_path    = str(RAW_BASE / "gps/road_flow.csv"),
        output_path = str(PROC_BASE / "gps/road_flow_processed.csv"),
    )
 
    # Merge all
    merged = merge_datasets(camera_df, sensor_df, weather_df, gps_df)
 
    # Save merged dataset
    merged_path = PROC_BASE / "merged_traffic_data.csv"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)
    print(f"\n[Done] Merged dataset saved to {merged_path}")
    print(f"[Done] Shape: {merged.shape}")
    print("\nCongestion distribution:")
    print(merged['congestion_level'].value_counts())
 
    return merged
 
 
if __name__ == "__main__":
    run_all()