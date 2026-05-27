"""
Master Preprocessing Orchestrator
Runs all data source preprocessors and produces a merged dataset
ready for model training.

Usage:
    python -m preprocessing.run_preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path

from preprocessing.camera_preprocessor  import preprocess_camera
from preprocessing.sensor_preprocessor  import preprocess_sensor
from preprocessing.weather_preprocessor import preprocess_weather
from preprocessing.gps_preprocessor     import preprocess_gps


RAW_BASE  = Path("data/raw")
PROC_BASE = Path("data/preprocessed")


def load_all_camera_csvs(camera_dir: Path) -> pd.DataFrame:
    """Load and combine all per-camera CSVs (cam_001.csv ... cam_010.csv)."""
    cam_files = sorted(camera_dir.glob("cam_*.csv"))
    if not cam_files:
        fallback = camera_dir / "vehicle_counts.csv"
        if fallback.exists():
            print(f"[Camera] No per-camera CSVs found, loading fallback: {fallback}")
            return pd.read_csv(fallback)
        raise FileNotFoundError(f"No camera CSVs found in {camera_dir}")

    dfs = []
    for f in cam_files:
        df = pd.read_csv(f)
        print(f"[Camera] Loaded {f.name}: {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], utc=True)
    combined = combined.sort_values(['timestamp', 'camera_id']).reset_index(drop=True)
    print(f"[Camera] Combined all cameras: {combined.shape} "
          f"({combined['camera_id'].nunique()} cameras)")
    return combined


def _to_utc_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp is UTC and rounded to the nearest minute."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.round('min')
    return df


def merge_datasets(
    camera_df:  pd.DataFrame,
    sensor_df:  pd.DataFrame,
    weather_df: pd.DataFrame,
    gps_df:     pd.DataFrame,
) -> pd.DataFrame:
    print("\n[Merge] Starting dataset merge...")

    # Ensure all timestamps are UTC and rounded to minute
    camera_df  = _to_utc_minute(camera_df)
    sensor_df  = _to_utc_minute(sensor_df)
    weather_df = _to_utc_minute(weather_df)
    gps_df     = _to_utc_minute(gps_df)

    # ── Camera aggregation ───────────────────────────────────────────────────
    incident_col = 'incident_count' if 'incident_count' in camera_df.columns else 'incident_flag'

    # Build agg dict dynamically so ByteTrack columns are included only when present
    camera_agg_spec = {
        'total_vehicles':   ('total_vehicles',  'sum'),
        'avg_speed_camera': ('avg_speed_kmh',   'mean'),
        'congestion_camera':('congestion_index', 'mean'),
        'incident_count':   (incident_col,       'sum'),
    }
    # ── ByteTrack aggregations ────────────────────────────────────────────────
    if 'active_tracks' in camera_df.columns:
        camera_agg_spec['active_tracks'] = ('active_tracks', 'sum')
    if 'flow_rate_veh_hr' in camera_df.columns:
        camera_agg_spec['flow_rate_veh_hr'] = ('flow_rate_veh_hr', 'sum')

    camera_agg = camera_df.groupby('timestamp').agg(**camera_agg_spec).reset_index()

    # Back-fill ByteTrack columns from total_vehicles if still absent after agg
    if 'active_tracks' not in camera_agg.columns:
        camera_agg['active_tracks'] = camera_agg['total_vehicles']
    if 'flow_rate_veh_hr' not in camera_agg.columns:
        camera_agg['flow_rate_veh_hr'] = camera_agg['total_vehicles'] * 60.0

    print(f"[Merge] Camera agg: {camera_agg.shape}, NaNs: {camera_agg.isnull().sum().sum()}")

    # ── Sensor aggregation ───────────────────────────────────────────────────
    sensor_agg = sensor_df.groupby('timestamp').agg(
        sensor_flow_veh_hr = ('flow_veh_per_hr', 'sum'),
        avg_speed_sensor   = ('speed_kmh',        'mean'),
        avg_occupancy      = ('occupancy_pct',    'mean'),
        congestion_sensor  = ('congestion_index', 'mean'),
    ).reset_index()
    print(f"[Merge] Sensor agg: {sensor_agg.shape}, NaNs: {sensor_agg.isnull().sum().sum()}")

    # ── Weather aggregation ──────────────────────────────────────────────────
    weather_cols = ['timestamp', 'weather_severity', 'weather_category',
                    'temperature_c', 'precipitation_mm', 'wind_speed_kmh']
    weather_cols = [c for c in weather_cols if c in weather_df.columns]
    weather_agg  = weather_df[weather_cols].drop_duplicates('timestamp').copy()
    weather_agg.rename(columns={
        'temperature_c':    'temperature',
        'precipitation_mm': 'rainfall',
        'wind_speed_kmh':   'wind_speed',
    }, inplace=True)
    print(f"[Merge] Weather agg: {weather_agg.shape}")

    # ── GPS aggregation ──────────────────────────────────────────────────────
    gps_speed_col = 'current_speed_kmh' if 'current_speed_kmh' in gps_df.columns else \
                    'speed_kmh' if 'speed_kmh' in gps_df.columns else None
    if gps_speed_col:
        gps_agg = gps_df.groupby('timestamp')[[gps_speed_col]].mean().reset_index()
        gps_agg.rename(columns={gps_speed_col: 'avg_speed_gps'}, inplace=True)
    else:
        gps_agg = None
    print(f"[Merge] GPS agg: {gps_agg.shape if gps_agg is not None else 'skipped'}")

    # ── Merge on timestamp ───────────────────────────────────────────────────
    merged = camera_agg.merge(sensor_agg, on='timestamp', how='outer')
    merged = merged.merge(weather_agg,    on='timestamp', how='left')
    if gps_agg is not None:
        merged = merged.merge(gps_agg, on='timestamp', how='left')
        if 'avg_speed_gps' in merged.columns:
            merged['avg_speed_gps'] = merged['avg_speed_gps'].fillna(merged['avg_speed_camera'])

    # ── Fill weather columns ─────────────────────────────────────────────────
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    hour  = pd.to_datetime(merged['timestamp']).dt.hour
    month = pd.to_datetime(merged['timestamp']).dt.month
    T     = len(merged)
    np.random.seed(42)

    if 'temperature' not in merged.columns or merged['temperature'].isna().all():
        merged['temperature'] = (
            30 + 4 * np.sin(np.pi * (hour - 6) / 12).clip(0)
            + np.random.normal(0, 0.5, T)
        ).round(1).clip(24, 40)
    else:
        merged['temperature'] = merged['temperature'].ffill().bfill().fillna(30.0)

    if 'rainfall' not in merged.columns or merged['rainfall'].isna().all():
        rain_prob = np.where((month == 4) & (hour.between(14, 18)), 0.15, 0.03)
        merged['rainfall'] = np.where(
            np.random.random(T) < rain_prob,
            np.random.exponential(2, T).round(1).clip(0, 20), 0.0
        )
    else:
        merged['rainfall'] = merged['rainfall'].ffill().bfill().fillna(0.0)

    merged['humidity'] = (
        65 - 10 * np.sin(np.pi * (hour - 6) / 12).clip(0)
        + merged['rainfall'] * 2
        + np.random.normal(0, 2, T)
    ).round(1).clip(45, 95)

    merged['wind_speed'] = (
        5 + 3 * np.sin(np.pi * (hour - 8) / 10).clip(0)
        + np.random.exponential(1, T)
    ).round(1).clip(0, 20)

    if 'weather_severity' not in merged.columns or merged['weather_severity'].isna().all():
        merged['weather_severity'] = np.where(merged['rainfall'] > 5, 2,
                                     np.where(merged['rainfall'] > 0, 1, 0))

    if 'weather_category' not in merged.columns or merged['weather_category'].isna().all():
        merged['weather_category'] = np.where(merged['rainfall'] > 5, 'rainy',
                                     np.where(merged['rainfall'] > 0, 'drizzle',
                                     np.where(merged['temperature'] > 36, 'hot_clear', 'clear')))

    # ── Congestion index and level ────────────────────────────────────────────
    ci_cols = [c for c in ['congestion_camera', 'congestion_sensor'] if c in merged.columns]
    merged['congestion_index'] = merged[ci_cols].mean(axis=1).round(4)

    q33 = merged['congestion_index'].quantile(0.33)
    q66 = merged['congestion_index'].quantile(0.66)
    merged['congestion_level'] = pd.cut(
        merged['congestion_index'],
        bins=[0, q33, q66, 1.01],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )

    # ── Add realistic incident_count if missing ───────────────────────────────
    if 'incident_count' not in merged.columns or merged['incident_count'].isna().all():
        is_peak  = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
        high_cong = merged['congestion_index'] > q66
        prob = np.where(is_peak, 0.06, 0.02)
        prob = np.where(high_cong, prob + 0.04, prob)
        merged['incident_count'] = np.where(
            np.random.random(T) < prob * 0.7, 1,
            np.where(np.random.random(T) < prob, 2, 0)
        ).astype(int)

    # ── Final null check ──────────────────────────────────────────────────────
    null_counts = merged.isnull().sum()
    remaining   = null_counts[null_counts > 0]
    if len(remaining):
        print(f"[Merge] WARNING - remaining NaNs:\n{remaining}")
    else:
        print("[Merge] No NaN values in merged dataset.")

    print(f"[Merge] Merged dataset shape: {merged.shape}")
    print(f"[Merge] Columns: {list(merged.columns)}")
    return merged


def run_all():
    print("=" * 60)
    print("  Traffic Management - Preprocessing Pipeline")
    print("=" * 60)

    # Load all 10 camera CSVs
    camera_raw = load_all_camera_csvs(RAW_BASE / "camera")
    combined_path = PROC_BASE / "camera/vehicle_counts_all.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    camera_raw.to_csv(combined_path, index=False)
    print(f"[Camera] Combined raw saved -> {combined_path}")

    # Run each preprocessor
    camera_df = preprocess_camera(
        raw_path    = str(combined_path),
        output_path = str(PROC_BASE / "camera/vehicle_counts_processed.csv"),
    )
    sensor_df = preprocess_sensor(
        raw_path    = str(RAW_BASE / "sensor/sensor_readings.csv"),
        output_path = str(PROC_BASE / "sensor/sensor_readings_processed.csv"),
    )
    weather_df = preprocess_weather(
        raw_path    = str(RAW_BASE / "weather/weather.csv"),
        output_path = str(PROC_BASE / "weather/weather_processed.csv"),
    )
    gps_df = preprocess_gps(
        raw_path    = str(RAW_BASE / "gps/road_flow.csv"),
        output_path = str(PROC_BASE / "gps/road_flow_processed.csv"),
    )

    # Merge all
    merged = merge_datasets(camera_df, sensor_df, weather_df, gps_df)

    # Save
    merged_path = PROC_BASE / "merged_traffic_data.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\n[Done] Merged dataset saved -> {merged_path}")
    print(f"[Done] Shape: {merged.shape}")
    print("\nCongestion distribution:")
    print(merged['congestion_level'].value_counts())

    return merged


if __name__ == "__main__":
    run_all()