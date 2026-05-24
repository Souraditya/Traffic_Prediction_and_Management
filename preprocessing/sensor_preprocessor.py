import pandas as pd
import numpy as np
from pathlib import Path
 
 
def load_sensor_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df
 
 
def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate sensor+timestamp pairs
    df = df.drop_duplicates(subset=['sensor_id', 'timestamp'])
 
    # Clip physically impossible values using actual column names
    df['speed_kmh']       = df['speed_kmh'].clip(lower=0, upper=120)
    df['occupancy_pct']   = df['occupancy_pct'].clip(lower=0, upper=100)
    df['flow_veh_per_hr'] = df['flow_veh_per_hr'].clip(lower=0)
    df['headway_sec']     = df['headway_sec'].clip(lower=0)
 
    # Interpolate missing values per sensor
    for col in ['speed_kmh', 'occupancy_pct', 'flow_veh_per_hr', 'headway_sec']:
        if df[col].isna().any():
            df[col] = df.groupby('sensor_id')[col].transform(
                lambda x: x.interpolate(method='linear')
            )
 
    return df.reset_index(drop=True)
 
 
def engineer_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Time features
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].apply(
        lambda h: 1 if (8 <= h <= 10 or 17 <= h <= 19) else 0
    )
 
    # Congestion index: low speed + high occupancy = congested
    max_speed = df['speed_kmh'].max()
    df['congestion_index'] = (
        0.5 * (df['occupancy_pct'] / 100) +
        0.5 * (1 - df['speed_kmh'] / max_speed if max_speed > 0 else 0)
    ).round(4)
 
    # Keep existing congestion_level if already present in raw data
    if 'congestion_level' not in df.columns:
        df['congestion_level'] = pd.cut(
            df['congestion_index'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
 
    return df
 
 
def preprocess_sensor(raw_path: str, output_path: str) -> pd.DataFrame:
    print(f"[Sensor] Loading from {raw_path}")
    df = load_sensor_data(raw_path)
    print(f"[Sensor] Loaded {len(df)} rows, {df['sensor_id'].nunique()} sensors")
 
    df = clean_sensor_data(df)
    print(f"[Sensor] After cleaning: {len(df)} rows")
 
    df = engineer_sensor_features(df)
    print(f"[Sensor] Features engineered: {list(df.columns)}")
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Sensor] Saved to {output_path}")
 
    return df
 
 
if __name__ == "__main__":
    preprocess_sensor(
        raw_path="data/raw/sensor/sensor_readings.csv",
        output_path="data/processed/sensor/sensor_readings_processed.csv"
    )