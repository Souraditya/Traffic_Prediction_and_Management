import pandas as pd
import numpy as np
from pathlib import Path
 
 
def load_gps_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df
 
 
def clean_gps_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.sort_values('timestamp').reset_index(drop=True)
 
    # Speed must be non-negative
    if 'speed_kmh' in df.columns:
        df['speed_kmh'] = df['speed_kmh'].clip(lower=0, upper=120)
 
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method='linear')
 
    return df
 
 
def engineer_gps_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].apply(
        lambda h: 1 if (8 <= h <= 10 or 17 <= h <= 19) else 0
    )
 
    # Travel time index: ratio of current speed to free-flow speed
    # Free-flow assumed at night (22:00-05:00) average
    if 'speed_kmh' in df.columns:
        night_mask = (df['hour'] >= 22) | (df['hour'] <= 5)
        free_flow_speed = df.loc[night_mask, 'speed_kmh'].mean()
        if pd.isna(free_flow_speed) or free_flow_speed == 0:
            free_flow_speed = 45  # default Kolkata free-flow
        df['travel_time_index'] = (free_flow_speed / df['speed_kmh'].replace(0, np.nan)).fillna(3.0).round(3)
        df['travel_time_index'] = df['travel_time_index'].clip(upper=5.0)
 
    return df
 
 
def preprocess_gps(raw_path: str, output_path: str) -> pd.DataFrame:
    print(f"[GPS] Loading from {raw_path}")
    df = load_gps_data(raw_path)
    print(f"[GPS] Loaded {len(df)} rows")
 
    df = clean_gps_data(df)
    df = engineer_gps_features(df)
    print(f"[GPS] Features engineered: {list(df.columns)}")
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[GPS] Saved to {output_path}")
 
    return df
 
 
if __name__ == "__main__":
    preprocess_gps(
        raw_path="data/raw/gps/road_flow.csv",
        output_path="data/processed/gps/road_flow_processed.csv"
    )
