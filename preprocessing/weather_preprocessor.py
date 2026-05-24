import pandas as pd
import numpy as np
from pathlib import Path
 
 
def load_weather_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df
 
 
def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
 
    # Interpolate missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method='linear')
 
    return df
 
 
def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    df['hour']       = df['timestamp'].dt.hour
    df['is_daytime'] = df['hour'].between(6, 20).astype(int)
 
    # Weather severity score (0–1): affects traffic speed
    # Uses rainfall and visibility if present, else defaults to 0
    severity = pd.Series(np.zeros(len(df)), index=df.index)
 
    if 'rainfall_mm' in df.columns:
        rain_norm = (df['rainfall_mm'] / df['rainfall_mm'].max().clip(min=1))
        severity += 0.5 * rain_norm
 
    if 'visibility_km' in df.columns:
        vis_norm = 1 - (df['visibility_km'] / df['visibility_km'].max().clip(min=1))
        severity += 0.3 * vis_norm
 
    if 'wind_speed_kmh' in df.columns:
        wind_norm = (df['wind_speed_kmh'] / df['wind_speed_kmh'].max().clip(min=1))
        severity += 0.2 * wind_norm
 
    df['weather_severity'] = severity.clip(0, 1).round(4)
 
    # Weather category
    df['weather_category'] = pd.cut(
        df['weather_severity'],
        bins=[0, 0.2, 0.5, 1.0],
        labels=['clear', 'moderate', 'severe'],
        include_lowest=True
    )
 
    return df
 
 
def preprocess_weather(raw_path: str, output_path: str) -> pd.DataFrame:
    print(f"[Weather] Loading from {raw_path}")
    df = load_weather_data(raw_path)
    print(f"[Weather] Loaded {len(df)} rows")
 
    df = clean_weather_data(df)
    df = engineer_weather_features(df)
    print(f"[Weather] Features engineered: {list(df.columns)}")
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Weather] Saved to {output_path}")
 
    return df
 
 
if __name__ == "__main__":
    preprocess_weather(
        raw_path="data/raw/weather/weather.csv",
        output_path="data/processed/weather/weather_processed.csv"
    )
