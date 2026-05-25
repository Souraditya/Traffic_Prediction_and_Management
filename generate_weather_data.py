"""
generate_weather_data.py
------------------------
Generates realistic weather data for all 10 Kolkata camera locations,
covering the full timestamp range of the merged traffic dataset.
Kolkata March-April: pre-monsoon, hot and humid with occasional showers.
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
 
np.random.seed(42)
 
# Load merged CSV for timestamp range
df_merged = pd.read_csv('data/preprocessed/merged_traffic_data.csv')
df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
timestamps = df_merged['timestamp'].values
T = len(timestamps)
 
# 10 weather zones — one per camera location
# Slight micro-climate variation (coastal vs inland, north vs south)
WEATHER_ZONES = {
    "zone_001": {"location": "Esplanade Junction",    "lat": 22.5726, "lon": 88.3639, "coastal_factor": 1.05},
    "zone_002": {"location": "Ultadanga Flyover",     "lat": 22.5800, "lon": 88.3500, "coastal_factor": 1.02},
    "zone_003": {"location": "Tollygunge Metro",      "lat": 22.5200, "lon": 88.3800, "coastal_factor": 1.03},
    "zone_004": {"location": "Howrah Bridge East",    "lat": 22.5958, "lon": 88.3467, "coastal_factor": 1.08},
    "zone_005": {"location": "Park Street Crossing",  "lat": 22.5553, "lon": 88.3523, "coastal_factor": 1.04},
    "zone_006": {"location": "EM Bypass Ruby",        "lat": 22.5411, "lon": 88.3961, "coastal_factor": 1.01},
    "zone_007": {"location": "VIP Road Airport Gate", "lat": 22.6054, "lon": 88.3936, "coastal_factor": 1.00},
    "zone_008": {"location": "Rashbehari Connector",  "lat": 22.5354, "lon": 88.3302, "coastal_factor": 1.03},
    "zone_009": {"location": "Salt Lake Sector V",    "lat": 22.5646, "lon": 88.4318, "coastal_factor": 1.02},
    "zone_010": {"location": "Joka Tram Depot",       "lat": 22.4946, "lon": 88.3195, "coastal_factor": 1.06},
}
 
hour_vals  = pd.to_datetime(df_merged['timestamp']).dt.hour.values
month_vals = pd.to_datetime(df_merged['timestamp']).dt.month.values
 
records = []
for zone_id, meta in WEATHER_ZONES.items():
    cf = meta["coastal_factor"]  # coastal zones slightly more humid/rainy
 
    for i in range(T):
        hour  = hour_vals[i]
        month = month_vals[i]
 
        # Temperature: 25-38°C, peaks 14:00-15:00
        temp = round(
            30 + 4 * max(0, np.sin(np.pi * (hour - 6) / 12))
            + np.random.normal(0, 0.5), 1
        )
        temp = float(np.clip(temp, 24, 40))
 
        # Rainfall: pre-monsoon, more likely afternoons in April
        rain_prob = 0.15 * cf if (month == 4 and 14 <= hour <= 18) else 0.03 * cf
        rainfall  = round(float(np.random.exponential(2)), 1) if np.random.random() < rain_prob else 0.0
        rainfall  = float(np.clip(rainfall, 0, 20))
 
        # Humidity: 50-90%, higher near coast and when raining
        humidity = round(
            60 * cf - 10 * max(0, np.sin(np.pi * (hour - 6) / 12))
            + rainfall * 2 + np.random.normal(0, 2), 1
        )
        humidity = float(np.clip(humidity, 45, 95))
 
        # Wind speed: 0-20 km/h
        wind_speed = round(
            5 + 3 * max(0, np.sin(np.pi * (hour - 8) / 10))
            + float(np.random.exponential(1)), 1
        )
        wind_speed = float(np.clip(wind_speed, 0, 20))
 
        # Visibility: reduced in rain/humidity
        visibility = round(float(np.clip(10 - rainfall * 0.5 - (humidity - 60) * 0.05 + np.random.normal(0, 0.5), 2, 10)), 1)
 
        # Weather category and severity
        if rainfall > 5:
            weather_category = 'rainy'
            weather_severity = 2
            traffic_impact   = 'high'
        elif rainfall > 0:
            weather_category = 'drizzle'
            weather_severity = 1
            traffic_impact   = 'moderate'
        elif temp > 36:
            weather_category = 'hot_clear'
            weather_severity = 1
            traffic_impact   = 'low'
        else:
            weather_category = 'clear'
            weather_severity = 0
            traffic_impact   = 'none'
 
        records.append({
            "zone_id":          zone_id,
            "location":         meta["location"],
            "timestamp":        pd.Timestamp(timestamps[i]).isoformat(),
            "latitude":         meta["lat"],
            "longitude":        meta["lon"],
            "temperature_c":    temp,
            "rainfall_mm":      rainfall,
            "humidity_pct":     humidity,
            "wind_speed_kmh":   wind_speed,
            "visibility_km":    visibility,
            "weather_category": weather_category,
            "weather_severity": weather_severity,
            "traffic_impact":   traffic_impact,
        })
 
df_weather = pd.DataFrame(records)
 
# Save raw weather
Path("data/raw/weather").mkdir(parents=True, exist_ok=True)
df_weather.to_csv("data/raw/weather/weather.csv", index=False)
 
print(f"Saved weather.csv")
print(f"Shape: {df_weather.shape}")
print(f"Zones: {df_weather['zone_id'].nunique()} -> {sorted(df_weather['zone_id'].unique())}")
print(f"Rows per zone: {T}")
print(f"\nWeather distribution:")
print(df_weather['weather_category'].value_counts())
print(f"\nSample:")
print(df_weather[df_weather['zone_id']=='zone_001'][['timestamp','temperature_c','rainfall_mm','humidity_pct','weather_category']].head(5).to_string())