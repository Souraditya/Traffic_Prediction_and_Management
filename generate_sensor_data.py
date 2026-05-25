"""
generate_sensor_data.py
-----------------------
Generates realistic IoT sensor data for 5 sensors across Kolkata,
covering the full timestamp range of the merged traffic dataset.
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
 
np.random.seed(42)
 
# Load merged CSV to get timestamp range
df_merged = pd.read_csv('data/preprocessed/merged_traffic_data.csv')
df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
timestamps = df_merged['timestamp'].values
 
T = len(timestamps)
 
# 5 sensor locations mapped to key road segments
SENSORS = {
    "sen_001": {"road_segment": "Esplanade-Dharmatala",   "lat": 22.5726, "lon": 88.3639, "type": "inductive_loop"},
    "sen_002": {"road_segment": "Ultadanga-VIP Crossing", "lat": 22.5800, "lon": 88.3500, "type": "radar"},
    "sen_003": {"road_segment": "Tollygunge-Regent",      "lat": 22.5200, "lon": 88.3800, "type": "inductive_loop"},
    "sen_004": {"road_segment": "Howrah-Foreshore Road",  "lat": 22.5958, "lon": 88.3467, "type": "radar"},
    "sen_005": {"road_segment": "Park Street-AJC Bose",   "lat": 22.5553, "lon": 88.3523, "type": "inductive_loop"},
}
 
# Base profiles per sensor
PROFILES = {
    "sen_001": {"base_flow": 800,  "base_speed": 35, "base_occ": 0.45},
    "sen_002": {"base_flow": 650,  "base_speed": 42, "base_occ": 0.38},
    "sen_003": {"base_flow": 500,  "base_speed": 28, "base_occ": 0.52},
    "sen_004": {"base_flow": 950,  "base_speed": 38, "base_occ": 0.48},
    "sen_005": {"base_flow": 720,  "base_speed": 32, "base_occ": 0.42},
}
 
# Use camera congestion as base signal
congestion = df_merged['congestion_index'].values
hour_vals  = pd.to_datetime(df_merged['timestamp']).dt.hour.values
 
records = []
for sen_id, meta in SENSORS.items():
    p = PROFILES[sen_id]
    for i in range(T):
        ts      = timestamps[i]
        hour    = hour_vals[i]
        cong    = congestion[i]
        is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
        rush_m  = 1.6 if is_rush else 1.0
 
        flow     = max(0, int(p["base_flow"] * rush_m * (1 + cong * 0.5) + np.random.normal(0, 30)))
        speed    = max(5, round(p["base_speed"] * (1 - cong * 0.6) + np.random.normal(0, 2), 1))
        occupancy = round(min(100, p["base_occ"] * 100 * (1 + cong * 0.8) + np.random.normal(0, 2)), 1)
        headway  = round(max(1, 3600 / flow) if flow > 0 else 999, 1)
 
        records.append({
            "sensor_id":       sen_id,
            "timestamp":       pd.Timestamp(ts).isoformat(),
            "latitude":        meta["lat"],
            "longitude":       meta["lon"],
            "road_segment":    meta["road_segment"],
            "speed_kmh":       speed,
            "occupancy_pct":   occupancy,
            "flow_veh_per_hr": flow,
            "headway_sec":     headway,
            "sensor_type":     meta["type"],
            "quality_flag":    1,
        })
 
df_sensor = pd.DataFrame(records)
 
# Save
Path("data/raw/sensor").mkdir(parents=True, exist_ok=True)
df_sensor.to_csv("data/raw/sensor/sensor_readings.csv", index=False)
 
print(f"Saved sensor_readings.csv")
print(f"Shape: {df_sensor.shape}")
print(f"Sensors: {df_sensor['sensor_id'].nunique()}")
print(f"Timestamp range: {df_sensor['timestamp'].min()} -> {df_sensor['timestamp'].max()}")
print(f"\nSample:")
print(df_sensor.head(5).to_string())
