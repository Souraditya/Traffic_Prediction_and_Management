"""
generate_camera_data.py
-----------------------
Re-generates synthetic camera data for all 10 Kolkata camera locations.
Produces:
  - data/raw/camera/cam_001.csv ... cam_010.csv  (per-camera raw CSVs)
  - data/preprocessed/camera/vehicle_counts_all.csv  (merged)
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone
import random
 
random.seed(42)
np.random.seed(42)
 
# ── Camera registry (all 10 locations) ───────────────────────────────────────
CAMERA_REGISTRY = {
    "cam_001": {"lat": 22.5726, "lon": 88.3639, "location": "Esplanade Junction"},
    "cam_002": {"lat": 22.5800, "lon": 88.3500, "location": "Ultadanga Flyover"},
    "cam_003": {"lat": 22.5200, "lon": 88.3800, "location": "Tollygunge Metro"},
    "cam_004": {"lat": 22.5958, "lon": 88.3467, "location": "Howrah Bridge East"},
    "cam_005": {"lat": 22.5553, "lon": 88.3523, "location": "Park Street Crossing"},
    "cam_006": {"lat": 22.5411, "lon": 88.3961, "location": "EM Bypass Ruby"},
    "cam_007": {"lat": 22.6054, "lon": 88.3936, "location": "VIP Road Airport Gate"},
    "cam_008": {"lat": 22.5354, "lon": 88.3302, "location": "Rashbehari Connector"},
    "cam_009": {"lat": 22.5646, "lon": 88.4318, "location": "Salt Lake Sector V"},
    "cam_010": {"lat": 22.4946, "lon": 88.3195, "location": "Joka Tram Depot"},
}
 
# ── Per-camera traffic characteristics ───────────────────────────────────────
# Each camera has slightly different traffic patterns
CAMERA_PROFILES = {
    "cam_001": {"base_cars": 15, "base_bikes": 8,  "base_buses": 2, "base_trucks": 1, "rush_scale": 3.5},
    "cam_002": {"base_cars": 12, "base_bikes": 6,  "base_buses": 3, "base_trucks": 2, "rush_scale": 3.0},
    "cam_003": {"base_cars": 10, "base_bikes": 10, "base_buses": 4, "base_trucks": 1, "rush_scale": 2.8},
    "cam_004": {"base_cars": 20, "base_bikes": 5,  "base_buses": 5, "base_trucks": 4, "rush_scale": 4.0},
    "cam_005": {"base_cars": 18, "base_bikes": 7,  "base_buses": 2, "base_trucks": 1, "rush_scale": 3.2},
    "cam_006": {"base_cars": 22, "base_bikes": 4,  "base_buses": 3, "base_trucks": 3, "rush_scale": 3.8},
    "cam_007": {"base_cars": 25, "base_bikes": 6,  "base_buses": 4, "base_trucks": 5, "rush_scale": 3.5},
    "cam_008": {"base_cars": 14, "base_bikes": 9,  "base_buses": 3, "base_trucks": 1, "rush_scale": 2.5},
    "cam_009": {"base_cars": 16, "base_bikes": 5,  "base_buses": 2, "base_trucks": 2, "rush_scale": 2.8},
    "cam_010": {"base_cars": 8,  "base_bikes": 12, "base_buses": 3, "base_trucks": 1, "rush_scale": 2.2},
}
 
# ── Time range: match existing merged CSV ────────────────────────────────────
timestamps = pd.date_range(
    start="2026-03-26 00:00:00+00:00",
    end="2026-04-08 16:58:00+00:00",
    freq="1min",
    tz="UTC"
)
T = len(timestamps)
 
# ── Output directories ────────────────────────────────────────────────────────
raw_dir   = Path("data/raw/camera")
prep_dir  = Path("data/preprocessed/camera")
raw_dir.mkdir(parents=True, exist_ok=True)
prep_dir.mkdir(parents=True, exist_ok=True)
 
all_records = []
 
for cam_id, meta in CAMERA_REGISTRY.items():
    profile = CAMERA_PROFILES[cam_id]
    records = []
 
    for ts in timestamps:
        hour    = ts.hour
        is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
        is_peak = (8 <= hour <= 10) or (17 <= hour <= 20)
        scale   = profile["rush_scale"] if is_rush else 1.0
 
        # Add noise
        noise = lambda: np.random.normal(1.0, 0.15)
 
        cars   = max(0, int(profile["base_cars"]   * scale * noise()))
        bikes  = max(0, int(profile["base_bikes"]  * scale * noise()))
        buses  = max(0, int(profile["base_buses"]  * scale * noise()))
        trucks = max(0, int(profile["base_trucks"] * scale * noise()))
        total  = cars + bikes + buses + trucks
 
        # Speed: inversely related to congestion
        base_speed = 45 - (total / 8)
        avg_speed  = max(5.0, round(base_speed + np.random.normal(0, 3), 1))
 
        # Congestion index
        congestion = round(min(1.0, max(0.0, total / 120 + np.random.normal(0, 0.02))), 4)
 
        # Incident: rare, more likely in rush hour
        incident_prob = 0.06 if is_peak else 0.02
        incident = int(np.random.choice([0, 1, 2],
                       p=[1 - incident_prob, incident_prob * 0.7, incident_prob * 0.3]))
 
        # YOLO confidence (mock)
        confidence = round(np.random.uniform(0.78, 0.95), 3)
 
        records.append({
            "timestamp":     ts.isoformat(),
            "camera_id":     cam_id,
            "location":      meta["location"],
            "latitude":      meta["lat"],
            "longitude":     meta["lon"],
            "cars":          cars,
            "motorcycles":   bikes,
            "buses":         buses,
            "trucks":        trucks,
            "total_vehicles": total,
            "avg_speed_kmh": avg_speed,
            "incident_flag": incident > 0,
            "incident_count": incident,
            "confidence":    confidence,
            "congestion_index": congestion,
        })
 
    df_cam = pd.DataFrame(records)
 
    # Save per-camera raw CSV
    cam_path = raw_dir / f"{cam_id}.csv"
    df_cam.to_csv(cam_path, index=False)
    print(f"Saved {cam_path}  ({len(df_cam)} rows)")
 
    all_records.append(df_cam)
 
# ── Save merged preprocessed CSV ─────────────────────────────────────────────
df_all = pd.concat(all_records, ignore_index=True)
df_all = df_all.sort_values(["timestamp", "camera_id"]).reset_index(drop=True)
merged_path = prep_dir / "vehicle_counts_all.csv"
df_all.to_csv(merged_path, index=False)
 
print(f"\nMerged CSV saved -> {merged_path}")
print(f"Total rows: {len(df_all)}")
print(f"Cameras   : {df_all['camera_id'].nunique()}")
print(f"Columns   : {df_all.columns.tolist()}")
print(f"\nSample:")
print(df_all[["timestamp","camera_id","location","total_vehicles","avg_speed_kmh","congestion_index","incident_count"]].head(10).to_string())