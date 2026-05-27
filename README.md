# Traffic Prediction and Management System - Kolkata

A spatio-temporal machine learning system for predicting and managing road traffic congestion across Kolkata's road network. Combines **XGBoost** (tabular prediction) with **STGCN-LSTM** (graph-based spatial-temporal prediction) for high-accuracy congestion forecasting across 10 key road segments.

Integrates **ByteTrack** multi-object tracking (Liu et al., 2024) into the camera pipeline for trajectory-based speed estimation, virtual line flow counting, and improved incident detection.

---

## Results

| Model | RMSE | MAE | R2 | Features |
|-------|------|-----|----|---------|
| XGBoost | 0.0050 | 0.0038 | **0.9981** | 36 |
| STGCN-LSTM | 0.0556 | 0.0368 | **0.9602** | 6 node features |

Trained on 19,739 timesteps (1-minute intervals, March 26 - April 8 2026) across 10 Kolkata road segments.

### Top XGBoost Feature Importances (ByteTrack enabled)

| Feature | Importance | Source |
|---------|-----------|--------|
| `active_tracks` | 56.1% | ByteTrack |
| `vehicle_count` | 30.2% | Camera |
| `flow_rate_veh_hr` | 13.6% | ByteTrack |
| `avg_speed` | 0.05% | Camera + Sensor |
| `occupancy` | 0.03% | IoT Sensor |

ByteTrack features account for **69.7%** of total prediction importance.

---

## Camera Network - 10 Locations

| Node | Camera | Location | Coordinates |
|------|--------|----------|-------------|
| 0 | cam_001 | Esplanade Junction | 22.5726N, 88.3639E |
| 1 | cam_002 | Ultadanga Flyover | 22.5800N, 88.3500E |
| 2 | cam_003 | Tollygunge Metro | 22.5200N, 88.3800E |
| 3 | cam_004 | Howrah Bridge East | 22.5958N, 88.3467E |
| 4 | cam_005 | Park Street Crossing | 22.5553N, 88.3523E |
| 5 | cam_006 | EM Bypass Ruby | 22.5411N, 88.3961E |
| 6 | cam_007 | VIP Road Airport Gate | 22.6054N, 88.3936E |
| 7 | cam_008 | Rashbehari Connector | 22.5354N, 88.3302E |
| 8 | cam_009 | Salt Lake Sector V | 22.5646N, 88.4318E |
| 9 | cam_010 | Joka Tram Depot | 22.4946N, 88.3195E |

---

## Architecture

```
Data Sources
  Camera (10 locations) + IoT Sensors (10) + GPS/HERE + Weather (10 zones) + OSMnx
        |
  ByteTrack Multi-Object Tracking (per camera)
  - Persistent vehicle IDs across frames
  - Trajectory-based speed estimation
  - Virtual line flow counting
  - Stationary track incident detection
        |
  Preprocessing Pipeline
  (clean, engineer features, merge on timestamp)
        |
  +-------------------------------------+
  |         Feature Engineering         |
  |  ByteTrack: active_tracks,          |
  |             flow_rate_veh_hr        |
  |  time encoding, lag features,       |
  |  weather interactions, rolling      |
  |  statistics, cyclical encoding      |
  +-------------------------------------+
        |                    |
  XGBoost Model        STGCN-LSTM Model
  (36 tabular          (GCN spatial +
   features)            LSTM temporal)
        |                    |
     Congestion Score Prediction [0, 1]
        |
  Route Suggestion Engine (Dijkstra)
  - Least-congested path between any 2 nodes
  - Edge weight: 0.6*congestion + 0.3*distance + 0.1*incidents
  - GPS-derived free-flow speeds per segment
```

### XGBoost
- 36 engineered features including ByteTrack tracking metrics, vehicle count, speed lags (1,2,3,6,12), weather interactions, cyclical time encodings, rolling statistics
- Top predictors: `active_tracks` (56%), `vehicle_count` (30%), `flow_rate_veh_hr` (14%)
- Trained with early stopping, 500 estimators, learning rate 0.05

### STGCN-LSTM
- **GCN**: captures spatial relationships between 10 road segment nodes via normalized adjacency matrix
- **LSTM**: captures temporal dependencies over a 12-timestep (12-minute) window
- **Architecture**: STGCNBlock (GCN x2 + BatchNorm + Dropout) -> LSTM (2 layers, hidden=128) -> FC (64) -> output (10 nodes)
- Road network topology: linear backbone (Esplanade-Howrah corridor) + cross-street connections

### ByteTrack Integration
Based on Liu et al. (2024), ByteTrack is integrated into the camera collector to provide:
- **`active_tracks`**: number of vehicles with confirmed persistent IDs in frame
- **`flow_rate_veh_hr`**: vehicles per hour from virtual line crossing counts
- **`avg_speed_kmh`**: trajectory-based speed (displacement between frames / FPS)
- **`incident_count`**: number of track IDs stationary for 75+ consecutive frames

> Reference: Liu, J., Xie, Y., Zhang, Y., & Li, H. (2024). Vehicle Flow Detection
> and Tracking Based on an Improved YOLOv8n and ByteTrack Framework.
> World Electric Vehicle Journal, 16(1), 13. https://doi.org/10.3390/wevj16010013

### Route Suggestion Engine
- Dijkstra's algorithm over the 10-node road graph
- Edge weights derived from real-time congestion predictions
- Road speeds loaded from GPS free-flow data (`free_flow_speed_kmh`)
- Returns path, coordinates, segment colors (green/orange/red), travel time
- JSON output ready for Leaflet.js/Google Maps integration

---

## Dataset

| Source | Raw Rows | Coverage |
|--------|----------|----------|
| Camera feeds | 197,390 | 10 locations x 19,739 timestamps |
| IoT sensors | 197,390 | 10 sensors x 19,739 timestamps |
| Weather | 197,390 | 10 zones x 19,739 timestamps |
| GPS / HERE Maps | 173,881 | Road segments across Kolkata |
| Merged training set | 19,739 | 20 columns, 0 NaN values |

Congestion distribution (balanced): Low 33.4% / Medium 32.8% / High 33.8%

---

## Project Structure

```
Traffic_Management/
|-- collectors/
|   |-- camera_collector.py   # YOLOv8 + ByteTrack (10 cameras)
|   |-- sensor_collector.py   # IoT sensor ingestion
|   |-- gps_collector.py      # HERE Maps GPS data
|   `-- weather_events_collector.py
|-- config/                   # Settings and model config
|-- data/
|   |-- raw/
|   |   |-- camera/           # cam_001.csv ... cam_010.csv
|   |   |-- sensor/           # sensor_readings.csv (10 sensors)
|   |   |-- weather/          # weather.csv (10 zones)
|   |   `-- gps/              # road_flow.csv
|   `-- preprocessed/
|       |-- merged_traffic_data.csv  # 19,739 rows, 20 columns
|       |-- camera/
|       |-- sensor/
|       |-- weather/
|       |-- adjacency.npy     # Road network adjacency matrix
|       `-- edges.json        # Road network edge list
|-- models/
|   |-- stgcn_lstm.py         # STGCN-LSTM architecture
|   |-- xgboost_model.py      # XGBoost with ByteTrack features
|   |-- train_stgcn.py        # STGCN training pipeline
|   `-- evaluate.py           # Evaluation and plotting
|-- preprocessing/            # Per-source preprocessing modules
|-- routing/
|   |-- route_suggester.py    # Dijkstra route engine (GPS speeds)
|   `-- __init__.py
|-- pipeline/                 # Pipeline orchestrator
|-- tests/                    # Unit tests
|-- generate_camera_data.py   # Generate 10-camera data with ByteTrack columns
|-- generate_sensor_data.py   # Generate 10-sensor data
|-- generate_weather_data.py  # Generate 10-zone weather data
|-- train.py                  # Master training script
`-- requirements.txt
```

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/Souraditya/Traffic_Prediction_and_Management.git
cd Traffic_Prediction_and_Management
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Regenerate data (optional)
```bash
python generate_camera_data.py
python generate_sensor_data.py
python generate_weather_data.py
python -m preprocessing.run_preprocessing
```

### 5. Train models
```bash
python train.py               # both models
python train.py --model xgboost
python train.py --model stgcn
```

### 6. Get a route suggestion
```bash
python routing/route_suggester.py --list
python routing/route_suggester.py --from "Esplanade Junction" --to "Salt Lake Sector V"
python routing/route_suggester.py --from "Howrah Bridge East" --to "Joka Tram Depot" --json
```

### Note on missing files
The following are excluded from the repo due to size and are regenerated locally:
- `data/preprocessed/node_features.npy` and `targets.npy` — run `train.py`
- `models/saved/` — run `train.py`
- `evaluation/` — run `train.py`

---

## Data Sources

| Source | Type | Description |
|--------|------|-------------|
| Camera feeds | Synthetic (YOLOv8 + ByteTrack ready) | Vehicle count, speed, flow, incidents |
| IoT sensors | Synthetic (inductive loop + radar) | Flow rate, occupancy, speed |
| GPS / HERE Maps | Synthetic (HERE API ready) | Free-flow speed, travel time index |
| Weather | Synthetic (OpenMeteo ready) | Temperature, rainfall, humidity, wind |
| OSMnx | Real | Kolkata road network graph |

> In production, synthetic collectors are replaced with live feeds from
> Kolkata Traffic Police CCTV (YOLOv8 + ByteTrack on Jetson/Raspberry Pi),
> HERE Maps API, and OpenMeteo.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- XGBoost 1.7+
- ultralytics (YOLOv8 + ByteTrack)
- scikit-learn, pandas, numpy
- See `requirements.txt` for full list

---

## References

Liu, J., Xie, Y., Zhang, Y., & Li, H. (2024). Vehicle Flow Detection and Tracking
Based on an Improved YOLOv8n and ByteTrack Framework. *World Electric Vehicle Journal*,
16(1), 13. https://doi.org/10.3390/wevj16010013

---

## License

MIT License