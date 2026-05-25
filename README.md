# Traffic Prediction and Management System - Kolkata

A spatio-temporal machine learning system for predicting and managing road traffic congestion across Kolkata's road network. Combines **XGBoost** (tabular prediction) with **STGCN-LSTM** (graph-based spatial-temporal prediction) for high-accuracy congestion forecasting across 10 key road segments.

---

## Results

| Model | RMSE | MAE | R2 |
|-------|------|-----|----|
| XGBoost | 0.0006 | 0.0004 | **0.9999** |
| STGCN-LSTM | 0.0556 | 0.0368 | **0.9602** |

Trained on 19,739 timesteps (1-minute intervals, March 26 - April 8 2026) across 10 Kolkata road segments.

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
  Preprocessing Pipeline
  (clean, engineer features, merge on timestamp)
        |
  +-------------------------------------+
  |         Feature Engineering         |
  |  time encoding, lag features,       |
  |  weather interactions, rolling      |
  |  statistics, cyclical encoding      |
  +-------------------------------------+
        |                    |
  XGBoost Model        STGCN-LSTM Model
  (34 tabular          (GCN spatial +
   features)            LSTM temporal)
        |                    |
     Congestion Score Prediction [0, 1]
```

### XGBoost
- 34 engineered features: vehicle count, speed lags (1,2,3,6,12), weather interactions, cyclical time encodings, rolling statistics
- Top predictors: `vehicle_count` (88%), `avg_speed` (11%), `occupancy` (0.9%)
- Trained with early stopping, 500 estimators, learning rate 0.05

### STGCN-LSTM
- **GCN**: captures spatial relationships between 10 road segment nodes via normalized adjacency matrix
- **LSTM**: captures temporal dependencies over a 12-timestep (12-minute) window
- **Architecture**: STGCNBlock (GCN x2 + BatchNorm + Dropout) -> LSTM (2 layers, hidden=128) -> FC (64) -> output (10 nodes)
- Road network topology: linear backbone (Esplanade-Howrah corridor) + cross-street connections

---

## Dataset

| Source | Raw Rows | Coverage |
|--------|----------|----------|
| Camera feeds | 197,390 | 10 locations x 19,739 timestamps |
| IoT sensors | 197,390 | 10 sensors x 19,739 timestamps |
| Weather | 197,390 | 10 zones x 19,739 timestamps |
| GPS / HERE Maps | 173,881 | Road segments across Kolkata |
| Merged training set | 19,739 | 18 features, 0 NaN values |

Congestion distribution (balanced): Low 33.2% / Medium 32.9% / High 33.8%

---

## Project Structure

```
Traffic_Management/
|-- collectors/               # Data collection modules
|   |-- camera_collector.py   # YOLOv8 vehicle detection (10 cameras)
|   |-- sensor_collector.py   # IoT sensor ingestion
|   |-- gps_collector.py      # HERE Maps GPS data
|   `-- weather_events_collector.py
|-- config/                   # Settings and model config
|-- data/
|   |-- raw/                  # Raw data per source
|   |   |-- camera/           # cam_001.csv ... cam_010.csv
|   |   |-- sensor/           # sensor_readings.csv (10 sensors)
|   |   |-- weather/          # weather.csv (10 zones)
|   |   `-- gps/              # road_flow.csv
|   `-- preprocessed/         # Processed and merged data
|       |-- merged_traffic_data.csv
|       |-- camera/
|       |-- sensor/
|       |-- weather/
|       |-- adjacency.npy     # Road network adjacency matrix
|       `-- edges.json        # Road network edge list
|-- models/
|   |-- stgcn_lstm.py         # STGCN-LSTM architecture
|   |-- xgboost_model.py      # XGBoost trainer with feature engineering
|   |-- train_stgcn.py        # STGCN training pipeline
|   `-- evaluate.py           # Evaluation and plotting
|-- preprocessing/            # Preprocessing modules per source
|-- pipeline/                 # Pipeline orchestrator
|-- tests/                    # Unit tests
|-- generate_camera_data.py   # Generate 10-camera synthetic data
|-- generate_sensor_data.py   # Generate 10-sensor synthetic data
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

### 4. (Optional) Regenerate data from scratch
```bash
python generate_camera_data.py
python generate_sensor_data.py
python generate_weather_data.py
python -m preprocessing.run_preprocessing
```

### 5. Train models
```bash
# Train both models
python train.py

# Train only XGBoost
python train.py --model xgboost

# Train only STGCN-LSTM
python train.py --model stgcn
```

### Note on missing files
The following files are excluded from the repo due to size and are regenerated locally:
- `data/preprocessed/node_features.npy` and `targets.npy` — generated by `train.py`
- `data/raw/sensor/sensor_readings.csv` — generated by `generate_sensor_data.py`
- `data/preprocessed/sensor/sensor_readings_processed.csv` — generated by `python -m preprocessing.run_preprocessing`
- `models/saved/` — generated by `train.py`
- `evaluation/` — generated by `train.py`

Run steps 4 and 5 above to regenerate everything from scratch.

### 6. View results
Training metrics are printed to the console and saved to `training.log`.
Evaluation plots and metrics are saved to `evaluation/` after each run.

---

## Data Sources

| Source | Type | Description |
|--------|------|-------------|
| Camera feeds | Synthetic (YOLOv8 ready) | Vehicle count by type, speed, congestion per location |
| IoT sensors | Synthetic (inductive loop + radar) | Flow rate, occupancy, speed per road segment |
| GPS / HERE Maps | Synthetic (HERE API ready) | Average speed, travel time index per road segment |
| Weather | Synthetic (OpenMeteo ready) | Temperature, rainfall, humidity, wind per zone |
| OSMnx | Real | Kolkata road network graph |

> In production, synthetic collectors are replaced with live API feeds from Kolkata Traffic Police CCTV, HERE Maps, and OpenMeteo.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- XGBoost 1.7+
- scikit-learn
- pandas, numpy
- ultralytics (YOLOv8)
- See `requirements.txt` for full list

---

## License

MIT License
