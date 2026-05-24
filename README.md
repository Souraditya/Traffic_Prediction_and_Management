# Traffic Prediction and Management System — Kolkata

A spatio-temporal machine learning system for predicting and managing road traffic congestion across Kolkata's road network. Combines **XGBoost** (tabular prediction) with **STGCN-LSTM** (graph-based spatial-temporal prediction) for high-accuracy congestion forecasting.

---

## Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| XGBoost | 0.0195 | 0.0114 | **0.990** |
| STGCN-LSTM | 0.0792 | 0.0496 | **0.857** |

---

## Architecture

```
Data Sources (Camera, IoT Sensors, GPS, Weather, OSMnx)
        ↓
  Preprocessing Pipeline
        ↓
  ┌─────────────────────────────────────┐
  │         Feature Engineering         │
  │  (time encoding, lag features,      │
  │   weather interactions, rolling     │
  │   statistics)                       │
  └─────────────────────────────────────┘
        ↓                    ↓
  XGBoost Model        STGCN-LSTM Model
  (tabular features)   (graph + temporal)
        ↓                    ↓
     Congestion Score Prediction [0, 1]
```

### XGBoost
- 34 engineered features including vehicle count, speed lags, weather interactions, cyclical time encodings
- Top predictors: `vehicle_count`, `speed_lag_1`, `sensor_flow`, `avg_speed`

### STGCN-LSTM
- Graph Convolutional Network (GCN) captures spatial relationships between 10 road segment nodes
- LSTM captures temporal dependencies over a 12-timestep window
- Kolkata road network topology encoded as a normalized adjacency matrix

---

## Project Structure

```
Traffic_Management/
├── collectors/          # Data collection modules (camera, GPS, IoT, weather)
├── config/              # Settings and model configuration
├── data/
│   └── preprocessed/    # Merged dataset and graph data files
├── models/              # Model definitions and training scripts
│   ├── stgcn_lstm.py    # STGCN-LSTM architecture
│   ├── xgboost_model.py # XGBoost trainer
│   ├── train_stgcn.py   # STGCN training pipeline
│   └── evaluate.py      # Evaluation and plotting
├── pipeline/            # Orchestration
├── preprocessing/       # Data preprocessing modules
├── tests/               # Unit tests
├── train.py             # Master training script
└── requirements.txt     # Dependencies
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

### 4. Run training
```bash
# Train both models
python train.py

# Train only XGBoost
python train.py --model xgboost

# Train only STGCN-LSTM
python train.py --model stgcn
```

### 5. View results
Training metrics are printed to the console and saved to `training.log`.
Evaluation plots and metrics are saved to `evaluation/` after each run.

---

## Data Sources

| Source | Description |
|--------|-------------|
| Camera feeds | Vehicle count, speed, congestion index |
| IoT sensors | Flow rate, occupancy |
| GPS / HERE Maps | Average speed across segments |
| OpenMeteo | Weather (temperature, rainfall, humidity, wind) |
| OSMnx | Kolkata road network graph |

---

## Requirements

- Python 3.10+
- PyTorch
- XGBoost
- scikit-learn
- pandas, numpy
- See `requirements.txt` for full list

---

## License

MIT License
