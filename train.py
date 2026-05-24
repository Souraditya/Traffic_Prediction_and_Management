"""
Master Training Orchestrator for Traffic Congestion Prediction.
Runs XGBoost and/or STGCN-LSTM training end-to-end from preprocessed data.
 
Usage:
    python train.py --model xgboost
    python train.py --model stgcn
    python train.py --model both
"""
 
import argparse
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
 
# Fix Unicode output on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train")
 
 
# --- Paths -------------------------------------------------------------------
PREPROCESSED_DIR = Path("data/preprocessed")
MODELS_DIR       = Path("models/saved")
EVAL_DIR         = Path("evaluation")
 
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
 
 
# --- XGBoost Pipeline --------------------------------------------------------
 
def run_xgboost(tune: bool = False):
    from models.xgboost_model import XGBoostTrafficModel
    from models.evaluate import ModelEvaluator
 
    logger.info("=" * 60)
    logger.info("  XGBoost Training Pipeline")
    logger.info("=" * 60)
 
    data_path = PREPROCESSED_DIR / "merged_traffic_data.csv"
    if not data_path.exists():
        logger.warning(f"{data_path} not found - trying fallback synthetic data.")
        df = _load_synthetic_fallback()
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        logger.info(f"Loaded {len(df):,} rows from {data_path}")
 
    trainer = XGBoostTrafficModel(model_dir=str(MODELS_DIR))
    metrics = trainer.train(df, tune_hyperparams=tune)
 
    logger.info("\nXGBoost Metrics:")
    for k, v in metrics.items():
        if k != "top_features":
            logger.info(f"  {k}: {v:.4f}")
    logger.info(f"  Top features: {metrics['top_features'][:5]}")
 
    trainer.save("xgb_traffic")
 
    from models.xgboost_model import build_features, FEATURE_COLS, TARGET_COL
    df_fe  = build_features(df)
    feats  = [c for c in FEATURE_COLS if c in df_fe.columns]
    split  = int(len(df_fe) * 0.8)
    X_test = df_fe[feats].values[split:]
    y_test = df_fe[TARGET_COL].values[split:]
    y_pred = trainer.model.predict(X_test)
 
    evaluator = ModelEvaluator(output_dir=str(EVAL_DIR))
    evaluator.evaluate(
        y_test, y_pred,
        model_name="xgboost",
        feature_importances=metrics["top_features"],
    )
    return metrics
 
 
# --- STGCN-LSTM Pipeline -----------------------------------------------------
 
def run_stgcn(config: dict = None):
    from models.train_stgcn import STGCNTrainer, TrafficGraphDataset, build_adjacency_matrix
    from models.evaluate import ModelEvaluator
    import torch
 
    logger.info("=" * 60)
    logger.info("  STGCN-LSTM Training Pipeline")
    logger.info("=" * 60)
 
    cfg = config or {
        "node_features": 6,
        "num_nodes":     10,
        "gcn_hidden":    64,
        "gcn_out":       32,
        "lstm_hidden":   128,
        "lstm_layers":   2,
        "dropout":       0.3,
        "lr":            1e-3,
        "weight_decay":  1e-4,
        "batch_size":    32,
        "epochs":        50,
        "early_stop_patience": 15,
        "seq_len":       12,
        "save_dir":      str(MODELS_DIR),
    }
 
    node_feat_path = PREPROCESSED_DIR / "node_features.npy"
    target_path    = PREPROCESSED_DIR / "targets.npy"
    adj_path       = PREPROCESSED_DIR / "adjacency.npy"
 
    if node_feat_path.exists():
        node_features = np.load(node_feat_path)
        targets       = np.load(target_path)
        adj           = np.load(adj_path)
        logger.info(f"Loaded graph data: {node_features.shape}, targets: {targets.shape}")
    else:
        logger.warning("Graph data not found - generating synthetic graph data.")
        node_features, targets, adj = _synthetic_graph_data(
            T=2000, N=cfg["num_nodes"], F=cfg["node_features"]
        )
 
    seq_len = cfg["seq_len"]
    split   = int(len(node_features) * 0.8)
 
    train_ds = TrafficGraphDataset(node_features[:split], targets[:split], adj, seq_len)
    val_ds   = TrafficGraphDataset(node_features[split:], targets[split:], adj, seq_len)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
 
    trainer = STGCNTrainer(cfg)
    trainer.fit(train_ds, val_ds)
 
    trainer.load_checkpoint("best")
    trainer.model.eval()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    device = trainer.device
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y, adj_b in val_loader:
            x, y, adj_b = x.to(device), y.to(device), adj_b[0].to(device)
            pred, _ = trainer.model(x, adj_b)
            all_true.extend(y.cpu().numpy().flatten())
            all_pred.extend(pred.cpu().numpy().flatten())
 
    evaluator = ModelEvaluator(output_dir=str(EVAL_DIR))
    evaluator.evaluate(
        np.array(all_true), np.array(all_pred),
        model_name="stgcn_lstm",
        loss_history=trainer.history,
    )
 
 
# --- Synthetic fallback helpers ----------------------------------------------
 
def _load_synthetic_fallback() -> pd.DataFrame:
    for pattern in ["data/raw/*.csv", "data/*.csv", "*.csv"]:
        import glob
        files = glob.glob(pattern)
        if files:
            dfs = [pd.read_csv(f) for f in files[:3]]
            df  = pd.concat(dfs, ignore_index=True)
            if "timestamp" not in df.columns:
                df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="15min")
            if "congestion_score" not in df.columns:
                df["congestion_score"] = np.random.uniform(0, 1, len(df))
            return df
 
    logger.warning("No CSVs found - generating fully synthetic tabular data.")
    T = 5000
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp":        pd.date_range("2024-01-01", periods=T, freq="15min"),
        "segment_id":       rng.integers(0, 50, T).astype(str),
        "vehicle_count":    rng.integers(10, 500, T),
        "avg_speed":        rng.uniform(5, 80, T),
        "congestion_score": rng.uniform(0, 1, T),
        "temperature":      rng.uniform(22, 38, T),
        "rainfall":         rng.uniform(0, 30, T),
        "humidity":         rng.uniform(40, 100, T),
        "wind_speed":       rng.uniform(0, 20, T),
        "weather_condition": rng.choice(["Clear", "Cloudy", "Rain", "Fog"], T),
        "hour":             pd.date_range("2024-01-01", periods=T, freq="15min").hour,
        "day_of_week":      pd.date_range("2024-01-01", periods=T, freq="15min").dayofweek,
        "month":            pd.date_range("2024-01-01", periods=T, freq="15min").month,
        "is_peak_hour":     rng.integers(0, 2, T),
        "is_weekend":       rng.integers(0, 2, T),
    })
 
 
def _synthetic_graph_data(T: int, N: int, F: int):
    rng = np.random.default_rng(42)
    node_features = rng.random((T, N, F)).astype(np.float32)
    targets = rng.uniform(0, 1, (T, N)).astype(np.float32)
    edges = [(i, (i + 1) % N) for i in range(N)] + \
            [(i, (i + 2) % N) for i in range(0, N, 3)]
    from models.train_stgcn import build_adjacency_matrix
    adj = build_adjacency_matrix(edges, N)
    return node_features, targets, adj
 
 
# --- Entry Point -------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Traffic Model Trainer")
    parser.add_argument("--model", choices=["xgboost", "stgcn", "both"], default="both")
    parser.add_argument("--tune", action="store_true", help="Tune XGBoost hyperparams")
    parser.add_argument("--config", type=str, default=None, help="Path to STGCN config JSON")
    args = parser.parse_args()
 
    stgcn_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            stgcn_config = json.load(f)
 
    if args.model in ("xgboost", "both"):
        run_xgboost(tune=args.tune)
 
    if args.model in ("stgcn", "both"):
        run_stgcn(config=stgcn_config)
 
    logger.info("\nAll training complete. Check evaluation/ for metrics and plots.")
 
 
if __name__ == "__main__":
    main()