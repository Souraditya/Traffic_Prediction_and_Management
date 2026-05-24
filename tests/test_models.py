"""
Unit and integration tests for model training components.
Run: pytest tests/test_models.py -v
"""
 
import pytest
import numpy as np
import pandas as pd
import torch
import sys, os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 
from models.stgcn_lstm import TrafficSTGCN_LSTM, GraphConvolution, STGCNBlock
from models.train_stgcn import (
    TrafficGraphDataset, build_adjacency_matrix, STGCNTrainer
)
from models.xgboost_model import build_features, XGBoostTrafficModel
from models.evaluate import (
    compute_regression_metrics,
    compute_classification_metrics,
    scores_to_levels,
    ModelEvaluator,
)
 
 
# ─── Fixtures ─────────────────────────────────────────────────────────────────
 
N, T, F = 10, 100, 6
 
@pytest.fixture
def simple_adj():
    edges = [(i, (i+1) % N) for i in range(N)]
    return build_adjacency_matrix(edges, N)
 
@pytest.fixture
def graph_data(simple_adj):
    rng = np.random.default_rng(0)
    node_feat = rng.random((T, N, F)).astype(np.float32)
    targets   = rng.uniform(0, 1, (T, N)).astype(np.float32)
    return node_feat, targets, simple_adj
 
@pytest.fixture
def tabular_df():
    rng = np.random.default_rng(0)
    size = 300
    return pd.DataFrame({
        "timestamp":        pd.date_range("2024-01-01", periods=size, freq="15min"),
        "segment_id":       [str(i % 10) for i in range(size)],
        "vehicle_count":    rng.integers(10, 500, size),
        "avg_speed":        rng.uniform(5, 80, size),
        "congestion_score": rng.uniform(0, 1, size),
        "temperature":      rng.uniform(22, 38, size),
        "rainfall":         rng.uniform(0, 30, size),
        "humidity":         rng.uniform(40, 100, size),
        "wind_speed":       rng.uniform(0, 20, size),
        "weather_condition": rng.choice(["Clear", "Rain", "Fog"], size),
        "hour":             pd.date_range("2024-01-01", periods=size, freq="15min").hour,
        "day_of_week":      pd.date_range("2024-01-01", periods=size, freq="15min").dayofweek,
        "month":            pd.date_range("2024-01-01", periods=size, freq="15min").month,
        "is_peak_hour":     rng.integers(0, 2, size),
        "is_weekend":       rng.integers(0, 2, size),
    })
 
 
# ─── GCN Layer Tests ──────────────────────────────────────────────────────────
 
def test_graph_convolution_shape(simple_adj):
    gc  = GraphConvolution(in_features=8, out_features=16)
    adj = torch.FloatTensor(simple_adj)
    x   = torch.randn(4, N, 8)
    out = gc(x, adj)
    assert out.shape == (4, N, 16)
 
def test_stgcn_block_shape(simple_adj):
    block = STGCNBlock(in_features=6, hidden=16, out_features=8)
    adj   = torch.FloatTensor(simple_adj)
    x     = torch.randn(4, N, 6)
    out   = block(x, adj)
    assert out.shape == (4, N, 8)
 
 
# ─── Full Model Tests ─────────────────────────────────────────────────────────
 
def test_stgcn_lstm_forward(simple_adj):
    model = TrafficSTGCN_LSTM(node_features=F, gcn_hidden=16, gcn_out=8,
                               lstm_hidden=32, lstm_layers=1, num_nodes=N, dropout=0.0)
    adj = torch.FloatTensor(simple_adj)
    x   = torch.randn(2, 12, N, F)
    out, _ = model(x, adj)
    assert out.shape == (2, N)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0,1]"
 
def test_stgcn_lstm_gradient_flows(simple_adj):
    model = TrafficSTGCN_LSTM(node_features=F, gcn_hidden=16, gcn_out=8,
                               lstm_hidden=32, lstm_layers=1, num_nodes=N, dropout=0.0)
    adj = torch.FloatTensor(simple_adj)
    x   = torch.randn(2, 12, N, F, requires_grad=False)
    out, _ = model(x, adj)
    loss = out.mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
 
 
# ─── Dataset Tests ────────────────────────────────────────────────────────────
 
def test_traffic_dataset_len(graph_data):
    nf, tg, adj = graph_data
    ds = TrafficGraphDataset(nf, tg, adj, seq_len=12)
    assert len(ds) == T - 12
 
def test_traffic_dataset_shapes(graph_data):
    nf, tg, adj = graph_data
    ds = TrafficGraphDataset(nf, tg, adj, seq_len=12)
    x, y, a = ds[0]
    assert x.shape == (12, N, F)
    assert y.shape == (N,)
    assert a.shape == (N, N)
 
 
# ─── Adjacency Matrix Tests ───────────────────────────────────────────────────
 
def test_adj_shape():
    edges = [(0, 1), (1, 2), (2, 0)]
    adj = build_adjacency_matrix(edges, 5)
    assert adj.shape == (5, 5)
 
def test_adj_row_sum():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    adj = build_adjacency_matrix(edges, 5, normalize=True)
    row_sums = adj.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-5)
 
 
# ─── XGBoost Tests ────────────────────────────────────────────────────────────
 
def test_build_features_columns(tabular_df):
    df_fe = build_features(tabular_df)
    assert "hour_sin" in df_fe.columns
    assert "congestion_lag_1" in df_fe.columns
    assert "rain_x_vehicles" in df_fe.columns
 
def test_xgboost_train_predict(tabular_df, tmp_path):
    model = XGBoostTrafficModel(model_dir=str(tmp_path))
    metrics = model.train(tabular_df)
    assert "val_rmse" in metrics
    assert metrics["val_rmse"] >= 0
    preds = model.predict(tabular_df.head(50))
    assert len(preds) > 0
 
def test_xgboost_save_load(tabular_df, tmp_path):
    model = XGBoostTrafficModel(model_dir=str(tmp_path))
    model.train(tabular_df)
    model.save("test_model")
    model2 = XGBoostTrafficModel(model_dir=str(tmp_path))
    model2.load("test_model")
    p1 = model.predict(tabular_df.head(10))
    p2 = model2.predict(tabular_df.head(10))
    np.testing.assert_allclose(p1, p2, rtol=1e-4)
 
 
# ─── Evaluation Tests ─────────────────────────────────────────────────────────
 
def test_regression_metrics():
    y_true = np.array([0.2, 0.5, 0.8, 0.3])
    y_pred = np.array([0.25, 0.45, 0.75, 0.35])
    m = compute_regression_metrics(y_true, y_pred)
    assert "RMSE" in m and "MAE" in m and "R2" in m
    assert m["RMSE"] >= 0
 
def test_scores_to_levels():
    scores = np.array([0.1, 0.4, 0.7, 0.9])
    levels = scores_to_levels(scores)
    assert list(levels) == ["Free Flow", "Moderate", "Heavy", "Severe"]
 
def test_evaluator_runs(tmp_path):
    evaluator = ModelEvaluator(output_dir=str(tmp_path))
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 1, 200)
    y_pred = y_true + rng.normal(0, 0.1, 200)
    y_pred = np.clip(y_pred, 0, 1)
    metrics = evaluator.evaluate(y_true, y_pred, model_name="test")
    assert "RMSE" in metrics
    assert (tmp_path / "test_metrics.json").exists()
    assert (tmp_path / "test_confusion_matrix.png").exists()
