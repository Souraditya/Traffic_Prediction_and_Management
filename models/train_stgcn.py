"""
Training pipeline for the Spatio-Temporal GCN + LSTM model.
Handles dataset preparation, training loop, validation, and checkpointing.
"""
 
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict, Optional
import json
 
from models.stgcn_lstm import TrafficSTGCN_LSTM
 
logger = logging.getLogger(__name__)
 
 
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
 
class TrafficGraphDataset(Dataset):
    """
    Sliding-window dataset over spatio-temporal traffic data.
 
    Args:
        node_features : (T, N, F) numpy array  - features per timestep per node
        targets       : (T, N)   numpy array  - congestion scores
        adj_matrix    : (N, N)   numpy array  - normalized adjacency
        seq_len       : int      - number of past timesteps used as input
    """
 
    def __init__(
        self,
        node_features: np.ndarray,
        targets: np.ndarray,
        adj_matrix: np.ndarray,
        seq_len: int = 12,
    ):
        self.X   = torch.FloatTensor(node_features)  # (T, N, F)
        self.y   = torch.FloatTensor(targets)         # (T, N)
        self.adj = torch.FloatTensor(adj_matrix)      # (N, N)
        self.seq_len = seq_len
        self.n_samples = len(self.X) - seq_len
 
    def __len__(self):
        return self.n_samples
 
    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.seq_len]     # (seq_len, N, F)
        y_target = self.y[idx + self.seq_len]            # (N,)
        return x_window, y_target, self.adj
 
 
# -----------------------------------------------------------------------------
# Adjacency Matrix Builder
# -----------------------------------------------------------------------------
 
def build_adjacency_matrix(
    edges: list,
    num_nodes: int,
    add_self_loops: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        if len(edge) == 3:
            i, j, w = edge
        else:
            i, j = edge; w = 1.0
        adj[i, j] += w
        adj[j, i] += w
 
    if add_self_loops:
        adj += np.eye(num_nodes, dtype=np.float32)
 
    if normalize:
        d = adj.sum(axis=1, keepdims=True)
        d = np.where(d == 0, 1, d)
        adj = adj / d
 
    return adj
 
 
# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
 
class STGCNTrainer:
    def __init__(self, config: Dict):
        # Fix random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
 
        self.model = TrafficSTGCN_LSTM(
            node_features = config["node_features"],
            gcn_hidden    = config.get("gcn_hidden", 64),
            gcn_out       = config.get("gcn_out", 32),
            lstm_hidden   = config.get("lstm_hidden", 128),
            lstm_layers   = config.get("lstm_layers", 2),
            num_nodes     = config["num_nodes"],
            dropout       = config.get("dropout", 0.3),
        ).to(self.device)
 
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.criterion = nn.MSELoss()
 
        self.save_dir = config.get("save_dir", "models/saved")
        os.makedirs(self.save_dir, exist_ok=True)
 
        self.history = {"train_loss": [], "val_loss": [], "val_mae": []}
        self.best_val_loss = float("inf")
 
    def fit(self, train_dataset, val_dataset):
        cfg = self.cfg
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=0,
        )
 
        epochs   = cfg.get("epochs", 50)
        patience = cfg.get("early_stop_patience", 10)
        no_improve = 0
 
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_mae = self._val_epoch(val_loader)
            self.scheduler.step(val_loss)
 
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
 
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )
 
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break
 
        self._save_history()
        logger.info("Training complete. Best val loss: {:.4f}".format(self.best_val_loss))
 
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y, adj in loader:
            x   = x.to(self.device)
            y   = y.to(self.device)
            adj = adj[0].to(self.device)
 
            self.optimizer.zero_grad()
            pred, _ = self.model(x, adj)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(x)
 
        return total_loss / len(loader.dataset)
 
    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae  = 0.0
        for x, y, adj in loader:
            x   = x.to(self.device)
            y   = y.to(self.device)
            adj = adj[0].to(self.device)
 
            pred, _ = self.model(x, adj)
            loss = self.criterion(pred, y)
            mae  = torch.mean(torch.abs(pred - y))
            total_loss += loss.item() * len(x)
            total_mae  += mae.item()  * len(x)
 
        n = len(loader.dataset)
        return total_loss / n, total_mae / n
 
    def save_checkpoint(self, tag: str = "latest"):
        path = os.path.join(self.save_dir, f"stgcn_{tag}.pt")
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss":   self.best_val_loss,
            "config":          self.cfg,
        }, path)
        logger.info(f"Checkpoint saved -> {path}")
 
    def load_checkpoint(self, tag: str = "best"):
        path = os.path.join(self.save_dir, f"stgcn_{tag}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        logger.info(f"Checkpoint loaded <- {path}")
 
    def _save_history(self):
        path = os.path.join(self.save_dir, "stgcn_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
