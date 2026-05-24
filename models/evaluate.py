"""
Model Evaluation utilities for the Traffic Prediction project.
Computes regression metrics and congestion-level classification metrics.
Generates plots: loss curve, prediction vs actual, feature importance.
"""
 
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
 
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
)
 
logger = logging.getLogger(__name__)
 
# Congestion levels (aligned with project spec)
CONGESTION_BINS   = [0.0, 0.3, 0.6, 0.8, 1.01]
CONGESTION_LABELS = ["Free Flow", "Moderate", "Heavy", "Severe"]
 
 
def congestion_level(score: float) -> str:
    for i, (lo, hi) in enumerate(zip(CONGESTION_BINS, CONGESTION_BINS[1:])):
        if lo <= score < hi:
            return CONGESTION_LABELS[i]
    return CONGESTION_LABELS[-1]
 
 
def scores_to_levels(scores: np.ndarray) -> np.ndarray:
    return np.array([congestion_level(s) for s in scores])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Core Metrics
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
 
 
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Convert regression outputs to congestion levels, then classify."""
    true_levels = scores_to_levels(y_true)
    pred_levels = scores_to_levels(y_pred)
    report = classification_report(true_levels, pred_levels, output_dict=True, zero_division=0)
    return report
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────
 
class ModelEvaluator:
    """
    Evaluates and visualizes model performance.
    Works for both STGCN-LSTM and XGBoost outputs.
    
    Usage:
        evaluator = ModelEvaluator(output_dir="evaluation")
        evaluator.evaluate(y_true, y_pred, model_name="xgboost")
    """
 
    def __init__(self, output_dir: str = "evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
 
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        loss_history: Optional[Dict] = None,
        feature_importances: Optional[List] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Full evaluation pipeline.
        Returns dict with all metrics.
        """
        y_true = np.clip(y_true.flatten(), 0, 1)
        y_pred = np.clip(y_pred.flatten(), 0, 1)
 
        reg_metrics  = compute_regression_metrics(y_true, y_pred)
        clf_report   = compute_classification_metrics(y_true, y_pred)
        all_metrics  = {**reg_metrics, "classification": clf_report}
 
        logger.info(f"\n{'='*50}")
        logger.info(f"  {model_name.upper()} Evaluation Results")
        logger.info(f"{'='*50}")
        for k, v in reg_metrics.items():
            logger.info(f"  {k:8s}: {v:.4f}")
 
        # Save metrics JSON
        metrics_path = os.path.join(self.output_dir, f"{model_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
 
        # Plots
        self._plot_prediction_vs_actual(y_true, y_pred, model_name, timestamps)
        self._plot_error_distribution(y_true, y_pred, model_name)
        if loss_history:
            self._plot_loss_curve(loss_history, model_name)
        if feature_importances:
            self._plot_feature_importance(feature_importances, model_name)
        self._plot_confusion_matrix(y_true, y_pred, model_name)
 
        logger.info(f"Evaluation artifacts saved to: {self.output_dir}/")
        return all_metrics
 
    # ── Individual plots ───────────────────────────────────────────────────
 
    def _plot_prediction_vs_actual(self, y_true, y_pred, name, timestamps):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        x = timestamps if timestamps is not None else np.arange(len(y_true))
 
        axes[0].plot(x, y_true, label="Actual",    color="#2196F3", linewidth=1)
        axes[0].plot(x, y_pred, label="Predicted", color="#FF5722", linewidth=1, alpha=0.8)
        axes[0].set_ylabel("Congestion Score")
        axes[0].set_title(f"{name} — Predicted vs Actual Congestion")
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
 
        errors = y_pred - y_true
        axes[1].fill_between(x, errors, 0, where=(errors >= 0),
                             color="#FF5722", alpha=0.4, label="Over-predict")
        axes[1].fill_between(x, errors, 0, where=(errors < 0),
                             color="#2196F3", alpha=0.4, label="Under-predict")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Residual")
        axes[1].set_xlabel("Time Step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_pred_vs_actual.png"), dpi=120)
        plt.close()
 
    def _plot_error_distribution(self, y_true, y_pred, name):
        errors = y_pred - y_true
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
        axes[0].hist(errors, bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Error"); axes[0].set_ylabel("Count")
 
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=10, color="#009688")
        axes[1].plot([0, 1], [0, 1], "r--", linewidth=1.5)
        axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
        axes[1].set_title("Actual vs Predicted (scatter)")
        axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
 
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_error_dist.png"), dpi=120)
        plt.close()
 
    def _plot_loss_curve(self, history: Dict, name: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.get("train_loss", []), label="Train Loss", color="#2196F3")
        ax.plot(history.get("val_loss",   []), label="Val Loss",   color="#FF5722")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"{name} — Training Curve")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_loss_curve.png"), dpi=120)
        plt.close()
 
    def _plot_feature_importance(self, importances: List[tuple], name: str):
        labels = [f[0] for f in importances[:15]]
        values = [f[1] for f in importances[:15]]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels[::-1], values[::-1], color="#4CAF50", alpha=0.8)
        ax.set_xlabel("Importance"); ax.set_title(f"{name} — Top Feature Importances")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_feature_importance.png"), dpi=120)
        plt.close()
 
    def _plot_confusion_matrix(self, y_true, y_pred, name: str):
        true_lv = scores_to_levels(y_true)
        pred_lv = scores_to_levels(y_pred)
        labels  = CONGESTION_LABELS
        cm      = confusion_matrix(true_lv, pred_lv, labels=labels)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
 
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Level"); ax.set_ylabel("Actual Level")
        ax.set_title(f"{name} — Congestion Level Confusion Matrix")
        plt.colorbar(im, ax=ax, label="Proportion")
 
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=10)
 
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_confusion_matrix.png"), dpi=120)
        plt.close()
 