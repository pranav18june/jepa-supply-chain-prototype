# LEGACY: retained for historical baseline comparison. Active controlled path: run_controlled.py.
"""Real-World Public Dataset Benchmark using Dingdong FreshRetailNet-50K (arXiv:2505.16319).

Evaluates demand recovery, forecasting accuracy, and stockout risk prediction across
real retail store-SKU time series with leakage-resistant chronological splitting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class JEPADemandRecoveryModel(nn.Module):
    """JEPA-style sequence encoder with multi-head self-attention and demand recovery head."""

    def __init__(self, in_features: int = 8, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.05, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        # Multi-task readout: [predicted_demand, predicted_stockout_logit]
        self.readout = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: [batch, seq_len, in_features]"""
        B, T, _ = x.shape
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-4)
        x_norm = (x - mean) / std

        tokens = self.proj(x_norm) + self.pos_emb[:, :T, :]
        encoded = self.norm(self.encoder(tokens)).mean(dim=1)
        out = self.readout(encoded)
        return out[:, 0], out[:, 1]


class BaselineMLP(nn.Module):
    """Direct MLP without latent attention representation."""

    def __init__(self, in_dim: int = 5 * 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.net(x)
        return out[:, 0], out[:, 1]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, stockout_true: np.ndarray, stockout_prob: np.ndarray) -> Dict[str, float]:
    y_pred = np.clip(y_pred, 0.0, None)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1e-6))

    try:
        auc = float(roc_auc_score(stockout_true, stockout_prob))
        prauc = float(average_precision_score(stockout_true, stockout_prob))
    except Exception:
        auc, prauc = 0.5, 0.5

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "Stockout_ROC_AUC": auc,
        "Stockout_PR_AUC": prauc,
    }


def run_fresh_retail_benchmark(
    parquet_path: str = "prototype_new/data/fresh_retail_net/data/eval.parquet",
    sample_size: int = 15000,
    seed: int = 42,
    epochs: int = 12,
    device: str = "cpu",
) -> Dict[str, Any]:
    print("=== Running Real-World Public Dataset Benchmark (FreshRetailNet-50K) ===")
    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)

    # 1. Load public dataset
    print(f"Loading parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.select([
        'store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt',
        'discount', 'holiday_flag', 'activity_flag', 'precpt', 'avg_temperature', 'avg_humidity'
    ]).to_pandas()

    dates = sorted(df['dt'].unique())
    print(f"Dataset contains {len(df)} rows across {len(dates)} dates: {dates}")

    # Subsample store-product series for fast rigorous benchmarking
    unique_pairs = df[['store_id', 'product_id']].drop_duplicates()
    sampled_pairs = unique_pairs.sample(n=min(sample_size, len(unique_pairs)), random_state=seed)
    df = df.merge(sampled_pairs, on=['store_id', 'product_id'])
    print(f"Sampled {len(sampled_pairs)} store-product time series ({len(df)} records).")

    # Reshape into time series: Days 1-5 = Lookback history, Day 6 = Val, Day 7 = Chronological Test
    pivot_sales = df.pivot(index=['store_id', 'product_id'], columns='dt', values='sale_amount').fillna(0.0)
    pivot_stockout = df.pivot(index=['store_id', 'product_id'], columns='dt', values='stock_hour6_22_cnt').fillna(0.0)
    pivot_discount = df.pivot(index=['store_id', 'product_id'], columns='dt', values='discount').fillna(1.0)
    pivot_holiday = df.pivot(index=['store_id', 'product_id'], columns='dt', values='holiday_flag').fillna(0.0)
    pivot_activity = df.pivot(index=['store_id', 'product_id'], columns='dt', values='activity_flag').fillna(0.0)
    pivot_precpt = df.pivot(index=['store_id', 'product_id'], columns='dt', values='precpt').fillna(0.0)
    pivot_temp = df.pivot(index=['store_id', 'product_id'], columns='dt', values='avg_temperature').fillna(25.0)
    pivot_humidity = df.pivot(index=['store_id', 'product_id'], columns='dt', values='avg_humidity').fillna(50.0)

    # Chronological Split
    history_dates = dates[:5]   # Days 1 to 5
    val_date = dates[5]         # Day 6
    test_date = dates[6]        # Day 7 (Chronologically held out)

    N = len(pivot_sales)
    T = len(history_dates)

    # Feature matrix [N, T=5, F=8]
    X_tensor = np.zeros((N, T, 8), dtype=np.float32)
    for t_idx, d in enumerate(history_dates):
        X_tensor[:, t_idx, 0] = pivot_sales[d].values
        X_tensor[:, t_idx, 1] = pivot_stockout[d].values
        X_tensor[:, t_idx, 2] = pivot_discount[d].values
        X_tensor[:, t_idx, 3] = pivot_holiday[d].values
        X_tensor[:, t_idx, 4] = pivot_activity[d].values
        X_tensor[:, t_idx, 5] = pivot_precpt[d].values
        X_tensor[:, t_idx, 6] = pivot_temp[d].values / 30.0
        X_tensor[:, t_idx, 7] = pivot_humidity[d].values / 100.0

    y_test_demand = pivot_sales[test_date].values.astype(np.float32)
    y_test_stockout = (pivot_stockout[test_date].values > 0).astype(np.float32)

    y_val_demand = pivot_sales[val_date].values.astype(np.float32)
    y_val_stockout = (pivot_stockout[val_date].values > 0).astype(np.float32)

    # Model 1: Seasonal Naive (Predict test from previous day y_{t-1})
    print("\nEvaluating Model 1: Seasonal Naive...")
    naive_pred = pivot_sales[dates[4]].values
    naive_prob = (pivot_stockout[dates[4]].values > 0).astype(np.float32)
    m1_metrics = compute_metrics(y_test_demand, naive_pred, y_test_stockout, naive_prob)
    print(f"  Naive WAPE: {m1_metrics['WAPE']:.4f}, MAE: {m1_metrics['MAE']:.3f}, AUC: {m1_metrics['Stockout_ROC_AUC']:.3f}")

    # Model 2: Strong Conventional Tabular Model (ExtraTreesRegressor)
    print("Evaluating Model 2: ExtraTrees Regressor...")
    X_flat = X_tensor.reshape(N, -1)
    rf = ExtraTreesRegressor(n_estimators=50, random_state=seed, min_samples_leaf=3, n_jobs=-1)
    rf.fit(X_flat, y_val_demand)
    rf_pred = rf.predict(X_flat)
    rf_prob = np.clip(rf_pred / max(np.mean(y_val_demand), 1e-4), 0.0, 1.0)
    m2_metrics = compute_metrics(y_test_demand, rf_pred, y_test_stockout, rf_prob)
    print(f"  ExtraTrees WAPE: {m2_metrics['WAPE']:.4f}, MAE: {m2_metrics['MAE']:.3f}, AUC: {m2_metrics['Stockout_ROC_AUC']:.3f}")

    # Convert to PyTorch tensors
    X_pt = torch.tensor(X_tensor, dtype=torch.float32, device=dev)
    X_flat_pt = torch.tensor(X_flat, dtype=torch.float32, device=dev)
    y_val_pt = torch.tensor(y_val_demand, dtype=torch.float32, device=dev)
    y_stockout_val_pt = torch.tensor(y_val_stockout, dtype=torch.float32, device=dev)

    # Model 3: Baseline MLP (Direct sequence flattening)
    print("Training Model 3: Baseline MLP...")
    mlp = BaselineMLP(in_dim=T * 8).to(dev)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 128
    for ep in range(epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            pred_d, pred_s = mlp(X_flat_pt[idx])
            loss = F.smooth_l1_loss(pred_d, y_val_pt[idx]) + 0.5 * F.binary_cross_entropy_with_logits(pred_s, y_stockout_val_pt[idx])
            opt_mlp.zero_grad()
            loss.backward()
            opt_mlp.step()

    mlp.eval()
    with torch.no_grad():
        mlp_pred_pt, mlp_prob_pt = mlp(X_flat_pt)
        mlp_pred = mlp_pred_pt.cpu().numpy()
        mlp_prob = torch.sigmoid(mlp_prob_pt).cpu().numpy()
    m3_metrics = compute_metrics(y_test_demand, mlp_pred, y_test_stockout, mlp_prob)
    print(f"  MLP WAPE: {m3_metrics['WAPE']:.4f}, MAE: {m3_metrics['MAE']:.3f}, AUC: {m3_metrics['Stockout_ROC_AUC']:.3f}")

    # Model 4: JEPA Demand Recovery Model (Self-attention latent representation)
    print("Training Model 4: JEPA Demand Recovery World Model...")
    jepa_model = JEPADemandRecoveryModel(in_features=8, d_model=64).to(dev)
    opt_jepa = torch.optim.AdamW(jepa_model.parameters(), lr=8e-4, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            pred_d, pred_s = jepa_model(X_pt[idx])
            loss = F.smooth_l1_loss(pred_d, y_val_pt[idx]) + 0.5 * F.binary_cross_entropy_with_logits(pred_s, y_stockout_val_pt[idx])
            opt_jepa.zero_grad()
            loss.backward()
            opt_jepa.step()

    jepa_model.eval()
    with torch.no_grad():
        jepa_pred_pt, jepa_prob_pt = jepa_model(X_pt)
        jepa_pred = jepa_pred_pt.cpu().numpy()
        jepa_prob = torch.sigmoid(jepa_prob_pt).cpu().numpy()
    m4_metrics = compute_metrics(y_test_demand, jepa_pred, y_test_stockout, jepa_prob)
    print(f"  JEPA WAPE: {m4_metrics['WAPE']:.4f}, MAE: {m4_metrics['MAE']:.3f}, AUC: {m4_metrics['Stockout_ROC_AUC']:.3f}")

    elapsed = time.time() - start_time
    results = {
        "dataset": "Dingdong-Inc/FreshRetailNet-50K",
        "provenance_url": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K",
        "sample_size": N,
        "history_window_days": T,
        "test_date": test_date,
        "elapsed_seconds": elapsed,
        "models": {
            "seasonal_naive": m1_metrics,
            "extratrees_tabular": m2_metrics,
            "baseline_mlp": m3_metrics,
            "jepa_representation": m4_metrics,
        }
    }

    # Save results
    out_dir = Path("prototype_new/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "real_data_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparisons
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        model_names = ["Seasonal Naive", "ExtraTrees", "Baseline MLP", "JEPA Representation"]
        wapes = [results["models"][k]["WAPE"] for k in ["seasonal_naive", "extratrees_tabular", "baseline_mlp", "jepa_representation"]]
        aucs = [results["models"][k]["Stockout_ROC_AUC"] for k in ["seasonal_naive", "extratrees_tabular", "baseline_mlp", "jepa_representation"]]

        colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
        axes[0].bar(model_names, wapes, color=colors)
        axes[0].set_ylabel("WAPE (lower is better)")
        axes[0].set_title("Demand Forecasting WAPE on FreshRetailNet-50K")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        axes[1].bar(model_names, aucs, color=colors)
        axes[1].set_ylabel("ROC-AUC (higher is better)")
        axes[1].set_title("Stockout Risk Classification AUC")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_dir / "real_data_predictions.png", dpi=200)
        plt.close(fig)
        print(f"Plot saved to: {out_dir / 'real_data_predictions.png'}")

    print(f"\n[SUCCESS] Real-world benchmark completed in {elapsed:.1f}s.")
    return results


if __name__ == "__main__":
    run_fresh_retail_benchmark()
