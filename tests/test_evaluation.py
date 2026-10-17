"""Unit tests for metrics and end-to-end evaluation."""

import pytest
import numpy as np
from prototype_new.src.evaluation.metrics import compute_metrics


def test_compute_metrics():
    # 5 worlds, 3 candidate plans
    # Ground truth costs
    realized_costs = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 15.0, 25.0],
        [30.0, 20.0, 10.0],
        [12.0, 18.0, 24.0],
        [50.0, 60.0, 70.0],
    ])

    # Perfect predictions (oracle)
    perfect_preds = np.copy(realized_costs)
    fill_rates = [0.95, 0.90, 0.98, 0.92, 0.85]
    stockouts = [0, 1, 0, 0, 2]

    metrics = compute_metrics(realized_costs, perfect_preds, fill_rates, stockouts)
    assert metrics.mean_regret == 0.0
    assert metrics.top1_accuracy == 1.0
    assert metrics.pairwise_ranking_accuracy == 1.0
    assert metrics.catastrophic_inversion_rate == 0.0

    # Inverted predictions
    inverted_preds = -realized_costs
    m_inv = compute_metrics(realized_costs, inverted_preds, fill_rates, stockouts)
    assert m_inv.mean_regret > 0.0
    assert m_inv.top1_accuracy == 0.0
    assert m_inv.pairwise_ranking_accuracy == 0.0
