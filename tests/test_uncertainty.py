"""Unit tests for uncertainty metrics and score normalization."""

import numpy as np
import pytest
from prototype_new.src.models.uncertainty import (
    compute_normalized_score_spread,
    compute_top_action_disagreement,
    compute_pairwise_ranking_disagreement,
    compute_best_vs_second_best_margin,
    compute_ensemble_uncertainties,
    EnsembleUncertaintyGate,
)


def test_top_action_disagreement():
    # Complete agreement
    assert compute_top_action_disagreement(["regular", "regular", "regular"]) == 0.0
    # 2 agree out of 3 -> modal fraction = 2/3 -> disagreement = 1/3
    assert abs(compute_top_action_disagreement(["regular", "regular", "split"]) - (1.0 / 3.0)) < 1e-4
    # All 3 disagree
    assert abs(compute_top_action_disagreement(["regular", "split", "expedited"]) - (2.0 / 3.0)) < 1e-4


def test_pairwise_ranking_disagreement():
    # 2 members, candidate 0 cheaper than 1 for both
    scores = np.array([
        [10.0, 20.0],
        [15.0, 25.0],
    ])
    assert compute_pairwise_ranking_disagreement(scores) == 0.0

    # Inverted ordering
    scores_inverted = np.array([
        [10.0, 20.0],
        [25.0, 15.0],
    ])
    assert compute_pairwise_ranking_disagreement(scores_inverted) == 1.0


def test_best_vs_second_best_margin():
    scores = {"regular": 10.0, "split": 15.0, "expedited": 25.0}
    assert compute_best_vs_second_best_margin(scores) == 5.0


def test_ensemble_uncertainty_gate():
    gate = EnsembleUncertaintyGate(threshold=0.5, metric_key="top_action_disagreement")
    assert not gate.should_fallback({"top_action_disagreement": 0.3})
    assert gate.should_fallback({"top_action_disagreement": 0.6})
    assert gate.should_fallback({"top_action_disagreement": 0.5})

    # Reverse direction for margins: trigger when margin < threshold
    margin_gate = EnsembleUncertaintyGate(threshold=2.0, metric_key="score_margin", reverse_direction=True)
    assert margin_gate.should_fallback({"score_margin": 1.0})
    assert not margin_gate.should_fallback({"score_margin": 3.0})
