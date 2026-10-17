"""Uncertainty estimation and fallback gating for decision reliability under disruption shift."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter


def compute_normalized_score_spread(member_scores: np.ndarray) -> float:
    """Computes cross-member spread of z-score normalized candidate scores.
    
    Args:
        member_scores: [num_members, num_candidates] array of candidate scores.
    Returns:
        float: mean across candidates of cross-member standard deviations.
    """
    if member_scores.shape[0] <= 1:
        return 0.0
    # Normalize per member across candidates
    means = member_scores.mean(axis=1, keepdims=True)
    stds = member_scores.std(axis=1, keepdims=True) + 1e-6
    z_scores = (member_scores - means) / stds
    # Cross-member std per candidate, averaged across candidates
    return float(np.mean(np.std(z_scores, axis=0, ddof=1 if member_scores.shape[0] > 1 else 0)))


def compute_top_action_disagreement(member_top_actions: List[str]) -> float:
    """Computes fraction of ensemble members that do not select the modal top action.
    
    Returns:
        float: 1.0 - (count of most common top action / num_members)
    """
    if not member_top_actions:
        return 0.0
    counts = Counter(member_top_actions)
    most_common_count = counts.most_common(1)[0][1]
    return float(1.0 - (most_common_count / len(member_top_actions)))


def compute_pairwise_ranking_disagreement(member_scores: np.ndarray) -> float:
    """Computes the fraction of candidate pairs where ensemble members disagree on rank order.
    
    Args:
        member_scores: [num_members, num_candidates] array
    Returns:
        float: fraction of candidate pairs with conflicting orderings among members
    """
    num_members, num_candidates = member_scores.shape
    if num_members <= 1 or num_candidates <= 1:
        return 0.0

    total_pairs = 0
    disagreed_pairs = 0

    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            # Direction for each member: True if member thinks i is cheaper than j
            diffs = member_scores[:, i] - member_scores[:, j]
            # Ignore exact ties within epsilon
            valid = np.abs(diffs) > 1e-4
            if not np.any(valid):
                continue
            signs = np.sign(diffs[valid])
            total_pairs += 1
            # If not all signs are identical, members disagree on this pair
            if not np.all(signs == signs[0]):
                disagreed_pairs += 1

    return float(disagreed_pairs / max(total_pairs, 1))


def compute_best_vs_second_best_margin(aggregate_scores: Dict[str, float]) -> float:
    """Computes score margin between second-best and best plan.
    
    In cost minimization, best is min score, second best is second lowest.
    Margin = score(second_best) - score(best) >= 0.
    A small margin indicates high decision ambiguity.
    """
    if len(aggregate_scores) < 2:
        return 0.0
    sorted_scores = sorted(aggregate_scores.values())
    return float(sorted_scores[1] - sorted_scores[0])


def compute_ensemble_uncertainties(
    member_predictions: List[Dict[str, float]],
    candidate_names: List[str],
) -> Dict[str, float]:
    """Computes the complete suite of uncertainty diagnostics from an ensemble of predictions.
    
    Args:
        member_predictions: List of dicts, one per member, mapping candidate_name -> score
        candidate_names: List of all candidate plan names
    Returns:
        Dict[str, float] with keys:
            - 'normalized_score_spread'
            - 'top_action_disagreement'
            - 'pairwise_rank_disagreement'
            - 'score_margin'
            - 'raw_score_std'
    """
    num_members = len(member_predictions)
    if num_members == 0:
        return {
            "normalized_score_spread": 0.0,
            "top_action_disagreement": 0.0,
            "pairwise_rank_disagreement": 0.0,
            "score_margin": 0.0,
            "raw_score_std": 0.0,
        }

    # Matrix: [num_members, num_candidates]
    matrix = np.array([
        [pred[name] for name in candidate_names]
        for pred in member_predictions
    ], dtype=np.float64)

    member_top_actions = [
        candidate_names[int(np.argmin(matrix[m]))]
        for m in range(num_members)
    ]

    mean_scores = {name: float(matrix[:, idx].mean()) for idx, name in enumerate(candidate_names)}

    return {
        "normalized_score_spread": compute_normalized_score_spread(matrix),
        "top_action_disagreement": compute_top_action_disagreement(member_top_actions),
        "pairwise_rank_disagreement": compute_pairwise_ranking_disagreement(matrix),
        "score_margin": compute_best_vs_second_best_margin(mean_scores),
        "raw_score_std": float(np.mean(np.std(matrix, axis=0))),
    }


class EnsembleUncertaintyGate:
    """Threshold gate triggering fallback to a robust conventional policy when uncertainty is high.
    
    Guarantees:
    - Never inspects oracle values, future outcomes, or regime labels.
    - Supports both raw Tensor epistemic dispersion and structured uncertainty dictionaries.
    """

    def __init__(
        self,
        threshold: float = 12.0,
        metric_key: str = "top_action_disagreement",
        reverse_direction: bool = False,
        uncertainty_threshold: Optional[float] = None,
    ):
        self.threshold = uncertainty_threshold if uncertainty_threshold is not None else threshold
        self.uncertainty_threshold = self.threshold
        self.metric_key = metric_key
        self.reverse_direction = reverse_direction

    def compute_uncertainty(self, predictions: List[Tensor]) -> Tensor:
        """Computes standard deviation across model predictions."""
        stacked = torch.stack(predictions, dim=0)
        return stacked.std(dim=0)

    def should_fallback(self, uncertainty: float | Dict[str, float] | Tensor) -> bool:
        if isinstance(uncertainty, dict):
            score = float(uncertainty.get(self.metric_key, 0.0))
        elif isinstance(uncertainty, Tensor):
            score = float(uncertainty.item())
        else:
            score = float(uncertainty)

        if self.reverse_direction:
            return score < self.threshold
        return score >= self.threshold
