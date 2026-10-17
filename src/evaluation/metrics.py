"""Evaluation metrics and paired statistical comparison utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple, Optional, Any
import numpy as np
from scipy import stats


@dataclass
class EvaluationMetrics:
    mean_regret: float
    median_regret: float
    regret_ci_95: Tuple[float, float]
    mean_cost: float
    across_world_cvar_95: float
    stockout_rate: float
    mean_fill_rate: float
    top1_accuracy: float
    pairwise_ranking_accuracy: float
    kendall_tau: float
    catastrophic_inversion_rate: float
    fallback_rate: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_regret": self.mean_regret,
            "median_regret": self.median_regret,
            "ci95_low": self.regret_ci_95[0],
            "ci95_high": self.regret_ci_95[1],
            "mean_cost": self.mean_cost,
            "across_world_cvar_95": self.across_world_cvar_95,
            "tail_cvar_95": self.across_world_cvar_95,  # backward compatibility
            "stockout_rate": self.stockout_rate,
            "mean_fill_rate": self.mean_fill_rate,
            "top1_accuracy": self.top1_accuracy,
            "pairwise_ranking_accuracy": self.pairwise_ranking_accuracy,
            "kendall_tau": self.kendall_tau,
            "catastrophic_inversion_rate": self.catastrophic_inversion_rate,
            "fallback_rate": self.fallback_rate,
        }


def compute_metrics(
    realized_costs: np.ndarray,      # [N, num_candidates]
    predicted_scores: np.ndarray,    # [N, num_candidates] (lower score = preferred)
    fill_rates: Sequence[float],     # [N]
    stockout_periods: Sequence[int], # [N]
    fallback_flags: Optional[Sequence[bool]] = None,
    catastrophic_threshold_ratio: float = 1.75,
    seed: int = 42,
) -> EvaluationMetrics:
    """Compute comprehensive operational, decision regret, and ranking metrics with explicit RNG."""
    rng = np.random.default_rng(seed)
    n_worlds, n_candidates = realized_costs.shape

    selected_indices = np.argmin(predicted_scores, axis=1)
    oracle_indices = np.argmin(realized_costs, axis=1)

    chosen_costs = realized_costs[np.arange(n_worlds), selected_indices]
    oracle_costs = realized_costs[np.arange(n_worlds), oracle_indices]

    regrets = chosen_costs - oracle_costs
    mean_regret = float(np.mean(regrets))
    median_regret = float(np.median(regrets))

    # Bootstrap 95% Confidence Interval for mean regret using explicit RNG
    if n_worlds >= 10:
        bootstrap_means = [
            float(np.mean(rng.choice(regrets, size=n_worlds, replace=True)))
            for _ in range(500)
        ]
        ci_low = float(np.percentile(bootstrap_means, 2.5))
        ci_high = float(np.percentile(bootstrap_means, 97.5))
    else:
        ci_low = mean_regret
        ci_high = mean_regret

    mean_cost = float(np.mean(chosen_costs))

    # Across-world tail cost: 95% CVaR of total cost across worlds
    cvar_cutoff = float(np.percentile(chosen_costs, 95))
    tail_subset = chosen_costs[chosen_costs >= cvar_cutoff]
    across_world_cvar_95 = float(np.mean(tail_subset)) if len(tail_subset) > 0 else cvar_cutoff

    top1_accuracy = float(np.mean(selected_indices == oracle_indices))

    # Catastrophic inversion: chosen plan significantly exceeds oracle by multiplier + buffer
    catastrophic_inversions = chosen_costs >= (oracle_costs * catastrophic_threshold_ratio + 5.0)
    catastrophic_rate = float(np.mean(catastrophic_inversions))

    pairwise_correct = []
    kendall_taus = []
    for i in range(n_worlds):
        rc = realized_costs[i]
        ps = predicted_scores[i]
        for a in range(n_candidates):
            for b in range(a + 1, n_candidates):
                if abs(rc[a] - rc[b]) > 0.1:
                    true_pref = rc[a] < rc[b]
                    pred_pref = ps[a] < ps[b]
                    pairwise_correct.append(1.0 if true_pref == pred_pref else 0.0)

        tau, _ = stats.kendalltau(rc, ps)
        if not np.isnan(tau):
            kendall_taus.append(tau)

    pairwise_acc = float(np.mean(pairwise_correct)) if pairwise_correct else 0.5
    mean_tau = float(np.mean(kendall_taus)) if kendall_taus else 0.0

    mean_fill = float(np.mean(fill_rates))
    stockout_rate = float(np.mean([1.0 if p > 0 else 0.0 for p in stockout_periods]))
    fallback_rate = float(np.mean(fallback_flags)) if fallback_flags is not None and len(fallback_flags) > 0 else 0.0

    return EvaluationMetrics(
        mean_regret=mean_regret,
        median_regret=median_regret,
        regret_ci_95=(ci_low, ci_high),
        mean_cost=mean_cost,
        across_world_cvar_95=across_world_cvar_95,
        stockout_rate=stockout_rate,
        mean_fill_rate=mean_fill,
        top1_accuracy=top1_accuracy,
        pairwise_ranking_accuracy=pairwise_acc,
        kendall_tau=mean_tau,
        catastrophic_inversion_rate=catastrophic_rate,
        fallback_rate=fallback_rate,
    )


def compute_paired_comparison(
    regrets_a: np.ndarray,
    regrets_b: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Computes paired difference statistics between two methods on identical worlds.
    
    Difference = Regret(A) - Regret(B). Negative means A is better (lower regret).
    """
    assert len(regrets_a) == len(regrets_b), "Paired comparisons require equal-length cohorts"
    diffs = np.asarray(regrets_a) - np.asarray(regrets_b)
    n = len(diffs)

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    rng = np.random.default_rng(seed)
    if n >= 5:
        boot_diffs = [
            float(np.mean(rng.choice(diffs, size=n, replace=True)))
            for _ in range(n_bootstrap)
        ]
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
    else:
        ci_low = mean_diff
        ci_high = mean_diff

    # Wilcoxon signed-rank test
    try:
        stat, p_val = stats.wilcoxon(regrets_a, regrets_b, zero_method="pratt")
        p_value = float(p_val)
    except Exception:
        p_value = 1.0

    return {
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p_value,
        "win_rate_a_over_b": float(np.mean(regrets_a < regrets_b)),
    }
