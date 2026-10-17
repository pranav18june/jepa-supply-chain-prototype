"""Uncertainty benchmark runner evaluating ensemble disagreement and fallback gating."""

from __future__ import annotations

import copy
import json
import os
import platform
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from ..env.simulator import MultiEchelonSimulator, SimulationConfig, SimulatedWorld
from ..env.disruptions import DisruptionRegime
from ..models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from ..models.uncertainty import EnsembleUncertaintyGate, compute_ensemble_uncertainties
from ..baselines.rule_based import RuleBasedPlanner
from ..baselines.forecast_optim import ForecastThenOptimizePlanner
from ..baselines.robust_optim import RobustOptimizationPlanner
from ..baselines.conventional_dynamics import ConventionalDynamicsModel
from .protocol import (
    train_jepa_world_model,
    train_conventional_dynamics,
    predict_jepa_candidate_scores,
)
from .traces import WorldTrace, save_traces_jsonl
from .gating import (
    GatedEnsemblePolicy,
    calibrate_gate_threshold,
    evaluate_policy_with_traces,
)
from .metrics import EvaluationMetrics, compute_paired_comparison


@dataclass
class UncertaintyBenchmarkConfig:
    n_train_worlds: int = 40
    n_val_worlds: int = 10
    n_calib_worlds: int = 15
    n_test_worlds_per_regime: int = 20
    n_splits: int = 3
    epochs: int = 20
    ensemble_size: int = 3
    device: str = "cpu"
    seed: int = 9601
    gate_metric: str = "top_action_disagreement"


class UncertaintyBenchmarkRunner:
    """Manages multi-split execution, gate calibration, and paired policy evaluations."""

    def __init__(self, config: UncertaintyBenchmarkConfig):
        self.config = config
        self.simulator = MultiEchelonSimulator()
        self.device = torch.device(config.device)

    def run_benchmark(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        all_traces: List[WorldTrace] = []
        regimes = [
            DisruptionRegime.IN_DISTRIBUTION,
            DisruptionRegime.SEVERITY_SHIFT,
            DisruptionRegime.COMPOUND_SHIFT,
            DisruptionRegime.MECHANISM_SHIFT,
        ]
        regime_names = [r.value for r in regimes]

        policy_names = [
            "single_member",
            "ensemble_ungated",
            "always_rule_based",
            "always_robust_optim",
            "gated_ensemble_robust",
            "gated_ensemble_rule",
        ]

        # Results accumulator: policy -> regime -> list of metrics across splits
        split_metrics: Dict[str, Dict[str, List[EvaluationMetrics]]] = {
            p: {r: [] for r in regime_names} for p in policy_names
        }
        raw_regrets: Dict[str, Dict[str, List[float]]] = {
            p: {r: [] for r in regime_names} for p in policy_names
        }

        calibrated_thresholds = []

        print(f"\n=== Starting Uncertainty-Gated JEPA Benchmark ===")
        print(f"Config: {self.config.n_train_worlds} train, {self.config.n_calib_worlds} calib, "
              f"{self.config.n_test_worlds_per_regime} test/regime, {self.config.ensemble_size} ensemble members, "
              f"{self.config.n_splits} splits")

        for split_idx in range(self.config.n_splits):
            split_seed = self.config.seed + split_idx * 1009
            np.random.seed(split_seed)
            torch.manual_seed(split_seed)
            sim_config = SimulationConfig(seed=split_seed, future_replications=6)

            print(f"\n--- Split {split_idx + 1}/{self.config.n_splits} (Seed: {split_seed}) ---")

            # 1. Generate disjoint world pools
            train_worlds = [
                self.simulator.generate_world(w_idx, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
                for w_idx in range(self.config.n_train_worlds)
            ]
            val_worlds = [
                self.simulator.generate_world(w_idx + 1000, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
                for w_idx in range(self.config.n_val_worlds)
            ]
            calib_worlds = [
                self.simulator.generate_world(w_idx + 2000, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
                for w_idx in range(self.config.n_calib_worlds)
            ]

            # 2. Train Ensemble of Learned JEPA Models
            ensemble_models: List[RiskSensitiveJEPAWorldModel] = []
            for m_idx in range(self.config.ensemble_size):
                m_seed = split_seed + m_idx * 137
                torch.manual_seed(m_seed)
                np.random.seed(m_seed)
                model = RiskSensitiveJEPAWorldModel(num_features=16, d_model=64, action_dim=8, risk_beta=0.25)
                trained_m = train_jepa_world_model(
                    model,
                    train_worlds,
                    val_worlds,
                    alignment_weight=1.0,
                    weight_mode="tail_cost",
                    epochs=self.config.epochs,
                    device=self.device,
                )
                ensemble_models.append(trained_m)

            # 3. Fit and Tune Baselines
            rule_planner = RuleBasedPlanner()
            robust_planner = RobustOptimizationPlanner()
            robust_planner.tune_on_validation(val_worlds)

            # 4. Calibrate Gate Threshold on Calibration Worlds
            calib_ungated = GatedEnsemblePolicy(
                ensemble_models=ensemble_models,
                fallback_planner=robust_planner,
                gate=EnsembleUncertaintyGate(threshold=float("inf"), metric_key=self.config.gate_metric),
                method_name="calib_ungated",
                run_id=f"split_{split_idx}",
                device=self.device,
            )
            calib_traces, _ = evaluate_policy_with_traces(calib_ungated, calib_worlds, "calib_ungated")
            thresh = calibrate_gate_threshold(
                calib_traces,
                robust_planner,
                calib_worlds,
                metric_key=self.config.gate_metric,
            )
            calibrated_thresholds.append(thresh)
            print(f"  Calibrated gate threshold ({self.config.gate_metric}): {thresh:.4f}")

            # 5. Instantiate Gated & Ungated Policies
            ungated_gate = EnsembleUncertaintyGate(threshold=float("inf"), metric_key=self.config.gate_metric)
            active_gate = EnsembleUncertaintyGate(threshold=thresh, metric_key=self.config.gate_metric)

            policies = {
                "single_member": GatedEnsemblePolicy(
                    [ensemble_models[0]], robust_planner, ungated_gate, "single_member", f"split_{split_idx}", self.device
                ),
                "ensemble_ungated": GatedEnsemblePolicy(
                    ensemble_models, robust_planner, ungated_gate, "ensemble_ungated", f"split_{split_idx}", self.device
                ),
                "always_rule_based": rule_planner,
                "always_robust_optim": robust_planner,
                "gated_ensemble_robust": GatedEnsemblePolicy(
                    ensemble_models, robust_planner, active_gate, "gated_ensemble_robust", f"split_{split_idx}", self.device
                ),
                "gated_ensemble_rule": GatedEnsemblePolicy(
                    ensemble_models, rule_planner, active_gate, "gated_ensemble_rule", f"split_{split_idx}", self.device
                ),
            }

            # 6. Evaluate all policies across all test regimes
            for regime in regimes:
                test_worlds = [
                    self.simulator.generate_world(w_idx + 3000, regime, sim_config, training_mode=False)
                    for w_idx in range(self.config.n_test_worlds_per_regime)
                ]
                for p_name, policy in policies.items():
                    traces, metrics = evaluate_policy_with_traces(
                        policy, test_worlds, p_name, run_id=f"split_{split_idx}_{regime.value}"
                    )
                    all_traces.extend(traces)
                    split_metrics[p_name][regime.value].append(metrics)
                    for t in traces:
                        raw_regrets[p_name][regime.value].append(t.evaluation.regret)

        elapsed = time.time() - start_time

        # 7. Aggregate Summary Metrics & Paired Comparisons
        summary: Dict[str, Dict[str, Any]] = {r: {} for r in regime_names}
        for r in regime_names:
            for p_name in policy_names:
                m_list = split_metrics[p_name][r]
                regrets = raw_regrets[p_name][r]
                summary[r][p_name] = {
                    "regret_mean": float(np.mean(regrets)),
                    "regret_std": float(np.std(regrets)),
                    "regret_median": float(np.median(regrets)),
                    "cost_mean": float(np.mean([m.mean_cost for m in m_list])),
                    "cvar_95": float(np.mean([m.across_world_cvar_95 for m in m_list])),
                    "top1_acc_mean": float(np.mean([m.top1_accuracy for m in m_list])),
                    "pairwise_acc_mean": float(np.mean([m.pairwise_ranking_accuracy for m in m_list])),
                    "fallback_rate_mean": float(np.mean([m.fallback_rate for m in m_list])),
                    "catastrophic_rate_mean": float(np.mean([m.catastrophic_inversion_rate for m in m_list])),
                }

        # Paired Comparisons: Gated Ensemble vs Baselines under Compound Shift
        paired_results = {}
        for baseline in ["ensemble_ungated", "always_robust_optim", "always_rule_based"]:
            paired_results[f"gated_robust_vs_{baseline}"] = compute_paired_comparison(
                raw_regrets["gated_ensemble_robust"]["compound_shift"],
                raw_regrets[baseline]["compound_shift"],
            )

        # 8. Uncertainty Diagnostics: Correlation & AUROC on Ungated Traces
        ungated_traces = [t for t in all_traces if t.inference.method == "ensemble_ungated"]
        diagnostic_results = self._compute_uncertainty_diagnostics(ungated_traces)

        # 9. Save Artifacts
        save_traces_jsonl(all_traces, output_dir / "world_traces.jsonl")

        manifest = {
            "run_id": f"uncertainty_{int(time.time())}",
            "elapsed_seconds": elapsed,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "config": asdict(self.config),
            "calibrated_thresholds": calibrated_thresholds,
            "total_traces": len(all_traces),
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        with open(output_dir / "summary_by_regime.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(output_dir / "paired_comparisons.json", "w") as f:
            json.dump(paired_results, f, indent=2)

        with open(output_dir / "uncertainty_diagnostics.json", "w") as f:
            json.dump(diagnostic_results, f, indent=2)

        print(f"\n[SUCCESS] Uncertainty benchmark finished in {elapsed:.1f}s.")
        print(f"Artifacts saved to: {output_dir.resolve()}")

        return {
            "summary": summary,
            "paired_comparisons": paired_results,
            "uncertainty_diagnostics": diagnostic_results,
            "manifest": manifest,
        }

    def _compute_uncertainty_diagnostics(self, traces: List[WorldTrace]) -> Dict[str, Any]:
        """Evaluate how well each uncertainty metric predicts decision regret and catastrophic errors."""
        if not traces:
            return {}

        regrets = np.array([t.evaluation.regret for t in traces])
        catastrophic_labels = np.array([1 if t.evaluation.is_catastrophic_inversion else 0 for t in traces])
        high_regret_labels = np.array([1 if t.evaluation.regret > float(np.median(regrets)) else 0 for t in traces])

        metric_keys = [
            "normalized_score_spread",
            "top_action_disagreement",
            "pairwise_rank_disagreement",
            "score_margin",
            "raw_score_std",
        ]

        diagnostics = {}
        for key in metric_keys:
            scores = np.array([t.inference.uncertainty_scores.get(key, 0.0) for t in traces])
            # For margin, invert score because small margin = high ambiguity
            effective_scores = -scores if key == "score_margin" else scores

            # Spearman correlation
            corr, p_val = stats.spearmanr(effective_scores, regrets)

            # AUROC and PR-AUC for high regret
            try:
                auc_high_regret = float(roc_auc_score(high_regret_labels, effective_scores))
                prec, rec, _ = precision_recall_curve(high_regret_labels, effective_scores)
                prauc_high_regret = float(auc(rec, prec))
            except Exception:
                auc_high_regret = 0.5
                prauc_high_regret = 0.5

            diagnostics[key] = {
                "spearman_correlation_with_regret": float(corr) if not np.isnan(corr) else 0.0,
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "auroc_high_regret": auc_high_regret,
                "prauc_high_regret": prauc_high_regret,
            }

        return diagnostics
