"""Comprehensive benchmark execution across methods and disruption shift regimes."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from ..env.simulator import MultiEchelonSimulator, SimulationConfig, SimulatedWorld
from ..env.disruptions import DisruptionRegime
from ..models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from ..baselines.rule_based import RuleBasedPlanner
from ..baselines.forecast_optim import ForecastThenOptimizePlanner
from ..baselines.robust_optim import RobustOptimizationPlanner
from .protocol import (
    train_jepa_world_model,
    train_conventional_dynamics,
    evaluate_model,
    evaluate_heuristic_planner,
)
from .metrics import EvaluationMetrics


@dataclass
class BenchmarkConfig:
    n_train_worlds: int = 120
    n_val_worlds: int = 30
    n_test_worlds_per_regime: int = 40
    n_splits: int = 5
    epochs: int = 30
    device: str = "cpu"
    seed: int = 9601


class BenchmarkRunner:
    """Runs the 8-method benchmark across the 4 disruption regimes over multiple splits."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.simulator = MultiEchelonSimulator()
        self.device = torch.device(self.config.device)

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Execute the complete evaluation suite."""
        print(f"=== Starting Risk-Sensitive JEPA Supply Chain Benchmark ===")
        print(f"Config: {self.config.n_train_worlds} train, {self.config.n_splits} splits, epochs={self.config.epochs}")
        start_time = time.time()

        all_results_by_regime: Dict[str, Dict[str, List[Dict[str, float]]]] = {
            regime.value: {} for regime in DisruptionRegime
        }

        methods = [
            "rule_based",
            "forecast_optim",
            "robust_optim",
            "conventional_dynamics",
            "standard_jepa",
            "matched_random_jepa",
            "generic_decision_jepa",
            "proposed_risk_sensitive_jepa",
        ]

        for regime in DisruptionRegime:
            for method in methods:
                all_results_by_regime[regime.value][method] = []

        # Run across multiple splits
        for split_idx in range(self.config.n_splits):
            split_seed = self.config.seed + 1009 * split_idx
            # Reset process-level RNGs at the split boundary so weights, minibatch
            # permutations, and bootstrap intervals are repeatable for a given split.
            np.random.seed(split_seed)
            torch.manual_seed(split_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(split_seed)
            print(f"\n--- Running Split {split_idx + 1}/{self.config.n_splits} (Seed: {split_seed}) ---")

            # 1. Generate Training & Validation Worlds (In-Distribution / Isolated training scenarios)
            sim_config = SimulationConfig(seed=split_seed, future_replications=6)
            train_worlds = [
                self.simulator.generate_world(w_idx, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
                for w_idx in range(self.config.n_train_worlds)
            ]
            val_worlds = [
                self.simulator.generate_world(w_idx + 1000, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
                for w_idx in range(self.config.n_val_worlds)
            ]

            # 2. Train and prepare models
            print("  Training baselines and JEPA models...")
            
            # Method 1: Rule-Based
            rule_planner = RuleBasedPlanner()

            # Method 2: Forecast + Optim
            forecast_planner = ForecastThenOptimizePlanner(n_estimators=30, random_state=split_seed)
            forecast_planner.fit(train_worlds)

            # Method 3: Robust Optim
            robust_planner = RobustOptimizationPlanner()

            # Method 4: Conventional Dynamics
            conv_model = train_conventional_dynamics(
                train_worlds,
                epochs=self.config.epochs,
                device=self.device,
            )

            # Method 5: Standard JEPA (lambda = 0, regression only)
            standard_jepa = RiskSensitiveJEPAWorldModel(num_features=16, d_model=64, action_dim=8)
            standard_jepa = train_jepa_world_model(
                standard_jepa,
                train_worlds,
                val_worlds,
                alignment_weight=0.0,
                weight_mode="uniform",
                epochs=self.config.epochs,
                device=self.device,
            )

            # Method 6: Matched-Random JEPA (Untrained frozen encoder diagnostic)
            matched_random_jepa = RiskSensitiveJEPAWorldModel(num_features=16, d_model=64, action_dim=8)
            matched_random_jepa = train_jepa_world_model(
                matched_random_jepa,
                train_worlds,
                val_worlds,
                alignment_weight=1.0,
                weight_mode="tail_cost",
                freeze_encoder=True,  # Frozen random weights
                epochs=self.config.epochs,
                device=self.device,
            )

            # Method 7: Generic Decision-Aligned JEPA (D-JEPA style, unweighted ranking)
            generic_decision_jepa = RiskSensitiveJEPAWorldModel(num_features=16, d_model=64, action_dim=8)
            generic_decision_jepa = train_jepa_world_model(
                generic_decision_jepa,
                train_worlds,
                val_worlds,
                alignment_weight=1.0,
                weight_mode="uniform",  # w_ij = 1.0
                epochs=self.config.epochs,
                device=self.device,
            )

            # Method 8: Proposed Risk-Sensitive Decision-Aligned JEPA (Tail-weighted ranking)
            proposed_jepa = RiskSensitiveJEPAWorldModel(num_features=16, d_model=64, action_dim=8)
            proposed_jepa = train_jepa_world_model(
                proposed_jepa,
                train_worlds,
                val_worlds,
                alignment_weight=1.0,
                weight_mode="tail_cost",  # w_ij focusing on high-cost tail outcomes
                epochs=self.config.epochs,
                device=self.device,
            )

            # 3. Evaluate across all 4 Disruption Regimes
            for regime in DisruptionRegime:
                test_worlds = [
                    self.simulator.generate_world(w_idx + 5000, regime, sim_config, training_mode=False)
                    for w_idx in range(self.config.n_test_worlds_per_regime)
                ]

                # Evaluate all methods
                m1_metrics = evaluate_heuristic_planner(rule_planner, test_worlds)
                m2_metrics = evaluate_heuristic_planner(forecast_planner, test_worlds)
                m3_metrics = evaluate_heuristic_planner(robust_planner, test_worlds)
                m4_metrics = evaluate_model(conv_model, test_worlds, "conventional_dynamics", self.device)
                m5_metrics = evaluate_model(standard_jepa, test_worlds, "jepa", self.device)
                m6_metrics = evaluate_model(matched_random_jepa, test_worlds, "jepa", self.device)
                m7_metrics = evaluate_model(generic_decision_jepa, test_worlds, "jepa", self.device)
                m8_metrics = evaluate_model(proposed_jepa, test_worlds, "jepa", self.device)

                eval_map = {
                    "rule_based": m1_metrics,
                    "forecast_optim": m2_metrics,
                    "robust_optim": m3_metrics,
                    "conventional_dynamics": m4_metrics,
                    "standard_jepa": m5_metrics,
                    "matched_random_jepa": m6_metrics,
                    "generic_decision_jepa": m7_metrics,
                    "proposed_risk_sensitive_jepa": m8_metrics,
                }

                for method_name, metrics in eval_map.items():
                    all_results_by_regime[regime.value][method_name].append(metrics.to_dict())

            print(f"  Split {split_idx + 1} completed.")

        # Aggregate summary statistics across splits
        summary_results = {}
        for regime_name, method_dict in all_results_by_regime.items():
            summary_results[regime_name] = {}
            for method_name, split_list in method_dict.items():
                regrets = [item["mean_regret"] for item in split_list]
                costs = [item["mean_cost"] for item in split_list]
                cvars = [item["tail_cvar_95"] for item in split_list]
                top1s = [item["top1_accuracy"] for item in split_list]
                pairwises = [item["pairwise_ranking_accuracy"] for item in split_list]
                catastrophics = [item["catastrophic_inversion_rate"] for item in split_list]

                summary_results[regime_name][method_name] = {
                    "regret_mean": float(np.mean(regrets)),
                    "regret_std": float(np.std(regrets)),
                    "cost_mean": float(np.mean(costs)),
                    "tail_cvar_mean": float(np.mean(cvars)),
                    "top1_acc_mean": float(np.mean(top1s)),
                    "pairwise_acc_mean": float(np.mean(pairwises)),
                    "catastrophic_rate_mean": float(np.mean(catastrophics)),
                }

        elapsed = time.time() - start_time
        return {
            "summary": summary_results,
            "raw_splits": all_results_by_regime,
            "elapsed_seconds": elapsed,
            "config": asdict(self.config),
        }
