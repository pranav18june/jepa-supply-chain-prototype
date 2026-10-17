# LEGACY: retained for historical baseline comparison. Active controlled path: run_controlled.py.
"""Ablation study runner isolating pretraining, decision alignment, and risk weighting."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

from prototype_new.src.env.simulator import MultiEchelonSimulator, SimulationConfig
from prototype_new.src.env.disruptions import DisruptionRegime
from prototype_new.src.models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from prototype_new.src.evaluation.protocol import train_jepa_world_model, evaluate_model


def run_ablations(
    n_train_worlds: int = 60,
    n_test_worlds: int = 30,
    epochs: int = 25,
    seed: int = 9601,
    device: str = "cpu",
) -> Dict[str, Any]:
    print("=== Running Risk-Sensitive JEPA Ablation Suite ===")
    sim = MultiEchelonSimulator()
    sim_config = SimulationConfig(seed=seed, future_replications=6)
    dev = torch.device(device)

    # Generate training scenarios
    train_worlds = [
        sim.generate_world(i, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
        for i in range(n_train_worlds)
    ]
    val_worlds = [
        sim.generate_world(i + 1000, DisruptionRegime.IN_DISTRIBUTION, sim_config, training_mode=True)
        for i in range(15)
    ]

    # Generate test worlds under the Primary Target: Compound Disruption Shift!
    test_worlds = [
        sim.generate_world(i + 3000, DisruptionRegime.COMPOUND_SHIFT, sim_config, training_mode=False)
        for i in range(n_test_worlds)
    ]

    ablation_conditions = {
        "1_full_proposed": {
            "alignment_weight": 1.0,
            "weight_mode": "tail_cost",
            "freeze_encoder": False,
            "risk_beta": 0.25,
        },
        "2_no_decision_alignment": {  # lambda = 0
            "alignment_weight": 0.0,
            "weight_mode": "uniform",
            "freeze_encoder": False,
            "risk_beta": 0.0,
        },
        "3_no_risk_weighting": {      # Generic D-JEPA (w_ij = 1.0)
            "alignment_weight": 1.0,
            "weight_mode": "uniform",
            "freeze_encoder": False,
            "risk_beta": 0.0,
        },
        "4_matched_random_init": {    # Frozen random encoder diagnostic
            "alignment_weight": 1.0,
            "weight_mode": "tail_cost",
            "freeze_encoder": True,
            "risk_beta": 0.25,
        },
        "5_diff_cost_weighting": {    # Cost-difference weighting instead of tail CVaR weighting
            "alignment_weight": 1.0,
            "weight_mode": "cost_difference",
            "freeze_encoder": False,
            "risk_beta": 0.25,
        },
        "6_no_risk_beta_readout": {   # Risk-weighted training but risk_beta = 0 in selection
            "alignment_weight": 1.0,
            "weight_mode": "tail_cost",
            "freeze_encoder": False,
            "risk_beta": 0.0,
        },
    }

    results = {}
    for name, cond in ablation_conditions.items():
        print(f"Running ablation condition: {name}...")
        model = RiskSensitiveJEPAWorldModel(
            num_features=16,
            d_model=64,
            action_dim=8,
            risk_beta=cond["risk_beta"],
        )
        trained_model = train_jepa_world_model(
            model,
            train_worlds,
            val_worlds,
            alignment_weight=cond["alignment_weight"],
            weight_mode=cond["weight_mode"],
            freeze_encoder=cond["freeze_encoder"],
            epochs=epochs,
            device=dev,
        )
        metrics = evaluate_model(trained_model, test_worlds, "jepa", dev)
        results[name] = metrics.to_dict()
        print(f"  -> Regret: {metrics.mean_regret:.3f}, Tail CVaR: {metrics.tail_cvar_95:.2f}, Pairwise Acc: {metrics.pairwise_ranking_accuracy:.3f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=str, default="prototype_new/results/ablation_results.json")
    args = parser.parse_args()

    results = run_ablations(epochs=args.epochs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblations saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
