# LEGACY: retained for historical baseline comparison. Active controlled path: run_controlled.py.
"""CLI entrypoint to execute uncertainty-gated benchmark runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from prototype_new.src.evaluation.uncertainty_benchmark import (
    UncertaintyBenchmarkConfig,
    UncertaintyBenchmarkRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Uncertainty-Gated JEPA Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--pilot", action="store_true", help="Run diagnostic pilot")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save run artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        config_path = Path("prototype_new/configs/uncertainty_smoke.json")
        default_dir = f"prototype_new/results/uncertainty_gate/smoke_{int(time.time())}"
    elif args.pilot:
        config_path = Path("prototype_new/configs/uncertainty_pilot.json")
        default_dir = f"prototype_new/results/uncertainty_gate/pilot_{int(time.time())}"
    elif args.config:
        config_path = Path(args.config)
        default_dir = f"prototype_new/results/uncertainty_gate/run_{int(time.time())}"
    else:
        config_path = Path("prototype_new/configs/uncertainty_smoke.json")
        default_dir = f"prototype_new/results/uncertainty_gate/smoke_{int(time.time())}"

    out_dir = Path(args.output_dir or default_dir)

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    cfg = UncertaintyBenchmarkConfig(**cfg_dict)
    runner = UncertaintyBenchmarkRunner(cfg)
    results = runner.run_benchmark(out_dir)

    print("\n" + "=" * 90)
    print(f"{'Regime':<18} | {'Policy':<24} | {'Regret Mean ± Std':<20} | {'CVaR-95':<10} | {'Fallback %':<10}")
    print("=" * 90)
    for regime, policies in results["summary"].items():
        for pol, stats in policies.items():
            r_str = f"{stats['regret_mean']:.3f} ± {stats['regret_std']:.3f}"
            c_str = f"{stats['cvar_95']:.2f}"
            fb_str = f"{stats['fallback_rate_mean'] * 100:.1f}%"
            print(f"{regime:<18} | {pol:<24} | {r_str:<20} | {c_str:<10} | {fb_str:<10}")
        print("-" * 90)


if __name__ == "__main__":
    main()
