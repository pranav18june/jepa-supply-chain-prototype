# LEGACY: retained for historical baseline comparison. Active controlled path: run_controlled.py.
"""CLI entrypoint to run the full benchmark across all 8 methods and 4 disruption regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from prototype_new.src.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Risk-Sensitive JEPA Supply Chain Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke benchmark")
    parser.add_argument("--output-dir", type=str, default="prototype_new/results", help="Directory to save results")
    parser.add_argument("--splits", type=int, default=None, help="Override number of world splits")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        config_path = Path("prototype_new/configs/quick_smoke.json")
    elif args.config:
        config_path = Path(args.config)
    else:
        config_path = Path("prototype_new/configs/quick_smoke.json")

    print(f"Loading benchmark config from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    if args.splits is not None:
        config_dict["n_splits"] = args.splits
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs

    benchmark_config = BenchmarkConfig(**config_dict)
    runner = BenchmarkRunner(benchmark_config)
    results = runner.run_full_benchmark()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_results.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Benchmark completed in {results['elapsed_seconds']:.1f}s.")
    print(f"Results saved to: {out_file.resolve()}")

    # Print summary table preview
    print("\n" + "="*80)
    print(f"{'Regime':<20} | {'Method':<30} | {'Regret Mean ± Std':<20} | {'Tail CVaR-95':<12}")
    print("="*80)
    for regime, methods in results["summary"].items():
        for method, stats in methods.items():
            r_str = f"{stats['regret_mean']:.3f} ± {stats['regret_std']:.3f}"
            c_str = f"{stats['tail_cvar_mean']:.2f}"
            print(f"{regime:<20} | {method:<30} | {r_str:<20} | {c_str:<12}")
        print("-" * 80)


if __name__ == "__main__":
    main()
