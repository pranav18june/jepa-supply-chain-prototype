# LEGACY: retained for historical baseline comparison. Active controlled path: run_controlled.py.
"""Generate markdown report, hypothesis verification matrix, and visualization plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_plots(benchmark_data: Dict[str, Any], output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    summary = benchmark_data["summary"]
    regimes = list(summary.keys())
    
    key_methods = [
        "rule_based",
        "forecast_optim",
        "robust_optim",
        "standard_jepa",
        "generic_decision_jepa",
        "proposed_risk_sensitive_jepa",
    ]
    method_labels = [
        "Rule-Based",
        "Forecast+Opt",
        "Robust Optim",
        "Standard JEPA",
        "Generic D-JEPA",
        "Proposed Risk JEPA",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Regret across regimes
    x = np.arange(len(regimes))
    width = 0.13
    for idx, (m, label) in enumerate(zip(key_methods, method_labels)):
        regrets = [summary[r][m]["regret_mean"] for r in regimes]
        axes[0].bar(x + (idx - 2.5) * width, regrets, width, label=label)

    axes[0].set_ylabel("Decision Regret (lower is better)")
    axes[0].set_title("Decision Regret Across Disruption Regimes")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r.replace("_", "\n").title() for r in regimes])
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Tail Cost (95% CVaR) under Compound Shift
    compound_cvars = [summary["compound_shift"][m]["tail_cvar_mean"] for m in key_methods]
    colors = ["#7f7f7f", "#aec7e8", "#ffbb78", "#1f77b4", "#ff7f0e", "#2ca02c"]
    axes[1].set_ylabel("95% CVaR Total Cost (lower is better)")
    axes[1].set_title("Tail-Risk Total Cost Under Compound Shift")
    axes[1].set_xticks(range(len(method_labels)))
    axes[1].set_xticklabels(method_labels, rotation=25, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to: {output_path.resolve()}")


def generate_markdown_report(
    benchmark_path: Path,
    ablation_path: Path,
    output_md_path: Path,
) -> str:
    with open(benchmark_path, "r") as f:
        bench = json.load(f)

    ablations = {}
    if ablation_path.exists():
        with open(ablation_path, "r") as f:
            ablations = json.load(f)

    summary = bench["summary"]
    regimes = list(summary.keys())

    md = []
    md.append("# Experimental Report: Risk-Sensitive Decision Alignment for JEPA-Based Supply-Chain Planning\n")
    md.append(f"**Execution Runtime:** {bench.get('elapsed_seconds', 0.0):.1f} seconds  ")
    md.append(f"**World Splits Evaluated:** {bench.get('config', {}).get('n_splits', 5)}  \n")

    md.append("## 1. Summary of Results Across Disruption Regimes\n")
    md.append("| Disruption Regime | Method | Mean Regret | Tail Cost (95% CVaR) | Top-1 Accuracy | Pairwise Acc |")
    md.append("|---|---|---:|---:|---:|---:|")

    for r in regimes:
        for m, stats in summary[r].items():
            regret_str = f"{stats['regret_mean']:.3f} ± {stats['regret_std']:.3f}"
            cvar_str = f"{stats['tail_cvar_mean']:.2f}"
            top1_str = f"{stats['top1_acc_mean'] * 100:.1f}%"
            pair_str = f"{stats['pairwise_acc_mean'] * 100:.1f}%"
            md.append(f"| `{r}` | `{m}` | {regret_str} | {cvar_str} | {top1_str} | {pair_str} |")

    # Hypothesis Verification
    md.append("\n## 2. Pre-Registered Hypotheses Verification Matrix\n")
    md.append("| Hypothesis | Formal Assertion | Result | Evidence from Experiment |")
    md.append("|---|---|---|---|")

    # Check H1: Action-ranking quality
    prop_acc = summary["compound_shift"]["proposed_risk_sensitive_jepa"]["pairwise_acc_mean"]
    std_acc = summary["compound_shift"]["standard_jepa"]["pairwise_acc_mean"]
    h1_pass = prop_acc >= std_acc
    md.append(f"| **H1 (Ranking)** | Proposed JEPA ranks candidate plans more accurately than standard JEPA | {'**SUPPORTED**' if h1_pass else '**REFUTED**'} | Pairwise accuracy: {prop_acc*100:.1f}% vs {std_acc*100:.1f}% |")

    # Check H2: Regret under Compound Shift
    prop_reg = summary["compound_shift"]["proposed_risk_sensitive_jepa"]["regret_mean"]
    std_reg = summary["compound_shift"]["standard_jepa"]["regret_mean"]
    h2_pass = prop_reg <= std_reg
    rel_red = ((std_reg - prop_reg) / std_reg) * 100 if std_reg > 0 else 0
    md.append(f"| **H2 (Regret)** | Proposed JEPA achieves lower regret under compound shift | {'**SUPPORTED**' if h2_pass else '**REFUTED**'} | Regret: {prop_reg:.3f} vs {std_reg:.3f} ({rel_red:.1f}% reduction) |")

    # Check H3: Tail risk behavior
    prop_cvar = summary["compound_shift"]["proposed_risk_sensitive_jepa"]["tail_cvar_mean"]
    std_cvar = summary["compound_shift"]["standard_jepa"]["tail_cvar_mean"]
    h3_pass = prop_cvar <= std_cvar
    cvar_red = ((std_cvar - prop_cvar) / std_cvar) * 100 if std_cvar > 0 else 0
    md.append(f"| **H3 (Tail Risk)** | Tail-weighted ranking reduces 95% CVaR total cost | {'**SUPPORTED**' if h3_pass else '**REFUTED**'} | 95% CVaR: {prop_cvar:.2f} vs {std_cvar:.2f} ({cvar_red:.1f}% reduction) |")

    # Check H4: Distribution Shift Degradation
    id_reg = summary["in_distribution"]["proposed_risk_sensitive_jepa"]["regret_mean"]
    std_id_reg = summary["in_distribution"]["standard_jepa"]["regret_mean"]
    prop_degrad = prop_reg - id_reg
    std_degrad = std_reg - std_id_reg
    h4_pass = prop_degrad <= std_degrad
    md.append(f"| **H4 (Shift Degradation)** | Gap between ID and shifted regret is smaller for proposed method | {'**SUPPORTED**' if h4_pass else '**REFUTED**'} | Shift degradation: +{prop_degrad:.3f} vs +{std_degrad:.3f} |")

    # Check H5: Attribution via Ablations
    if ablations:
        full_reg = ablations.get("1_full_proposed", {}).get("mean_regret", 0)
        no_align_reg = ablations.get("2_no_decision_alignment", {}).get("mean_regret", 0)
        no_risk_reg = ablations.get("3_no_risk_weighting", {}).get("mean_regret", 0)
        h5_pass = full_reg < no_align_reg
        md.append(f"| **H5 (Attribution)** | Decision alignment and risk weighting drive performance gain | {'**SUPPORTED**' if h5_pass else '**REFUTED**'} | Full: {full_reg:.3f}, No-Align: {no_align_reg:.3f}, Unweighted: {no_risk_reg:.3f} |")

    if ablations:
        md.append("\n## 3. Ablation Study Breakdown (Compound Shift)\n")
        md.append("| Ablation Condition | Regret | Tail Cost (CVaR-95) | Pairwise Accuracy | Top-1 Accuracy |")
        md.append("|---|---:|---:|---:|---:|")
        for cond, metrics in ablations.items():
            md.append(f"| `{cond}` | {metrics['mean_regret']:.3f} | {metrics['tail_cvar_95']:.2f} | {metrics['pairwise_ranking_accuracy']*100:.1f}% | {metrics['top1_accuracy']*100:.1f}% |")

    content = "\n".join(md)
    with open(output_md_path, "w") as f:
        f.write(content)

    print(f"Markdown report generated at: {output_md_path.resolve()}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="prototype_new/results/benchmark_results.json")
    parser.add_argument("--ablations", type=str, default="prototype_new/results/ablation_results.json")
    parser.add_argument("--output-md", type=str, default="prototype_new/results/REPORT.md")
    parser.add_argument("--output-plot", type=str, default="prototype_new/results/benchmark_comparison.png")
    args = parser.parse_args()

    bench_path = Path(args.benchmark)
    abl_path = Path(args.ablations)
    out_md = Path(args.output_md)
    out_plot = Path(args.output_plot)

    if not bench_path.exists():
        print(f"Error: {bench_path} not found. Please run run_benchmark.py first.")
        return

    generate_markdown_report(bench_path, abl_path, out_md)
    with open(bench_path, "r") as f:
        bench_data = json.load(f)
    generate_plots(bench_data, out_plot)


if __name__ == "__main__":
    main()
