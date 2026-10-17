"""Analysis and visualization of uncertainty diagnostics, paired comparisons, and fallback reliability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from prototype_new.src.evaluation.traces import load_traces_jsonl, WorldTrace


def generate_uncertainty_plots(traces: List[WorldTrace], summary: Dict[str, Any], output_dir: Path) -> None:
    if not HAS_MATPLOTLIB or not traces:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Regret by Regime Bar Chart (Paired)
    fig, ax = plt.subplots(figsize=(12, 5))
    regimes = list(summary.keys())
    policies = ["ensemble_ungated", "always_robust_optim", "always_rule_based", "gated_ensemble_robust"]
    labels = ["Ensemble Ungated", "Always Robust", "Always Rule-Based", "Gated Ensemble (Robust)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    x = np.arange(len(regimes))
    width = 0.18
    for idx, (pol, label, c) in enumerate(zip(policies, labels, colors)):
        regrets = [summary[r][pol]["regret_mean"] for r in regimes]
        errs = [summary[r][pol]["regret_std"] for r in regimes]
        ax.bar(x + (idx - 1.5) * width, regrets, width, yerr=errs, capsize=3, label=label, color=c, alpha=0.85)

    ax.set_ylabel("Decision Regret (lower is better)")
    ax.set_title("Decision Regret Across Disruption Regimes (With Epistemic Gating)")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n").title() for r in regimes])
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "regret_by_regime_paired.png", dpi=200)
    plt.close(fig)

    # 2. Uncertainty vs Realized Regret Scatterplot
    ungated_traces = [t for t in traces if t.inference.method == "ensemble_ungated"]
    if ungated_traces:
        disagreements = [t.inference.uncertainty_scores.get("top_action_disagreement", 0.0) for t in ungated_traces]
        regrets = [t.evaluation.regret for t in ungated_traces]
        regimes_trace = [t.evaluation.regime for t in ungated_traces]

        fig, ax = plt.subplots(figsize=(8, 5))
        regime_colors = {
            "in_distribution": "#2ca02c",
            "severity_shift": "#1f77b4",
            "compound_shift": "#d62728",
            "mechanism_shift": "#9467bd",
        }
        for r_name in set(regimes_trace):
            sub_x = [d for d, r in zip(disagreements, regimes_trace) if r == r_name]
            sub_y = [rg for rg, r in zip(regrets, regimes_trace) if r == r_name]
            ax.scatter(sub_x, sub_y, label=r_name.replace("_", " ").title(), color=regime_colors.get(r_name, "gray"), alpha=0.7)

        ax.set_xlabel("Top-Action Disagreement across Ensemble Members")
        ax.set_ylabel("Realized Decision Regret")
        ax.set_title("Epistemic Disagreement vs. Realized Decision Regret")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / "uncertainty_vs_regret.png", dpi=200)
        plt.close(fig)

    print(f"Figures saved to: {output_dir.resolve()}")


def generate_report_markdown(
    traces: List[WorldTrace],
    summary: Dict[str, Any],
    paired: Dict[str, Any],
    diagnostics: Dict[str, Any],
    manifest: Dict[str, Any],
    output_path: Path,
) -> None:
    lines = []
    lines.append("# Empirical Report: Uncertainty Diagnostics and Gated Fallback Planning\n")
    lines.append(f"**Run ID:** `{manifest.get('run_id', 'unknown')}`  ")
    lines.append(f"**Execution Runtime:** {manifest.get('elapsed_seconds', 0.0):.1f} seconds  ")
    lines.append(f"**Total Decision Traces Audited:** {manifest.get('total_traces', len(traces))}  \n")

    lines.append("## 1. Summary of Regret and Tail Performance by Policy and Regime\n")
    lines.append("| Regime | Policy | Mean Regret ± Std | Median Regret | 95% CVaR | Fallback Rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r, pols in summary.items():
        for p, stats in pols.items():
            r_str = f"{stats['regret_mean']:.3f} ± {stats['regret_std']:.3f}"
            fb_str = f"{stats['fallback_rate_mean'] * 100:.1f}%"
            lines.append(f"| `{r}` | `{p}` | {r_str} | {stats['regret_median']:.3f} | {stats['cvar_95']:.2f} | {fb_str} |")

    lines.append("\n## 2. Paired Hypothesis Tests (Compound Shift Primary Target)\n")
    lines.append("| Comparison Contrast | Mean Regret Diff (95% CI) | Median Diff | Wilcoxon p-value | Win Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for comp, stats in paired.items():
        ci_str = f"[{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
        lines.append(f"| `{comp}` | {stats['mean_difference']:.3f} {ci_str} | {stats['median_difference']:.3f} | {stats['p_value']:.4e} | {stats['win_rate_a_over_b'] * 100:.1f}% |")

    lines.append("\n## 3. Uncertainty Signal Diagnostic Predictiveness\n")
    lines.append("| Uncertainty Metric | Spearman Corr with Regret | p-value | High-Regret AUROC | High-Regret PR-AUC |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, stats in diagnostics.items():
        lines.append(f"| `{key}` | {stats['spearman_correlation_with_regret']:.3f} | {stats['p_value']:.4e} | {stats['auroc_high_regret']:.3f} | {stats['prauc_high_regret']:.3f} |")

    # Worst-case failure analysis
    lines.append("\n## 4. Top-5 Highest Regret Failure Cases (Ungated Ensemble)\n")
    ungated_traces = [t for t in traces if t.inference.method == "ensemble_ungated"]
    sorted_failures = sorted(ungated_traces, key=lambda t: t.evaluation.regret, reverse=True)[:5]
    lines.append("| World ID | Regime | Chosen Plan | Oracle Plan | Chosen Cost | Oracle Cost | Regret | Disagreement |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for t in sorted_failures:
        u_dis = t.inference.uncertainty_scores.get("top_action_disagreement", 0.0)
        lines.append(f"| {t.evaluation.world_id} | `{t.evaluation.regime}` | `{t.inference.chosen_plan}` | `{t.evaluation.oracle_plan}` | {t.evaluation.chosen_cost:.2f} | {t.evaluation.oracle_cost:.2f} | {t.evaluation.regret:.2f} | {u_dis:.2f} |")

    output_path.write_text("\n".join(lines))
    print(f"Report written to: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncertainty gate benchmark artifacts")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing run artifacts")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save report and plots")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir or in_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    with open(in_dir / "summary_by_regime.json", "r") as f:
        summary = json.load(f)
    with open(in_dir / "paired_comparisons.json", "r") as f:
        paired = json.load(f)
    with open(in_dir / "uncertainty_diagnostics.json", "r") as f:
        diagnostics = json.load(f)

    traces = load_traces_jsonl(in_dir / "world_traces.jsonl")

    generate_uncertainty_plots(traces, summary, out_dir)
    generate_report_markdown(traces, summary, paired, diagnostics, manifest, out_dir / "uncertainty_report.md")


if __name__ == "__main__":
    main()
