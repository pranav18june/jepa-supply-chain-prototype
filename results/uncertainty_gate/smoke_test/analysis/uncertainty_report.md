# Empirical Report: Uncertainty Diagnostics and Gated Fallback Planning

**Run ID:** `uncertainty_1792153217`  
**Execution Runtime:** 1.9 seconds  
**Total Decision Traces Audited:** 240  

## 1. Summary of Regret and Tail Performance by Policy and Regime

| Regime | Policy | Mean Regret ± Std | Median Regret | 95% CVaR | Fallback Rate |
|---|---|---:|---:|---:|---:|
| `in_distribution` | `single_member` | 4.432 ± 4.639 | 2.935 | 453.04 | 0.0% |
| `in_distribution` | `ensemble_ungated` | 1.666 ± 4.998 | 0.000 | 445.47 | 0.0% |
| `in_distribution` | `always_rule_based` | 4.594 ± 4.502 | 2.935 | 453.04 | 0.0% |
| `in_distribution` | `always_robust_optim` | 0.000 ± 0.000 | 0.000 | 445.47 | 0.0% |
| `in_distribution` | `gated_ensemble_robust` | 0.000 ± 0.000 | 0.000 | 445.47 | 80.0% |
| `in_distribution` | `gated_ensemble_rule` | 4.432 ± 4.639 | 2.935 | 445.47 | 80.0% |
| `severity_shift` | `single_member` | 28.774 ± 13.375 | 31.255 | 856.43 | 0.0% |
| `severity_shift` | `ensemble_ungated` | 4.084 ± 6.989 | 0.000 | 841.78 | 0.0% |
| `severity_shift` | `always_rule_based` | 73.803 ± 59.210 | 45.018 | 848.14 | 0.0% |
| `severity_shift` | `always_robust_optim` | 4.084 ± 6.989 | 0.000 | 841.78 | 0.0% |
| `severity_shift` | `gated_ensemble_robust` | 4.084 ± 6.989 | 0.000 | 841.78 | 80.0% |
| `severity_shift` | `gated_ensemble_rule` | 55.079 ± 56.596 | 37.228 | 841.78 | 80.0% |
| `compound_shift` | `single_member` | 33.699 ± 14.068 | 32.032 | 1906.27 | 0.0% |
| `compound_shift` | `ensemble_ungated` | 24.574 ± 13.960 | 24.542 | 1890.91 | 0.0% |
| `compound_shift` | `always_rule_based` | 0.000 ± 0.000 | 0.000 | 1880.81 | 0.0% |
| `compound_shift` | `always_robust_optim` | 19.840 ± 15.570 | 16.743 | 1890.91 | 0.0% |
| `compound_shift` | `gated_ensemble_robust` | 19.840 ± 15.570 | 16.743 | 1890.91 | 90.0% |
| `compound_shift` | `gated_ensemble_rule` | 4.197 ± 12.591 | 0.000 | 1890.91 | 90.0% |
| `mechanism_shift` | `single_member` | 5.216 ± 0.847 | 5.270 | 1733.41 | 0.0% |
| `mechanism_shift` | `ensemble_ungated` | 5.550 ± 0.892 | 5.491 | 1734.79 | 0.0% |
| `mechanism_shift` | `always_rule_based` | 97.661 ± 24.677 | 93.375 | 1808.87 | 0.0% |
| `mechanism_shift` | `always_robust_optim` | 5.652 ± 1.043 | 5.799 | 1734.79 | 0.0% |
| `mechanism_shift` | `gated_ensemble_robust` | 5.652 ± 1.043 | 5.799 | 1734.79 | 90.0% |
| `mechanism_shift` | `gated_ensemble_rule` | 91.602 ± 36.207 | 93.375 | 1734.79 | 90.0% |

## 2. Paired Hypothesis Tests (Compound Shift Primary Target)

| Comparison Contrast | Mean Regret Diff (95% CI) | Median Diff | Wilcoxon p-value | Win Rate |
|---|---:|---:|---:|---:|
| `gated_robust_vs_ensemble_ungated` | -4.734 [-9.469, 0.000] | 0.000 | 2.5000e-01 | 100.0% |
| `gated_robust_vs_always_robust_optim` | 0.000 [0.000, 0.000] | 0.000 | 1.0000e+00 | 0.0% |
| `gated_robust_vs_always_rule_based` | 19.840 [10.476, 29.813] | 16.743 | 1.9531e-03 | 0.0% |

## 3. Uncertainty Signal Diagnostic Predictiveness

| Uncertainty Metric | Spearman Corr with Regret | p-value | High-Regret AUROC | High-Regret PR-AUC |
|---|---:|---:|---:|---:|
| `normalized_score_spread` | -0.031 | 8.5021e-01 | 0.475 | 0.480 |
| `top_action_disagreement` | -0.016 | 9.2354e-01 | 0.500 | 0.712 |
| `pairwise_rank_disagreement` | 0.000 | 1.0000e+00 | 0.500 | 0.750 |
| `score_margin` | -0.152 | 3.4865e-01 | 0.415 | 0.522 |
| `raw_score_std` | -0.051 | 7.5692e-01 | 0.475 | 0.484 |

## 4. Top-5 Highest Regret Failure Cases (Ungated Ensemble)

| World ID | Regime | Chosen Plan | Oracle Plan | Chosen Cost | Oracle Cost | Regret | Disagreement |
|---|---|---|---|---:|---:|---:|---:|
| com-w3003-19287 | `compound_shift` | `priority_store` | `expedited` | 1340.56 | 1296.72 | 43.84 | 0.50 |
| com-w3006-11043 | `compound_shift` | `priority_store` | `expedited` | 1522.89 | 1480.92 | 41.97 | 0.00 |
| com-w3004-72656 | `compound_shift` | `split` | `expedited` | 1619.73 | 1581.12 | 38.61 | 0.50 |
| com-w3005-79253 | `compound_shift` | `priority_store` | `expedited` | 1580.69 | 1547.47 | 33.22 | 0.50 |
| com-w3009-70888 | `compound_shift` | `priority_store` | `expedited` | 1805.04 | 1776.64 | 28.40 | 0.50 |