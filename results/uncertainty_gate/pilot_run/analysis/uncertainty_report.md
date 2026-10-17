# Empirical Report: Uncertainty Diagnostics and Gated Fallback Planning

**Run ID:** `uncertainty_1792153253`  
**Execution Runtime:** 12.2 seconds  
**Total Decision Traces Audited:** 1440  

## 1. Summary of Regret and Tail Performance by Policy and Regime

| Regime | Policy | Mean Regret ± Std | Median Regret | 95% CVaR | Fallback Rate |
|---|---|---:|---:|---:|---:|
| `in_distribution` | `single_member` | 3.266 ± 14.431 | 0.000 | 562.79 | 0.0% |
| `in_distribution` | `ensemble_ungated` | 9.890 ± 33.664 | 0.000 | 562.79 | 0.0% |
| `in_distribution` | `always_rule_based` | 17.609 ± 50.354 | 3.368 | 725.46 | 0.0% |
| `in_distribution` | `always_robust_optim` | 0.899 ± 1.819 | 0.000 | 562.79 | 0.0% |
| `in_distribution` | `gated_ensemble_robust` | 0.899 ± 1.819 | 0.000 | 562.79 | 100.0% |
| `in_distribution` | `gated_ensemble_rule` | 17.609 ± 50.354 | 3.368 | 562.79 | 100.0% |
| `severity_shift` | `single_member` | 10.237 ± 16.698 | 0.000 | 977.82 | 0.0% |
| `severity_shift` | `ensemble_ungated` | 20.574 ± 76.651 | 0.000 | 1045.80 | 0.0% |
| `severity_shift` | `always_rule_based` | 60.613 ± 45.572 | 43.872 | 1099.20 | 0.0% |
| `severity_shift` | `always_robust_optim` | 3.577 ± 9.581 | 0.000 | 969.11 | 0.0% |
| `severity_shift` | `gated_ensemble_robust` | 3.577 ± 9.581 | 0.000 | 1045.80 | 100.0% |
| `severity_shift` | `gated_ensemble_rule` | 60.613 ± 45.572 | 43.872 | 1045.80 | 100.0% |
| `compound_shift` | `single_member` | 25.087 ± 33.387 | 20.130 | 1899.79 | 0.0% |
| `compound_shift` | `ensemble_ungated` | 32.123 ± 47.298 | 20.623 | 1899.79 | 0.0% |
| `compound_shift` | `always_rule_based` | 2.296 ± 7.272 | 0.000 | 1881.40 | 0.0% |
| `compound_shift` | `always_robust_optim` | 19.012 ± 16.967 | 18.788 | 1899.79 | 0.0% |
| `compound_shift` | `gated_ensemble_robust` | 19.012 ± 16.967 | 18.788 | 1899.79 | 100.0% |
| `compound_shift` | `gated_ensemble_rule` | 2.296 ± 7.272 | 0.000 | 1899.79 | 100.0% |
| `mechanism_shift` | `single_member` | 5.463 ± 1.022 | 5.484 | 1759.35 | 0.0% |
| `mechanism_shift` | `ensemble_ungated` | 5.352 ± 1.432 | 5.484 | 1759.35 | 0.0% |
| `mechanism_shift` | `always_rule_based` | 100.078 ± 26.562 | 98.837 | 1844.18 | 0.0% |
| `mechanism_shift` | `always_robust_optim` | 5.574 ± 1.040 | 5.577 | 1759.35 | 0.0% |
| `mechanism_shift` | `gated_ensemble_robust` | 5.574 ± 1.040 | 5.577 | 1759.35 | 100.0% |
| `mechanism_shift` | `gated_ensemble_rule` | 100.078 ± 26.562 | 98.837 | 1759.35 | 100.0% |

## 2. Paired Hypothesis Tests (Compound Shift Primary Target)

| Comparison Contrast | Mean Regret Diff (95% CI) | Median Diff | Wilcoxon p-value | Win Rate |
|---|---:|---:|---:|---:|
| `gated_robust_vs_ensemble_ungated` | -13.111 [-26.005, -2.622] | 0.000 | 2.5391e-02 | 100.0% |
| `gated_robust_vs_always_robust_optim` | 0.000 [0.000, 0.000] | 0.000 | nan | 0.0% |
| `gated_robust_vs_always_rule_based` | 16.716 [12.143, 21.906] | 18.788 | 2.7709e-07 | 0.0% |

## 3. Uncertainty Signal Diagnostic Predictiveness

| Uncertainty Metric | Spearman Corr with Regret | p-value | High-Regret AUROC | High-Regret PR-AUC |
|---|---:|---:|---:|---:|
| `normalized_score_spread` | -0.094 | 1.4490e-01 | 0.478 | 0.488 |
| `top_action_disagreement` | 0.106 | 1.0236e-01 | 0.507 | 0.545 |
| `pairwise_rank_disagreement` | 0.182 | 4.7343e-03 | 0.550 | 0.588 |
| `score_margin` | -0.093 | 1.4907e-01 | 0.456 | 0.474 |
| `raw_score_std` | -0.017 | 7.9794e-01 | 0.517 | 0.519 |

## 4. Top-5 Highest Regret Failure Cases (Ungated Ensemble)

| World ID | Regime | Chosen Plan | Oracle Plan | Chosen Cost | Oracle Cost | Regret | Disagreement |
|---|---|---|---|---:|---:|---:|---:|
| sev-w3006-64683 | `severity_shift` | `regular` | `priority_store` | 1048.96 | 559.68 | 489.28 | 0.67 |
| sev-w3005-62961 | `severity_shift` | `regular` | `priority_store` | 773.91 | 495.69 | 278.22 | 0.67 |
| sev-w3010-49112 | `severity_shift` | `regular` | `priority_store` | 1241.77 | 1011.67 | 230.10 | 0.33 |
| com-w3017-75095 | `compound_shift` | `regular` | `expedited` | 1693.04 | 1489.43 | 203.61 | 0.33 |
| com-w3010-34794 | `compound_shift` | `regular` | `expedited` | 1463.41 | 1260.61 | 202.80 | 0.33 |