# Experimental Report: Risk-Sensitive Decision Alignment for JEPA-Based Supply-Chain Planning

**Execution Runtime:** 11.0 seconds  
**World Splits Evaluated:** 3  

## 1. Summary of Results Across Disruption Regimes

| Disruption Regime | Method | Mean Regret | Tail Cost (95% CVaR) | Top-1 Accuracy | Pairwise Acc |
|---|---|---:|---:|---:|---:|
| `in_distribution` | `rule_based` | 23.572 ± 13.441 | 754.48 | 33.3% | 65.1% |
| `in_distribution` | `forecast_optim` | 5.037 ± 0.652 | 623.96 | 35.6% | 67.3% |
| `in_distribution` | `robust_optim` | 0.907 ± 0.304 | 610.16 | 64.4% | 90.2% |
| `in_distribution` | `conventional_dynamics` | 0.907 ± 0.304 | 610.16 | 64.4% | 87.8% |
| `in_distribution` | `standard_jepa` | 48.064 ± 53.650 | 726.56 | 37.8% | 78.0% |
| `in_distribution` | `matched_random_jepa` | 0.786 ± 0.140 | 610.16 | 68.9% | 88.2% |
| `in_distribution` | `generic_decision_jepa` | 4.685 ± 5.640 | 610.16 | 62.2% | 87.1% |
| `in_distribution` | `proposed_risk_sensitive_jepa` | 1.085 ± 0.230 | 610.16 | 62.2% | 87.6% |
| `severity_shift` | `rule_based` | 68.045 ± 8.962 | 963.69 | 0.0% | 46.0% |
| `severity_shift` | `forecast_optim` | 36.978 ± 2.534 | 852.10 | 0.0% | 46.9% |
| `severity_shift` | `robust_optim` | 4.344 ± 2.383 | 821.70 | 84.4% | 76.9% |
| `severity_shift` | `conventional_dynamics` | 16.733 ± 15.427 | 861.16 | 75.6% | 64.9% |
| `severity_shift` | `standard_jepa` | 96.866 ± 97.463 | 974.73 | 62.2% | 57.3% |
| `severity_shift` | `matched_random_jepa` | 4.344 ± 2.383 | 821.70 | 84.4% | 66.9% |
| `severity_shift` | `generic_decision_jepa` | 5.704 ± 3.150 | 821.70 | 80.0% | 66.4% |
| `severity_shift` | `proposed_risk_sensitive_jepa` | 6.363 ± 1.509 | 821.70 | 77.8% | 66.2% |
| `compound_shift` | `rule_based` | 3.240 ± 2.198 | 1756.49 | 77.8% | 66.8% |
| `compound_shift` | `forecast_optim` | 36.920 ± 6.396 | 1792.91 | 0.0% | 33.2% |
| `compound_shift` | `robust_optim` | 20.904 ± 6.134 | 1777.22 | 22.2% | 63.3% |
| `compound_shift` | `conventional_dynamics` | 20.953 ± 6.085 | 1777.22 | 20.0% | 53.5% |
| `compound_shift` | `standard_jepa` | 63.967 ± 48.785 | 1830.41 | 15.6% | 43.4% |
| `compound_shift` | `matched_random_jepa` | 21.597 ± 7.084 | 1782.34 | 22.2% | 52.8% |
| `compound_shift` | `generic_decision_jepa` | 27.706 ± 15.618 | 1777.22 | 22.2% | 51.0% |
| `compound_shift` | `proposed_risk_sensitive_jepa` | 21.564 ± 5.508 | 1782.51 | 22.2% | 52.8% |
| `mechanism_shift` | `rule_based` | 102.441 ± 5.273 | 1829.84 | 0.0% | 28.5% |
| `mechanism_shift` | `forecast_optim` | 4.992 ± 0.160 | 1739.13 | 0.0% | 68.6% |
| `mechanism_shift` | `robust_optim` | 5.458 ± 0.176 | 1739.66 | 0.0% | 61.5% |
| `mechanism_shift` | `conventional_dynamics` | 36.963 ± 44.511 | 1769.48 | 0.0% | 55.5% |
| `mechanism_shift` | `standard_jepa` | 3.502 ± 2.236 | 1737.68 | 35.6% | 81.9% |
| `mechanism_shift` | `matched_random_jepa` | 5.444 ± 0.157 | 1739.66 | 0.0% | 71.7% |
| `mechanism_shift` | `generic_decision_jepa` | 5.441 ± 0.151 | 1739.66 | 0.0% | 72.2% |
| `mechanism_shift` | `proposed_risk_sensitive_jepa` | 5.336 ± 0.275 | 1739.66 | 0.0% | 72.2% |

## 2. Pre-Registered Hypotheses Verification Matrix

| Hypothesis | Formal Assertion | Result | Evidence from Experiment |
|---|---|---|---|
| **H1 (Ranking)** | Proposed JEPA ranks candidate plans more accurately than standard JEPA | **SUPPORTED** | Pairwise accuracy: 52.8% vs 43.4% |
| **H2 (Regret)** | Proposed JEPA achieves lower regret under compound shift | **SUPPORTED** | Regret: 21.564 vs 63.967 (66.3% reduction) |
| **H3 (Tail Risk)** | Tail-weighted ranking reduces 95% CVaR total cost | **SUPPORTED** | 95% CVaR: 1782.51 vs 1830.41 (2.6% reduction) |
| **H4 (Shift Degradation)** | Gap between ID and shifted regret is smaller for proposed method | **REFUTED** | Shift degradation: +20.480 vs +15.902 |
| **H5 (Attribution)** | Decision alignment and risk weighting drive performance gain | **REFUTED** | Full: 29.664, No-Align: 28.599, Unweighted: 28.096 |

## 3. Ablation Study Breakdown (Compound Shift)

| Ablation Condition | Regret | Tail Cost (CVaR-95) | Pairwise Accuracy | Top-1 Accuracy |
|---|---:|---:|---:|---:|
| `1_full_proposed` | 29.664 | 1793.07 | 50.7% | 10.0% |
| `2_no_decision_alignment` | 28.599 | 1793.07 | 51.3% | 13.3% |
| `3_no_risk_weighting` | 28.096 | 1793.07 | 51.7% | 13.3% |
| `4_matched_random_init` | 28.096 | 1793.07 | 52.3% | 13.3% |
| `5_diff_cost_weighting` | 28.096 | 1793.07 | 51.7% | 13.3% |
| `6_no_risk_beta_readout` | 28.096 | 1793.07 | 51.7% | 13.3% |