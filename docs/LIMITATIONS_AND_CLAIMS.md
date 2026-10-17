# Limitations and scientific claim audit

**Status: partial research implementation. Critical scientific acceptance gates remain open.** Passing tests does not validate policy benefit.

| Claim | Evidence / status |
|---|---|
| Parameterized feasible actions replace five templates in the new proposed path | Supported: integer per-item dispatch targets, CEM and MPC; old five-template runner explicitly legacy. Request feasibility is distinct from realized shipment fulfillment. |
| Adapter preserves upstream default behavior | Supported on exact tested fixtures: 50 items/8 days/seed 5 and 3 items/32 days/seed 19, comparing records, state and RNG. Not exhaustive proof for all simulator extensions. |
| Actions affect transitions | Supported in paired environment tests; learned prediction accepts aligned action sequences. Counterfactual accuracy across action support remains insufficiently evaluated. |
| JEPA planners outperform the native policy | **Not supported.** Initial smoke strongly underperformed; broader development still has lower mean cost under native policy. No favorable conclusion is substituted. |
| SIGReg is better than EMA in general | Not established; only a small developmental comparison, two initialization members and eight scenarios. |
| Model uncertainty is calibrated / CVaR-sensitive | Not supported. Ensemble scoring supports an optional dispersion penalty; reported experiments use zero uncertainty weight. |
| Released ISOMORPH data contains arbitrary chosen actions | Not supported; shipment outcomes do not identify full control decisions. Newly generated controlled tuples are separate. |
| Pinned source exactly generated the pinned public release | Unresolved. Release horizon corroborated by manifest and baseline row counts; source README/default horizon differs. |
| Topology generalization | Not tested; fixed 13-node network, numerical fixed-order encoder. |
| True dollar savings / real-world policy benefit | Unsupported: simulator and cost assumptions are synthetic, retail is observational. |
| FreshRetailNet sales are true unconstrained demand | False under stockouts; report observed-sales targets and stockout strata only. |
| Cache retail smoke is untouched final evaluation | False; old experiments already used official eval. New runner labels this explicitly. |
| Old reference-source benchmark scores are interchangeable with new results | False; architecture, action set, objectives, protocol and data differ. Historical results are retained but unverified. |

## Completed engineering checks

Source inventory and backup; selected reference hashes/licenses; bounded release inspection; narrow ISOMORPH local control extension with parity and conservation checks; real logged controlled transitions; split-overlap rejection; autoregressive EMA/SIGReg/supervised/no-action model variants; CEM, integer projection, goal-weight response, execute-observe-replan and logged numerical fallback; checkpoint buffers round-trip; deterministic smoke repetition; corrected retail next-day windows and target semantics.

## Partial / unresolved

Reference audit identifies architectures and exact audited-file identities, not every upstream capability or weight release. ISOMORPH full-release integrity and historical generator mapping are unresolved. Training/cost/action coverage, service quality, OOD stress and statistical power are insufficient for paper claims. Scope of control is dispatch only; upstream source/intermediate replenishment remains fixed. Source lead-time RNG does not provide full policy-invariant common random numbers. Small-catalog capacity configuration differs in load from the release. No hard predicted service constraint, terminal value, graph encoder or calibrated tail criterion is claimed.

Retail main chronological split ran on a 24,000-row bounded official-train subset after full-file/initial remote reads timed out; checked HTTP ranges succeeded. A fresh test-date partition was retained within this subset. Existing eval cache is verified, but its previous use prevents a pristine-final-holdout claim. Current retail baselines omit the full requested supervised temporal/JEPA ablation suite and multi-seed confidence intervals.

## Not started / deliberately excluded

Unseen topology experiments; full-release neural Track A benchmark; systematic cost and censoring sensitivity; full structural-shift suite; real operations interventions; optional retail-driven simulator coupling; OR-Gym fallback (unneeded for the validated narrow action interface); incorporation of restricted SupplyChain-JEPA code. No image/video weights are transferred to numerical data.


## Final evidence interpretation

Matched-observation forecast dispatch mean cost 690.349375 versus EMA 1806.12725, SIGReg 1004.977875 and supervised 1798.540875 in the development scenarios. Native reference 697.177375 has stronger same-day information. These results support keeping the simple forecast baseline; they do not justify replacing it with a learned planner. Paired action-delta MAE is 29.3893 (EMA), 38.7054 (SIGReg), 31.4679 (supervised), versus 26.8048 for the zero-delta no-action model; metric mixes state units and is diagnostic, but it does not support a strong action-consequence accuracy claim.

Retail subset sales MAE: latent adaptation 0.432846, ExtraTrees 0.434827, last value 0.488741. Stockout AUROC: latent 0.662660, ExtraTrees 0.668363, history frequency 0.625401. No multi-seed significance or JEPA-specific attribution is established; a matched supervised neural ablation is still needed. No real-world intervention claim follows.

## round_claude claim separation (2026-10-17)

| Track | Supported claim | Not supported / not claimed |
|---|---|---|
| A. Native-policy trajectory prediction (synthetic) | Ridge beats persistence on mixed-unit RMSE in 3/4 legacy test rollouts at horizons 1/3/8 (diagnostic only). | Any action-conditioned or causal claim; no action variation exists in these trajectories. |
| B. Controlled action-effect / counterfactual prediction (decision_v3d) | In-distribution, the supervised direct-delta MLP (5-seed ensemble) beats zero-change on paired one-step action effects (−18%, episode-bootstrap CI excludes 0); its ensemble spread ranks its 7-day cost error (Spearman 0.58 iid, 0.84 OOD). | OOD action effects (no learned model beats zero-change); any benefit of EMA/SIGReg/JEPA-only latent objectives (none observed; latent models reproduce only 12–35% of the true request-sweep response); calibrated uncertainty. |
| C. Simulator planning (decision_v3d, 60 paired test scenarios) | Learned-model CEM-MPC beats the native ISOMORPH rule on average under information-matched timing and assumed costs. | Advantage over the validation-tuned base-stock rule (all learned planners worse on average; CIs include 0); robustness; real-world effectiveness; real monetary value. |
| D. FreshRetailNet observational prediction (random 300-row-group sample, 5 seeds) | End-to-end neural networks slightly beat ExtraTrees on observed-sales MAE (−0.0052, entity-clustered CI excludes 0). | Any JEPA-specific benefit (EMA latent loss vs supervised: −0.0003, CI includes 0); latent demand recovery; decision/policy claims. |

Additional limitations: single fixed 13-node topology and 3-item catalogue; dispatch requests are the only controllable action; costs are project-chosen accounting units; planning used one training seed per learned model plus one ensemble; CEM warm-started at the native action; no hard service-level constraints. The earlier report's "no-action" planner was an action-blind model planner, not a zero-dispatch policy.
