# Experiment and reproducibility protocol

Run from `/Users/pranav/Downloads/JEPA` with Python dependencies in `requirements-verified.txt` (exact observed environment) or `requirements.txt` (portable lower bounds). CPU is the verified execution device. MPS exists but has not been qualified for numerical reproducibility. No root Git history exists; `execution_audit/pre_edit_source.tar.gz` preserves the pre-edit core, and source hashes identify final artifacts.

## Commands

```sh
python3 -m pip install -r prototype_new/requirements-verified.txt
python3 prototype_new/experiments/fetch_audit_sources.py
PYTHONPATH=. python3 -m pytest prototype_new/tests -q
python3 prototype_new/experiments/inspect_isomorph_release.py
PYTHONPATH=. python3 prototype_new/experiments/run_isomorph_prediction.py --output prototype_new/results/isomorph_prediction_new
PYTHONPATH=. python3 prototype_new/experiments/run_controlled.py --config prototype_new/configs/controlled_smoke.json --output prototype_new/results/controlled_smoke_new
PYTHONPATH=. python3 prototype_new/experiments/run_controlled.py --config prototype_new/configs/controlled_development.json --output prototype_new/results/controlled_development_new
PYTHONPATH=. python3 prototype_new/experiments/fetch_retail_subset.py
PYTHONPATH=. python3 prototype_new/experiments/run_real_data_benchmark_v2.py --config prototype_new/configs/retail_audited.json --output-dir prototype_new/results/retail_audited_new
```

Existing-cache exploratory smoke (does not establish an untouched holdout):

```sh
PYTHONPATH=. python3 prototype_new/experiments/run_real_data_benchmark_v2.py --config prototype_new/configs/retail_cache_smoke.json --output-dir prototype_new/results/retail_cache_smoke_new
```

Do not rerun into completed result directories. Data, binary checkpoints and bulky outputs are ignored by Git. Pinned source acquisition is small; optional retail train file is 106,436,287 bytes and verified by SHA256 `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d`. Full ISOMORPH (~99 GB card estimate) is neither needed nor fetched. Upstream parity fixture is fetched independently of vendored code; absent fixtures are explicitly SKIPPED, never passed.

## A — native trajectory prediction

`run_isomorph_prediction.py`: seeds 4701–4720, 12/4/4 train/validation/test rollouts, 40 days, 3 items, disjoint full configurations. Native policy only. Fit Ridge scaler on training, choose alpha from {.1,1,10,100} on validation, evaluate persistence/Ridge at 1/3/8 steps. Store per-feature MAE and mixed-unit aggregate RMSE; aggregate is diagnostic only. No actions are inferred. A JEPA comparison and full-release evaluation remain unperformed.

## B — controlled planning

Smoke: generation seeds 1701–1720, model seeds 2701/2702, 24 days, 3 items, 3-step prediction, 12/4/4 rollouts, 15 epochs. CEM 32 samples × 3 iterations, 4 elites, integer requests 0–600. Two independently initialized models form each ensemble. The four model methods use matching architecture, initialization seeds and search budgets. Baselines native policy and random feasible requests execute real simulator transitions. Learned outcome components are never used as realized costs.

Development: seeds 2701–2756, model seeds 3701/3702, 48 days, 8-step horizon, 40/8/8 rollouts, 40 epochs. Adds low/intermittent behavior after discovering zero shortage coverage in the initial smoke. This is a disclosed developmental follow-up, not independent confirmation selected after favorable results. Checkpoints use validation state/cost prediction error only. Test files are not used for early stopping. No repeated test-set tuning or selective seed reporting.

Raw files: `transitions.jsonl`, `split_manifest.json`, `training.json`, `prediction.json`, `planning_steps.jsonl`, `episodes.json`, `summary.json`, method/member checkpoints and config. Summary cost is sum of realized three-component accounting; fill rate is same-day served/arriving demand and does not include late backlog service. Mean cost confidence intervals bootstrap scenario/seed episodes (2,000 resamples with fixed seed), not dependent steps. Eight episodes are too few for strong generalization or tail-risk claims. No global-optimality regret is reported. Model variance is not called calibrated uncertainty.

Initial and repeated smoke agree exactly on split manifests, training logs, prediction metrics and realized episodes excluding measured latency. End-to-end repeatability does not establish model validity.

## C — retail

Use official train file for the frozen chronological configuration. A separate convenience subset of 256 entities supplies train-only preprocessing. Forecast horizon is always next day. Future observations are target-encoder inputs only in the training loss, never predictor input. Sales remain censored observations. Official eval is excluded from the main configuration because earlier code already consumed it.

The cache smoke trains on June 28–29 targets, validates June 30, and evaluates July 1–2 using two prior days, 512/256/512 windows. It is an engineering protocol check with historical contamination disclosed. Never pool it into a final test result. No policy effectiveness follows from predictive metrics.

## Remaining confirmatory work

Freeze a fresh protocol before new large experiments; expand native ISOMORPH behavior/configuration coverage, use multiple training seeds, supported structural stressors and learning curves, match stronger forecast-then-optimize and supervised sequence baselines, test horizon/search/uncertainty/cost sensitivities, measure per-target/action-effect calibration, and preserve untouched scenario/retail holdouts. Add variable-topology modeling only with a defined hypothesis and verified environment support. Negative results remain reportable.

### Additional executed audits

```sh
PYTHONPATH=. python3 prototype_new/experiments/evaluate_action_effects.py
PYTHONPATH=. python3 prototype_new/experiments/evaluate_forecast_baseline.py
PYTHONPATH=. python3 prototype_new/experiments/verify_results.py prototype_new/results/controlled_development
PYTHONPATH=. python3 prototype_new/experiments/verify_results.py prototype_new/results/retail_audited_verified
```

The forecast baseline uses exactly pre-day observable inventory, backlog, pipeline and EMA with the already fixed 7-day coverage coefficient. It was added as an explicitly post-audit diagnostic and is stored separately from historical summaries; subsequent controlled runs include it directly. Native policy sees same-day demand, so its timing advantage is disclosed. Action-effect diagnostics compare zero/maximum dispatch after 12 zero-dispatch days on held-out scenarios using fixed frozen member-0 models; these are new probes, not checkpoint tuning. The learned delta errors do not establish reliable counterfactual accuracy.

The executed bounded training-subset retail run is `results/retail_audited_verified/`; its metrics exactly reproduce `retail_audited/` after checksum/provenance validation was added. Source subset identity and transfer accounting are in DATA_PROTOCOL.md. Complete-file fetch remains optional and failed locally; bounded acquisition succeeded.

## D — decision_v3 protocol (round_claude, pre-registered 2026-10-17 before any v3 model was trained)

Environment `DecisionEnv` (`src/env/isomorph.py`): post-demand decision timing shared by the native rule and every controller; demand tabulated at reset and lead-time draws keyed by (day,node,item) so branches cloned from one state share all exogenous randomness; lane capacity exposed as metadata. Data: `configs/decision_v3d.json` (v3a–v3c preserved, failed coverage gates; see CHANGE_REPORT). Splits: train 400 / validation 60 / test 60 iid episodes, plus test_ood 48 episodes (higher demand, tighter capacity, bursts, longer lead times, held-out `piecewise_constant` behavior). Model selection uses validation only; test and test_ood are evaluated once per frozen configuration.

Primary estimands (fixed before training):

- **P1 action effect** — paired branches from replayed test states (probe days 8, 20, 32, 44), candidate first actions {0, native, 0.5×native, 2×native, max, 3 uniform draws}, common continuation = the logged actions. Metric: one-step delta (branch minus zero-action branch) MAE in physical units on action-affected groups (destination pipeline, last-mile warehouse inventory, destination inventory/backlog), compared with the zero-delta predictor.
- **P2 decision ranking** — Spearman correlation between predicted and realized 7-step cost across the candidate first actions of each probe, and top-1 regret (realized 7-step cost of the model's choice minus the best candidate).
- **P3 rollout** — open-loop multi-step state MAE per group at horizons 1/3/7/14 on logged test action sequences, versus persistence and the supervised direct-dynamics model.

Practical-relevance rule: a model "beats" a comparator on a metric only if the episode-clustered 95% bootstrap CI of the paired difference excludes zero and the relative improvement is ≥10%. Costs remain assumed accounting units (holding 0.001, backlog 1, transport 0.01 per unit-edge). Comparator families: persistence, train-mean delta, ridge delta, supervised direct-delta MLP (+5-member ensemble), latent model with identical encoder/predictor/decoder trained (a) decoder-only, (b) +EMA-target latent loss, (c) +SIGReg latent loss without EMA, (d) latent+SIGReg only with a frozen-latent decoder fitted afterwards; known-mechanics simulator with forecast demand (hybrid reference) and exact simulator clone (oracle ceiling, diagnostic only).
