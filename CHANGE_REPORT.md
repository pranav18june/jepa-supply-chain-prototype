# Execution change report

Status: IN PROGRESS. No research acceptance claim. Date: 2026-10-17 (Asia/Kolkata).

## Phase 0 — inventory and baseline

Read IMPLEMENTATION_PLAN.md in full. Root is /Users/pranav/Downloads/JEPA, without Git metadata. Reference SupplyChain-JEPA checkout is clean at 214d115506db80e51922bb95f252ffc7edf7374f. Full file inventory, dependency versions, hardware and source backup are under results/execution_audit. Original references and historical results preserved.

- `PYTHONPATH=. python3 -m pytest prototype_new/tests -q`: exit 0, **21 PASS / 0 FAIL / 0 SKIPPED**, 3.29 s; baseline_tests.log.
- `PYTHONPATH=. python3 prototype_new/experiments/run_benchmark.py --quick --splits 1 --epochs 1 --output-dir prototype_new/results/execution_audit/legacy_smoke`: exit 0; baseline_smoke.log and legacy_smoke/benchmark_results.json. Engineering smoke only, not new scientific evidence.
- Python 3.14.6, torch 2.12.1, macOS arm64, MPS available, CUDA unavailable; ~176 GiB free initially.

Pre-existing scientific problems: five fixed templates; supervised cost training described as JEPA; retail v2 trains on labels called validation and predicts a different horizon from the same inputs; no latent JEPA loss in retail Transformer; hardcoded favorable conclusions. Historical outputs remain unverified and are not new evidence.

Added docs/WORKSPACE_INVENTORY.md and this report. No source redesign before baseline.

## Phase 1/2 — source and release audit (ongoing)

SupplyChain-JEPA explicitly reserves rights: inspect concepts, do not copy source. LeWM and V-JEPA local archives have MIT license files but no Git metadata; do not invent archive commits. ISOMORPH Hub manifest pinned to 478ee4d339eeab4a4208b32c6a4b4a2881c999a5. FreshRetailNet Hub pinned to c374e85975c82816c495acfa904cc55ab14a47d3. Metadata downloaded with immutable resolve URLs; full ISOMORPH corpus not downloaded.

## Phase 3–6 — controlled path implemented

Added `src/env/isomorph.py`, attributed MIT `src/vendor/isomorph/`, `src/models/controlled.py`, `src/planning/cem.py`, `src/data/protocols.py`, pinned source-fetch/release-inspection scripts and controlled experiment runner/configs. Optional integer per-item destination dispatch targets retain native routing/capacity/replenishment transitions. A 3-item catalog is explicitly a small-catalog experiment retaining original network capacity, not the 50-item release. No source/reference original edited. Goal accounting is assumed holding/backlog/unit-edge cost, not published monetary data.

- `PYTHONPATH=. python3 -m pytest prototype_new/tests/test_controlled.py -q`: exit 0, 10 PASS / 0 FAIL / 0 SKIPPED, 2.94 s. Exact default-policy parity against independent unmodified source for (seed 5, 50 items, 8 days) and (seed 19, 3 items, 32 days).
- `PYTHONPATH=. python3 -m pytest prototype_new/tests -q`: exit 0, 36 PASS / 0 FAIL / 0 SKIPPED, 3.95 s; tests_phase3.log. Adds conservation, scalar SIGReg reference, chronology and bounded-read checks.
- `PYTHONPATH=. python3 prototype_new/experiments/run_controlled.py`: exit 0; controlled_smoke/. Seed 1701 generation; config includes 12/4/4 rollout partitions, 24 days, 3-step horizon, 15 epochs, two model initializations per method. Negative result: native mean cost 349.99 vs EMA/SIGReg/supervised learned planners 8893.0635. No training transition had positive backlog (0/288); unsupported shortage costs explain a serious behavior-coverage gap. This is not a successful planning result.

The follow-up development config uses new seeds starting 2701, low-dispatch and intermittent behavior, 48-day episodes, 8-step prediction, 40 epochs. It is a diagnostic extension after inspecting the first smoke; the old smoke holdout is retired from confirmatory use. Old results retained.

Root .gitignore now excludes data/caches/checkpoints/build artifacts. requirements-verified.txt records actually installed versions; portable requirements gain pyarrow and huggingface-hub. Source snapshots exist because there is no root Git history.

## Phase 7–9 — evaluation, reproducibility and final audit

**Overall status: PARTIAL. No claim that the implementation plan's scientific acceptance gates are complete.**

Primary architecture is in `src/env/isomorph.py`, `src/models/controlled.py`, `src/planning/cem.py`, `src/data/protocols.py`, `src/data/http_range.py`. Original ISOMORPH MIT source is attributed under `src/vendor/isomorph/`; five fixed templates remain legacy baselines only. Active entrypoints and legacy status are documented in README. The old retail v2 source was replaced in place; its original is in the source backup. IMPLEMENTATION_PLAN.md was not changed.

### Provenance and release findings

- ISOMORPH dataset: `478ee4d339eeab4a4208b32c6a4b4a2881c999a5`, CC BY 4.0. Simulator: `9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7`, MIT. Manifest 49 rollouts, common seed 2025, horizon 52,560 days. Baseline daily/inventory rows independently corroborate this horizon. Pinned source README examples/default are 7,300 days; exact historical source-to-release mapping remains unresolved. ~99 GB is a card estimate, not a measured full download. Shipments are execution outcomes, not arbitrary chosen actions.
- Local dispatch-target extension preserves routing/packing/replenishment; parity evidence above. OR-Gym not selected because this narrow interface is testable. It does not offer arbitrary source orders/routing/expediting. Native policy sees same-day demand; learned and added forecast controller use pre-day state. Original builder uses horizon-wide demand intensity to size last-mile capacity. Both limitations are documented.
- LeWorldModel identity verified by matching local jepa.py/module.py/train.py/LICENSE to upstream `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; MIT. V-JEPA AC source/trainer/LICENSE match `204698b45b3712590f06245fbfba32d3be539812`; MIT. Full archive commits are unknown. No visual weights used. EMA and SIGReg are separate numerical-state adaptations, both with auxiliary supervision.
- SupplyChain-JEPA origin/clean commit `214d115506db80e51922bb95f252ffc7edf7374f`, branch main; no open-source license, no copied code. Separate outsourced archive provenance is not conflated with that checkout.
- FreshRetailNet pinned `c374e85975c82816c495acfa904cc55ab14a47d3`, CC BY 4.0. Existing eval SHA matches Hub. Full train fetch and first remote attempt FAILED with timeouts (fetch.log, fetch_retry.log); bounded range acquisition SUCCEEDED: 24,000 rows / 24 row groups, 10,229,439 bytes transferred, source rows 4.5M. Subset SHA and range provenance are verified. No full-source hash claim. Cleanup removed only this run's incomplete download files.

### Exact verification and experiment commands

All commands run from `/Users/pranav/Downloads/JEPA`. Outputs are under `prototype_new/results/`.

| Command | Exit / result | Evidence |
|---|---|---|
| `PYTHONPATH=. python3 -m pytest prototype_new/tests -q` (initial) | 0; 21 PASS, 0 FAIL, 0 SKIPPED | execution_audit/baseline_tests.log |
| Same command (final) | 0; 39 PASS, 0 FAIL, 0 SKIPPED; 3.85 s | execution_audit/final_tests.log |
| `python3 -m compileall -q prototype_new/src prototype_new/experiments` | 0 | Python compilation check |
| `PYTHONPATH=. python3 prototype_new/experiments/run_controlled.py` | 0; seed 1701; controlled_smoke.json | controlled_smoke/ |
| Same with `--output prototype_new/results/controlled_smoke_repeat` | 0; same manifests/training/prediction/realized episodes, excluding latency | execution_audit/reproducibility.json |
| Same with `--config prototype_new/configs/controlled_development.json --output prototype_new/results/controlled_development` | 0; seed 2701; 40/8/8 rollouts, 48 days, horizon 8, 40 epochs, 2 members | controlled_development/ |
| Same with `--output prototype_new/results/controlled_final_smoke` | 0; final runner additionally includes matched-observation forecast baseline | controlled_final_smoke/ |
| `PYTHONPATH=. python3 prototype_new/experiments/run_isomorph_prediction.py` | 0; seeds 4701–4720; 12/4/4 native rollouts; 24 test baseline/horizon/rollout values | isomorph_prediction/ |
| `PYTHONPATH=. python3 prototype_new/experiments/evaluate_forecast_baseline.py` | 0; same eight development scenarios, fixed pre-day 7-day target | controlled_development/forecast_baseline.json |
| `PYTHONPATH=. python3 prototype_new/experiments/evaluate_action_effects.py` | 0; frozen member 0; eight paired scenarios/method | controlled_development/action_effects.json |
| `PYTHONPATH=. python3 prototype_new/experiments/fetch_retail_subset.py` | 0; bounded pinned acquisition | execution_audit/retail_subset_fetch.log; data/fresh_retail_net/data/train_subset.manifest.json |
| `PYTHONPATH=. python3 prototype_new/experiments/run_real_data_benchmark_v2.py --config prototype_new/configs/retail_cache_smoke.json --output-dir prototype_new/results/retail_cache_smoke` | 0; seed 5701; exploratory cache protocol | retail_cache_smoke/ |
| Same runner with `--output-dir prototype_new/results/retail_audited` | 0; retail_audited.json, seed 5701, 256 entities | retail_audited/ |
| Same runner with `--output-dir prototype_new/results/retail_audited_verified` | 0; checksum/provenance validated; metrics reproduce exactly | retail_audited_verified/ |
| `PYTHONPATH=. python3 prototype_new/experiments/verify_results.py prototype_new/results/controlled_development` | 0; 48 episodes recomputed from step logs | raw/summary consistency |
| Same verifier for `prototype_new/results/retail_audited_verified` | 0; 9,984 prediction rows recompute metrics | raw/summary consistency |
| Same verifier for `prototype_new/results/controlled_final_smoke` | 0 | final-run summary consistency |

Upstream V-JEPA/LeWM/SupplyChain-JEPA full training/test suites were NOT RUN; local adaptation test counts do not include or imply their success. Source clones failed with network timeout, so pinned raw source/API acquisition was used and recorded. Reference originals were not edited.

### Track B: developmental controlled results

Costs below are actual simulator-executed assumed accounting units, not learned predictions or real dollars. Native policy has stronger information timing; matched forecast baseline is also provided separately. No superiority claim.

| Method | Episodes | Mean cost | Mean same-day fill |
|---|---:|---:|---:|
| native_policy | 8 | 697.177375 | 1.000000 |
| random_feasible | 8 | 2422.355250 | 1.000000 |
| ema_jepa | 8 | 1806.127250 | 0.937732 |
| sigreg_jepa | 8 | 1004.977875 | 0.988742 |
| supervised | 8 | 1798.540875 | 0.928318 |
| no_action | 8 | 2544.385125 | 1.000000 |
| forecast_dispatch (matched observations) | 8 | 690.349375 | 1.000000 |

The first smoke's zero-shortage training coverage was a real failure (0/288 positive-backlog transitions), not erased by the later development result. Development includes low/intermittent action policies. Paired action-delta probes still do not show strong counterfactual accuracy: no-action zero-delta baseline has lower mixed-unit delta MAE than learned models. Latent objective and physical policy benefit must not be conflated.

### Track A: synthetic native prediction

Generated native-policy trajectories only; null actions. Persistence and validation-selected ridge (alpha 100) evaluated at 1/3/8 steps. Per-feature errors/raw values are in isomorph_prediction/metrics.json. Full-release JEPA comparison is not performed; mixed-unit aggregate RMSE is diagnostic, not a physically meaningful single operational score.

### Track C: real observational prediction

Frozen train/validation/test target dates end May 31 / June 12 / June 25, with five prior days as input, 256 convenience-selected series, seed 5701, 20 epochs and validation-only epoch selection. Counts: 15,360 / 3,072 / 3,328. Labels are observed sales and next-day stockout indicator, not true uncensored demand. Hourly validation confirms [6:22] alignment. Official eval is excluded from this run; the separately labeled cache smoke is not pristine holdout evidence.

| Method | Test windows | Sales MAE | Sales WAPE | Stockout AUROC |
|---|---:|---:|---:|---:|
| last_value | 3328 | 0.488741 | 0.404577 | 0.625401 |
| extratrees | 3328 | 0.434827 | 0.359948 | 0.668363 |
| ema_latent_supervised | 3328 | 0.432846 | 0.358307 | 0.662660 |

The small sales-error gap versus ExtraTrees is not evidence of a general JEPA benefit; stockout AUROC is lower for the latent model. No confidence-interval or matched supervised-neural attribution claim is made. The data does not support real-world counterfactual policies. No retail/simulator coupling was performed.

### Acceptance status

| Workstream | Status |
|---|---|
| Recursive inventory, preservation, baseline logs | Completed for local core; reference full-suite execution not performed |
| Reference identities/licenses and selected-source audit | Completed at documented scope; archive whole-tree commits unknown |
| ISOMORPH manifest/schema pinning | Partial; actual baseline rows checked, full release not scanned, historical generator mapping unresolved |
| Controlled interface / parity / physical invariants | Completed for narrow dispatch extension and tested fixtures; not arbitrary order/network control |
| Parameterized action / CEM / MPC / logged fallback | Implemented and tested; no hard service constraint or calibrated risk guarantee |
| Action-conditioned latent learning | Implemented with aligned generated data; counterfactual accuracy/behavior coverage still insufficient |
| Track A evaluation | Partial; native generated baselines, no full-release neural comparison |
| Track B evaluation | Partial/negative; matched forecast and native policies outperform learned methods; broad stress/ablation/power gates open |
| Track C evaluation | Partial; verified chronological bounded subset, not full benchmark or multi-seed neural ablations |
| Reproducibility / raw-summary checks | Passed at executed CPU smoke scope; no clean-venv install or accelerator verification |
| OR-Gym / retail-to-simulator coupling / topology transfer | Not started; not necessary for narrow adapter, no claims |
| Scientific acceptance for paper / real operational claims | NOT SATISFIED |

### Deviations and remaining work

No silent plan rewrite. Limited dispatch control is an explicitly scoped valid local extension. Fixed graph, 3-item catalog and assumed costs narrow conclusions. Full inventory/procurement controls, future-scenario OOD coverage, longer-horizon/terminal handling, service constraints, calibrated uncertainty, learning curves, full baseline/ablation suite and multiple independently trained runs remain. Resolve source-release provenance with authoritative evidence rather than inventing it. Preserve new untouched holdouts before any further protocol tuning. A simple forecast controller is the supported practical baseline in these developmental results.

### File-level changes

Machine-readable paths/hashes and exact diff against the pre-edit archive: `results/execution_audit/changed_files.json`, `changes.patch`, `final_source_hashes.json`. Root `.gitignore` was added; IMPLEMENTATION_PLAN.md and original reference trees were preserved. Generated artifacts are ignored, not committed. Changed core files at this checkpoint:

- added: `prototype_new/CHANGE_REPORT.md`
- modified: `prototype_new/README.md`
- added: `prototype_new/configs/controlled_development.json`
- added: `prototype_new/configs/controlled_smoke.json`
- added: `prototype_new/configs/retail_audited.json`
- added: `prototype_new/configs/retail_cache_smoke.json`
- added: `prototype_new/docs/ARCHITECTURE.md`
- added: `prototype_new/docs/DATA_PROTOCOL.md`
- added: `prototype_new/docs/EXPERIMENT_PROTOCOL.md`
- added: `prototype_new/docs/LIMITATIONS_AND_CLAIMS.md`
- added: `prototype_new/docs/REFERENCE_AUDIT.md`
- added: `prototype_new/docs/WORKSPACE_INVENTORY.md`
- added: `prototype_new/experiments/evaluate_action_effects.py`
- added: `prototype_new/experiments/evaluate_forecast_baseline.py`
- added: `prototype_new/experiments/fetch_audit_sources.py`
- added: `prototype_new/experiments/fetch_retail_subset.py`
- modified: `prototype_new/experiments/generate_report.py`
- added: `prototype_new/experiments/inspect_isomorph_release.py`
- modified: `prototype_new/experiments/run_ablations.py`
- modified: `prototype_new/experiments/run_benchmark.py`
- added: `prototype_new/experiments/run_controlled.py`
- added: `prototype_new/experiments/run_isomorph_prediction.py`
- modified: `prototype_new/experiments/run_real_data_benchmark.py`
- modified: `prototype_new/experiments/run_real_data_benchmark_v2.py`
- modified: `prototype_new/experiments/run_uncertainty_gate.py`
- added: `prototype_new/experiments/verify_results.py`
- modified: `prototype_new/pyproject.toml`
- added: `prototype_new/requirements-verified.txt`
- modified: `prototype_new/requirements.txt`
- modified: `prototype_new/src/env/actions.py`
- added: `prototype_new/src/env/isomorph.py`
- added: `prototype_new/src/models/LEWM_LICENSE`
- added: `prototype_new/src/models/controlled.py`
- added: `prototype_new/src/planning/__init__.py`
- added: `prototype_new/src/planning/cem.py`
- added: `prototype_new/src/vendor/__init__.py`
- added: `prototype_new/src/vendor/isomorph/LICENSE`
- added: `prototype_new/src/vendor/isomorph/PROVENANCE.json`
- added: `prototype_new/src/vendor/isomorph/__init__.py`
- added: `prototype_new/src/vendor/isomorph/simulator.py`
- added: `prototype_new/tests/test_controlled.py`
- added: `prototype_new/tests/test_data_protocols.py`
- added: `prototype_new/tests/test_http_range.py`

### Final preservation and supplemental verification

- `PYTHONPATH=. python3 -m pytest prototype/tests -q`: exit 0, 12 PASS / 0 FAIL / 0 SKIPPED, 2.79 s. This is a post-change check of the unchanged legacy tree, not the pre-edit core baseline.
- Final retail schema/hourly validator passed all 24,000 downloaded subset rows after adding operating-hour count consistency.
- Final raw-summary verifiers passed 9,984 retail prediction rows and 28 final-smoke episodes. No final background download or experiment is left running.
- Root ignore rules scope data artifacts to `/prototype_new/data/` and `/prototype/data/`, so the new `src/data/` source package remains includable.

Additional new source files included in the final diff: `src/data/__init__.py`, `src/data/protocols.py`, `src/data/http_range.py`. Final changed core file count: 46.
