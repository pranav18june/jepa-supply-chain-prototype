# Data protocol

## A — ISOMORPH released / generated native trajectories

Dataset: https://huggingface.co/datasets/Isomorph2026/isomorph-supply-chain-benchmark at **478ee4d339eeab4a4208b32c6a4b4a2881c999a5**, CC BY 4.0. Source: https://github.com/tuhinsahai/ISOMORPH at **9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7**, MIT. Paper linked by source: ISOMORPH: A Supply Chain Digital Twin for Simulation, Dataset Generation, and Forecasting Benchmarks, arXiv:2605.12768. Code/data correspondence is not assumed from the default branch.

Downloaded only small cards/licenses/manifest/scenario plus remote Parquet metadata/topology. `results/execution_audit/isomorph_*` records the audit. The manifest has 49 rollouts, seed 2025 throughout and horizon 52,560 days. The 50-item standalone baseline and mixture baseline have equivalent configuration. They cannot cross partitions. All release runs sharing one seed prevents a strict disjoint-seed evaluation on the release alone; generated independent rollouts are used instead.

Observed baseline Parquet metadata:

| File | Rows | Columns / meaning |
|---|---:|---|
| daily_records | 2,628,000 | day:int64, item:string, demand, served_from_stock, new_backlog_today, dest_on_hand_end_before_ship, dest_backlog_end_before_ship:int64 |
| shipments | 3,587,346 | day/arrival_day:int64, from/to/item:string, units:int64, path_nodes/edge_times:string |
| inventory_history | 34,164,000 | day:int64, node/item:string, on_hand:int64 |
| edge_list | 16 | edge_id:int64, from/to:string, travel_time_days:int64, container_volume:double, num_containers:int64, cap_per_day:double |

Counts match 52,560 × 50 daily rows and × 13 × 50 inventory rows. Other published products include backlog/intransit histories, service summary and demand-signal arrays; see the pinned card. Shipments are realized outcomes, not complete chosen actions; source procurement and unexecuted requests are not established by shipment rows. No action labels are inferred for released traces. Full-table missingness, duplicates and conservation have **not** been scanned. `bounded_parquet` enforces explicit columns and bounded batches for local data; `inspect_isomorph_release.py` reads remote footers and tiny topology only.

**Release discrepancy:** the pinned Hub manifest/scenario and actual baseline row counts support 52,560 days, whereas pinned source README examples use 7,300 and source defaults also use 7,300. The Hub card reports ~99 GB. File representation differs (Hub Parquet, source CSV). This establishes that the README command is not the exact release-generation command; it does not prove which simulator commit generated every release file. Exact historical generator provenance remains unresolved. No full corpus was downloaded.

## B — newly controlled ISOMORPH

`IsomorphEnv` generates actual state/request/next-state tuples and separate physical outcomes. Action provenance: locally extended simulator control. State provenance: simulator_state; weekday and remaining-time fields: derived. Costs: explicit assumptions. Released action coverage is never substituted for generated logs.

Rollout manifests record seed, full scenario and canonical configuration hash. `validate_rollout_splits` rejects shared rollout ID, seed or identical config across splits, including renamed duplicates. Randomly adjacent windows are never partitioned. The fixed graph and shared structural priors are not independent topology samples. Short diagnostic episodes and small catalogs are not a substitute for stress/OOD learning curves.

Demand has its own seeded NumPy generator; source lead times use the original simulator RNG. Same config/seed yields identical demand across policies, but policy-dependent source order counts may consume lead-time randomness differently. Comparisons are paired by scenario/seed, **not full common random numbers for every disturbance**. Original source also fixes network/item initialization seed at 2025. This is retained and documented, not silently “fixed.”

Source-generated native-policy trajectories in Track A contain null actions. Controlled Track B samples contain exact integer dispatch targets; realized shipments are logged separately. Native parity, inventory conservation, capacity nonnegativity, positive transit delays and action effects are tested.

## C — FreshRetailNet observational retail

Dataset: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K at **c374e85975c82816c495acfa904cc55ab14a47d3**, CC BY 4.0. Paper: https://arxiv.org/abs/2505.16319. Official card: 4.5M train rows, 350K eval rows. Verified local eval schema and 350,000 rows/350 row groups. Eval SHA256 **1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e** matches the pinned Hub LFS digest; no redundant download.

Important fields: store/product IDs, dt, globally scaled sale_amount and 24-element hours_sale; hours_stock_status is hourly out-of-stock status, stock_hour6_22_cnt counts out-of-stock hours during 06:00–21:59 (array slice `[6:22]`). Normalized sales units are not physical unscaled demand. Observed sales are stockout-censored; no demand recovery is claimed.

Validation rejects duplicate entity/date keys, malformed hourly lengths, nonfinite/negative sales, invalid stock flags and nonchronological cutoffs. Windows require consecutive dates and strictly prior observations. Targets: next-day observed sales and next-day `stock_hour6_22_cnt > 0`. Target-day status/sales are training labels only. Train-only normalization; validation-only epoch selection; separate chronological target dates. Metrics include observed-sales MAE/RMSE/WAPE (null for zero denominator), AUROC (null for single class), average precision and Brier, plus stockout/in-stock strata. No thresholds are chosen from test data.

`retail_audited.json`: proposed official-train split through May 31 / June 12 / June 25, 2024, with 5-day histories. Only run if the pinned training file is available and dates validate. `retail_cache_smoke.json`: 2-day histories on existing eval cache, train targets June 28–29, validation June 30, test July 1–2. This is **exploratory engineering evidence**, since old experiments already used official eval. It is not an untouched final holdout. Convenience subset selection (first 256 entities in file order) is recorded and not presented as population-representative.

No retail-to-simulator coupling is implemented. Neither track C nor synthetic tracks A/B supports a real-world counterfactual policy claim.

### Executed training-subset audit

Full-file and initial HfFileSystem access timed out. `fetch_retail_subset.py` then used an immutable revision URL and checked HTTP 206/Content-Range responses, a 32 MiB budget and 256 KiB chunks. It fetched row groups 0–23 (24,000 rows) from the official 4,500,000-row train Parquet with 10,229,439 bytes transferred. Source object length 106,436,287; full-source SHA is published by Hub but **not locally verified** because the full object was not downloaded. The subset's own SHA256 is verified before every run and its row-group/column provenance is in `train_subset.manifest.json`.

Subset dates: March 28–June 25, 2024. All 24,000 inspected hourly status arrays agree with `stock_hour6_22_cnt = sum(hours_stock_status[6:22])` (not `[6:23]`). Validator now checks this equality and range 0–16. The 256 selected complete series yield 15,360 train targets (April 2–May 31), 3,072 validation (June 1–12), 3,328 test (June 13–25). Test dates were excluded from fitting and epoch selection. This subset test is separate from previously consumed official eval, but remains a convenience-sample, single-seed result rather than full-dataset evidence.

## decision_v3 controlled data (round_claude, 2026-10-17)

Generator: `experiments/collect_decision_data.py` with `DecisionEnv(crn=True)`. Each transition persists: episode index (→ manifest rollout id, split, seed, scenario, config hash, behavior family and parameters, full demand table for diagnostics), day, post-demand state, executed (validated) request, next post-demand state, realized dispatch per item, last-mile utilization per lane, transport/holding/backlog units, next-day demand and served, cost. Behavior policies read only the observation and their own RNG (`np.random.default_rng([seed,17])`); none reads exogenous tables. Stored as `results/<run>/data/transitions.npz` + `manifest.json` (ignored path).

| Version | Config | Seeds | Episodes (train/val/test/ood) | Coverage gate | Status |
|---|---|---|---|---|---|
| v3a | `decision_v3.json` | 31000– | 240/48/48/48 | FAIL: backlog-positive 5.0%, recovery events 63, at-bound requests 0.2% | preserved, not used |
| v3b | `decision_v3b.json` | 32000– | 240/48/48/48 | FAIL: backlog 6.6%, at-bound 1.7% | preserved, not used |
| v3c | `decision_v3c.json` | 33000– | 240/48/48/48 | FAIL: recovery events 96 (<100) | preserved, not used |
| **v3d** | `decision_v3d.json` | 34000–34567 | 400/60/60/48 | **PASS** (all 8 checks) | primary |

Gates (`experiments/coverage_report.py`, fixed before any v3 model was trained): train backlog-positive ≥10%, truncated requests ≥10%, lane-saturated steps ≥3%, zero requests ≥5%, at-bound requests ≥2%, recovery events ≥100, no behavior family >40% of transitions, iid test mass in (backlog, capacity, last-mile stock, request) cells unseen in training ≤5%. v3d train: 24,000 transitions; backlog-positive 13.6%, truncated 25.3% (12.7% by capacity), saturated 14.0%, recovery events 201, unseen-cell test mass 0.000 (iid) / 0.0045 (ood). The v3a→v3d iterations changed behavior policies and training size only, never models; v3d was declared the final data revision before modeling.

Behavior families: `native_noise` (native rule × lognormal noise, random zero days), `basestock` (per-item coverage k~U[0,14] order-up-to), `uniform` (U[0,1000], bang-bang extremes with p=0.25), `intermittent` (bursts of uniform requests), `mixed_item` (per-item family switching every 5–14 days), `starve_recover` (zero dispatch 10–30 days then base-stock recovery). test_ood: higher demand, tighter capacity, longer lead times, demand bursts, held-out `piecewise_constant` behavior.

## FreshRetailNet representative sample (round_claude)

The historical 24-row-group subset contains 267 series from only 3 stores (of 898 in eval) — not representative. `fetch_retail_subset.py --sample 300 --seed 6101` draws 300 official-train row groups uniformly without replacement via bounded HTTP range reads (pinned revision c374e85…, retries added to `HTTPRangeReader`); the manifest records row groups, seed, transferred bytes and subset SHA256. Row groups are entity clusters, so this is a cluster sample of series; the resulting store/product counts are recorded in the run manifest.
