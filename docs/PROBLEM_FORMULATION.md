# Problem formulation — decision_v3 (round_claude, 2026-10-17)

This defines the controlled action-conditioned prediction and planning problem used from round_claude onward. Earlier runs (`IsomorphEnv`, pre-demand timing) are retained for reproduction only; their numbers are not comparable with decision_v3 numbers.

## Environment

ISOMORPH day-level simulator, MIT source `tuhinsahai/ISOMORPH@9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7` (`simulator/Supplychaingeo_item50.py`), vendored with two documented local changes (dispatch override; step split into pre/post-dispatch phases with unchanged order). Adapter: `DecisionEnv` in `src/env/isomorph.py`. Fixed 13-node network (3 magic-replenishment sources, 9 intermediate (s,S) warehouses, destination NewYork), first 3 catalog items, network capacities from the original 50-item builder. Scenario parameters (per episode, sampled from `configs/decision_v3d.json`): `ss_scale`, `base_lambda_lo/hi`, `leadtime_scale`, `containers_scale`, `burst_rate_scale`.

## Information timing (one decision day t)

1. Upstream (s,S) replenishment orders due on day t arrive; destination shipments due on day t arrive and first clear backlog.
2. Daily edge container capacity resets.
3. Destination demand of day t is realized; the demand EMA updates; unmet demand becomes backlog.
4. **Observation `s_t` is formed and the controller chooses `a_t`.**
5. Dispatch: for each item (round-robin order), up to `a_t[i]` units are allocated from warehouses in shortest-path order, bounded by warehouse inventory and per-edge container capacity; shipped units enter the destination pipeline with deterministic arrival day.
6. Inter-warehouse (s,S) replenishment and source ordering (lead time ~ Normal(mean, 0.2·mean), ceil, ≥1).
7. Steps 1–3 for day t+1, then `s_{t+1}` is observed.

The built-in ISOMORPH rule chooses after step 3 too, so native and learned controllers are information-matched. The legacy pre-demand protocol gave the native rule same-day demand that learned/forecast controllers did not see.

## State `s_t` (214 dimensions, names from `DecisionEnv.state_names()`)

| Group | Content | Status |
|---|---|---|
| per node × item | on-hand inventory, backlog, outstanding order qty, days to arrival | observed |
| per item | demand EMA (α=0.05), today's realized demand | observed |
| per item | destination pipeline by arrival offset 1..17 days | observed |
| calendar | sin/cos of day-of-week | observed |
| capacity | daily last-mile lane capacity (2 lanes) | observed exogenous metadata |
| demand intensity λ(t), future demand draws, future lead-time draws | — | hidden exogenous (never observed by any controller) |

Simulator-only oracle variables (full demand table, keyed lead-time draws) are used only by the diagnostic oracle reference, never by a controller or a learned model. Lane capacity is computed upstream from the realized horizon-mean intensity of the 50-item catalog; across training episodes it correlates with future 3-item demand (Spearman 0.70) but adds little over today's demand (R² 0.729 → 0.765). It is exposed equally to all controllers and documented as such.

## Action `a_t`

Integer per-item destination dispatch request, `a_t ∈ {0,…,1000}^3`. Projection = round and clip; `validate` rejects anything else. The executed (validated) request is logged; realized dispatch (≤ request) is a recorded outcome, not an action. This is **not** procurement, supplier selection, rerouting or expediting control; those are unavailable in this interface.

## Exogenous randomness and pairing

Demand is tabulated at reset from the upstream generator (identical values to upstream sequential calls). With `crn=True`, source lead-time draws are keyed by (day, node, item). Hence branches cloned from one state share all exogenous randomness and differ only by the intervention (verified: `tests/test_decision_env.py`; 0 replay mismatches on 108 test episodes). Under the legacy sequential RNG, lead-time draws were consumed policy-dependently (1 vs 111 draws over 48 days for native vs random), so multi-step pairing was broken there.

## Objective (assumed accounting units, not monetary data)

Per transition t → t+1: `cost = 0.001·Σ inventory(all nodes, s_{t+1}) + 1.0·Σ destination backlog(s_{t+1}) + 0.01·Σ shipped units × path edges (day t dispatch and replenishment)`. Holding and backlog are functions of the next observed state; transport is a step outcome predicted as an extra target. Upstream holding is large and mostly action-independent, so controllable differences are a small share of total cost; component breakdowns and regret are reported. Service is reported as same-day fill rate (served from stock / demand) and backlog. Cost-weight sensitivity is required for planning claims.

## Prediction targets

Next-state per-feature deltas for every feature whose delta varies in training (161 of 214), plus transport units. Multi-step predictions are open-loop rollouts given an action sequence. Latent losses are auxiliary; only physical-unit predictions and their implied costs are evaluated.

## Primary estimands

Pre-registered in `EXPERIMENT_PROTOCOL.md` section D: P1 paired one-step action-effect error on action-affected groups vs zero-delta; P2 Spearman/regret of predicted 7-step cost across paired candidate first actions; P3 open-loop rollout error at 1/3/7/14 steps. Planning (section E, if reached) is evaluated by realized simulator cost over 60 decision days plus a 14-day native-rule tail that penalizes terminal state.
