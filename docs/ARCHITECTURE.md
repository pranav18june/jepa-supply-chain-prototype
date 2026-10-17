# Active architecture

The controlled entrypoint is `experiments/run_controlled.py`; the old `run_benchmark.py` and its five-template policies are **legacy baselines**, not the proposed planner. No historical output was overwritten.

## Physical environment and action contract

`src/env/isomorph.py:IsomorphEnv` wraps an attributed local copy of the pinned ISOMORPH base simulator. The only change inside the original simulator is an optional `external_dispatch_targets` override of `ship_target`. `None` executes the original policy. It is a **local extension**, not a native public agent API.

At the start of day t the controller observes the previous day's state and submits one nonnegative integer **requested destination-dispatch quantity per active item**. Bounds are explicit in `DispatchSpace` (smoke: 0–600 units/item/day), and projection clips and rounds. This request is an upper target, not a promise that physical shipments equal the request. After arrivals and demand, original source selection/routing and first-fit packing execute up to that target subject to on-hand stock and every route's container capacity. Remaining warehouse and source replenishment uses original (s,S) logic. No custom supplier choice, expedite action, arbitrary route or money-cost mechanics are claimed.

Logs separate chosen targets, realized dispatch, native replenishment shipments, before/after observation, demand, served units and cost components. Zero request suppresses destination dispatch; it does not suppress native warehouse replenishment. Actions in native-policy trajectories are `null`, not fabricated labels.

The 3-item experiment uses the first-N catalog convention supported by upstream v2, retaining the 50-item builder's network/capacity settings. It is deliberately a small-catalog experiment, not equivalent to the released 50-item workload. Parity also tests the original 50-item catalog.

## Observation and time

`observe()` returns fixed node/item ordering of on-hand inventory, backlog, outstanding-order quantity and remaining days, demand EMA, destination pipeline indexed by arrival offset, and weekday sin/cos. Full simulator RNG/future demand intensity and future demand samples are hidden. Pipeline and outstanding orders are available under the explicit digital-twin observability assumption. State is partially observable because the demand process has latent factors. No arbitrary-topology or graph-network generalization claim.

Day transition: receive warehouse orders → receive destination shipments/serve old backlog → reset edge capacities → sample/serve new demand → controlled destination dispatch → native intermediate replenishment → native source orders → snapshot. Training tuple `(observation_t, request_t, observation_{t+1}, components_t)` follows that convention. Multi-step sequences stay inside one rollout.

## Model

`src/models/controlled.py:ControlledWorldModel` encodes normalized numerical state into a latent, then autoregresses one latent per action. Readouts estimate future normalized state and three nonnegative accounting components. Tensor contracts: state `[B,D]`, action sequence `[B,H,A]`, latent `[B,H,L]`, future state `[B,H,D]`, components `[B,H,3]`.

- EMA variant: frozen target encoder, stop-gradient future targets, explicit post-optimizer EMA update.
- SIGReg variant: shared encoder, differentiable targets, Epps–Pulley regularizer; no EMA. Formula checked against scalar numerical integration.
- Supervised variant: identical trainable architecture/readouts without latent loss.
- No-action ablation: EMA variant receives zero action embeddings.

Auxiliary state and cost supervision exists in all variants. These are reference-inspired adaptations, not reproductions of V-JEPA or LeWM. Mean/std and cost scales fit only on training transitions, persist as model buffers, and round-trip in checkpoints. Validation state/cost error selects checkpoints. Training logs include losses, gradient norms and latent dispersion.

## Goals and planning

Physical cost assumptions: 0.001 per unit-day of network on-hand inventory, 1 per unit-day of destination backlog, 0.01 per unit-edge shipped (all shipments). These are dimensioned accounting units, **not validated monetary costs**. `Goal` separately weights the three components. Service is reported from newly served demand; it is not a hard service constraint in the planner.

`src/planning/cem.py:CEMPlanner` samples integer-projected action sequences, evaluates model rollouts, selects elites, refits mean/std and retains the best feasible sequence. A tiny enumeration test verifies an exact toy optimum and response to goal weights. Model ensembles score expected weighted cost plus optional standard-deviation penalty. This is not calibrated tail risk or CVaR. Default experiments use uncertainty weight zero; no uncertainty-aware success claim.

`RecedingHorizonController` executes only the first action and replans after every observation. Each result records components, seed, evaluations, latency and fallback. Nonfinite model output triggers a logged zero-dispatch fallback. This is a feasible computational fallback, not an assurance of service safety. Horizon remains fixed near episode end; planned continuation has no terminal-value correction, a limitation of these developmental runs.

## Retail

`experiments/run_real_data_benchmark_v2.py` replaces the invalid legacy v2 protocol in place. Chronological next-day windows use historical observed sales and stockout-hour counts. A separately trained EMA latent model with supervised sales and stockout heads is compared with last-value/history-frequency and ExtraTrees baselines. Retail has no action input and supplies no planning evidence.

### Native-policy comparison caveats found in source audit

The native dispatch rule sees same-day sampled demand before choosing its target, whereas the external controller commits its request from the previous observation. Its information timing is therefore stronger; the native policy is a reference, not a perfectly matched-information algorithm comparison. Matched learned variants share the same timing. A matched-information forecast policy is required before attributing performance differences solely to representation learning. Also, the original builder sizes last-mile container volume using full-horizon generated demand intensity. This native scenario-construction assumption is retained for parity and further limits real-world causal interpretation.

## decision_v3 architecture (round_claude, 2026-10-17)

Active path; the earlier `IsomorphEnv` / `ControlledWorldModel` / `run_controlled.py` path is retained only to reproduce earlier results.

```
vendor/isomorph/simulator.py  (MIT, step split into before/after dispatch)
        │
env/isomorph.py: DecisionEnv  (post-demand decision, tabulated demand, keyed lead times, clone(), native_action())
        │
experiments/collect_decision_data.py ─► results/decision_v3d/data/{transitions.npz, manifest.json}
        │                                   └─► experiments/coverage_report.py (gate)
experiments/run_dynamics_v3.py ─► models/controlled.py: Normalizer, ConstantDelta, RidgeDynamics, DeltaDynamics, LatentDynamics(objective), Ensemble
        │
experiments/evaluate_dynamics_v3.py (replay + clone; P1/P2/P3, sweep, uncertainty; sim_forecast reference)
        └─► experiments/summarize_dynamics_v3.py (seed means, episode-clustered bootstrap)
experiments/run_planning_v3.py ─► planning/cem.py (CEMPlanner with warm start) ; baselines native, base-stock k, zero, random; sim_forecast / sim_oracle references
experiments/run_retail_attribution.py (FreshRetailNet track; independent of the simulator)
```

Shared interface: every dynamics model implements `rollout(states [B,D], actions [B,H,A]) -> (states [B,H,D], transport [B,H])` in physical units; costs are computed from predicted states by the same known cost function in evaluation and planning (no learned cost head). Targets are per-feature normalized one-step deltas (features whose delta never varies are held fixed; constant drifts kept); inventory-type features are projected nonnegative. `LatentDynamics` rolls out in latent space and decodes consecutive latents into deltas; objectives `latent_sup`, `ema`, `sigreg`, `jepa_only` share identical encoder/predictor/decoder (390,999 trainable parameters; `DeltaDynamics` 242,647).
