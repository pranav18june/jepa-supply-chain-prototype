# Supply-chain world model: audited research prototype

Active implementation follows `../IMPLEMENTATION_PLAN.md`. **Research acceptance is partial**: the model/planner path runs, but learned planners have not beaten the native policy in the executed developmental experiments. See [CHANGE_REPORT.md](CHANGE_REPORT.md) for exact results and unresolved gates.

## Active paths

- **Track A:** `experiments/run_isomorph_prediction.py` — native-policy ISOMORPH prediction baselines.
- **Track B:** `experiments/run_controlled.py` — actual logged dispatch actions, separate EMA/SIGReg/supervised models, integer CEM, execute-observe-replan.
- **Track C:** `experiments/run_real_data_benchmark_v2.py` — chronological retail observed-sales/stockout prediction, never policy evaluation.

```sh
# From the JEPA root:
python3 prototype_new/experiments/fetch_audit_sources.py
PYTHONPATH=. python3 -m pytest prototype_new/tests -q
PYTHONPATH=. python3 prototype_new/experiments/run_controlled.py --output prototype_new/results/my_smoke
```

Dependencies: `requirements.txt`; exact tested environment: `requirements-verified.txt`. Independent upstream parity fixtures need the source-fetch command; unavailable fixtures produce explicit skips. No full ISOMORPH download or visual-model checkpoint is required.

The ISOMORPH action API is a local dispatch-target extension, not a native arbitrary-order API. It retains original physical transitions and native replenishment. Costs are explicit synthetic accounting assumptions. Small-catalog results are not full-release benchmark results. Optional ensemble dispersion is not calibrated risk.

Documentation: [inventory](docs/WORKSPACE_INVENTORY.md), [reference audit](docs/REFERENCE_AUDIT.md), [architecture](docs/ARCHITECTURE.md), [data protocol](docs/DATA_PROTOCOL.md), [commands and experiments](docs/EXPERIMENT_PROTOCOL.md), [limitations and claims](docs/LIMITATIONS_AND_CLAIMS.md).

## Legacy code and preserved results

`run_benchmark.py`, `run_ablations.py`, `run_uncertainty_gate.py`, `run_real_data_benchmark.py`, their original configs and five-template `src/env/actions.py` belong to the prior prototype. They remain for historical baseline comparisons and do not implement the new proposed planner. Legacy “JEPA” method names are not evidence of a latent JEPA training objective. Existing historical metrics/reports are preserved, not endorsed. The original v2 retail runner and README are recoverable from the pre-edit source archive.

Use new output directories. Datasets, caches, checkpoints and generated outputs are ignored by Git; the workspace initially had no root Git repository. Reproduction commands and source/data pins are documented rather than silently committing downloaded material.
