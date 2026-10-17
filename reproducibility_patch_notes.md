# Reproducibility patch notes

This patch is limited to repeatable experiment execution. It does **not** change the model architecture, losses, action space, simulator dynamics, or benchmark definitions.

## Changes

1. `src/env/simulator.py`: replaced Python's built-in `hash(regime.value)` in the random-seed construction with `zlib.crc32(regime.value.encode("utf-8"))`. Python salts `hash()` across processes by default, so the same nominal simulation seed previously generated different operational histories in different processes.
2. `src/evaluation/benchmark.py`: reset NumPy and PyTorch RNGs at the start of each split using the split seed. This makes model initialisation, minibatch permutations, and bootstrap interval sampling repeatable for a given CPU environment.

## Verification

- Unit tests: 12 passed.
- Benchmark command: `PYTHONPATH=. python3 prototype/experiments/run_benchmark.py --splits 3 --epochs 20 --output-dir prototype/results_seedfixed`
- Repeated the same command in a separate output directory; `summary` and `raw_splits` in the two result JSON files were exactly equal. Wall-clock runtime differs and is intentionally excluded from this comparison.

## Important scope note

The seed patch makes a given current benchmark configuration repeatable; it does not establish that the current benchmark is a good evaluation protocol. Known issues are documented in the diagnostic report, including the public-data train/test setup, the mismatch between the advertised CVaR-95 target and code, unused validation worlds, and the absence of a JEPA latent-prediction training objective in the current supervised cost-prediction model.
