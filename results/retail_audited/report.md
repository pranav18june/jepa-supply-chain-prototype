# Track C predictive results

Frozen chronological split of a pinned official-train subset. Test dates June 13–25 excluded from fitting and epoch selection; official eval excluded. Convenience subset, not full benchmark.

Observed-sales prediction only; no latent-demand recovery or policy claim.

| Method | MAE | Stockout AUROC |
|---|---:|---:|
| last_value | 0.488741 | 0.6254012856384293 |
| extratrees | 0.434827 | 0.6683633933856876 |
| ema_latent_supervised | 0.432846 | 0.6626598291888776 |
