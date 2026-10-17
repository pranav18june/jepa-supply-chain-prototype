# Track C predictive results

EXPLORATORY CACHE SMOKE ONLY. Official eval was already used in legacy experiments; this is not an untouched final holdout.

Observed-sales prediction only; no latent-demand recovery or policy claim.

| Method | MAE | Stockout AUROC |
|---|---:|---:|
| last_value | 0.599121 | 0.6195099470005376 |
| extratrees | 0.637346 | 0.5712727552039327 |
| ema_latent_supervised | 0.855777 | 0.6306628773331285 |
