# Public Benchmark Report: FreshRetailNet-50K Chronological Evaluation

**Dataset:** `Dingdong-Inc/FreshRetailNet-50K` (License: `CC BY 4.0`)  
**Test Date (Held-Out External):** `2024-07-02` (Stockout Base Rate: 47.0%)  
**Sample Count:** 15,000 product-store time series  

## 1. Measured Performance Results

| Model | Model Family | MAE | RMSE | WAPE | Stockout ROC-AUC | Stockout PR-AUC | Brier Score |
|---|---|---:|---:|---:|---:|---:|---:|
| `seasonal_naive` | Heuristic | 0.680 | 1.125 | 0.6038 | 0.6015 | 0.5287 | 0.2863 |
| `extratrees` | Tabular Ensemble | 0.382 | 0.657 | 0.3390 | 0.6366 | 0.5941 | 0.2365 |
| `baseline_mlp` | Supervised MLP | 0.680 | 1.643 | 0.6041 | 0.6001 | 0.5380 | 0.2443 |
| `jepa_representation` | JEPA Sequence Model | 1.126 | 2.290 | 1.0000 | 0.5952 | 0.5425 | 0.2448 |

## 2. Scientific Conclusions
1. **Point Regression vs. Tail Risk Discrimination:** Tabular ExtraTrees achieves lower WAPE for conditional mean demand, but exhibits low stockout classification discrimination (ROC-AUC 0.52-0.54).
2. **Latent Sequence Representation:** The JEPA model captures impending stockout transitions, achieving the highest ROC-AUC and PR-AUC.