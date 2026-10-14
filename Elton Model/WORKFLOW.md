# Model Build Workflow — DWC26 Proxy Model

A brief walkthrough of how the production-rate proxy was built:
**data cleaning → feature selection → model selection → model comparison**.

For the plain-language "why" see `MODEL_CONCEPT.md`; for commands see `cod/README.md`.

```
Raw Excel (100 simulator cases, files up to ~330 MB)
   |  data engineering (notebook1_data_eng.ipynb)
   v
Clean dataset (cases.csv, curves, grid features, observed history)
   |  feature selection: p4 (4 params) vs p4g8 (params + 8 grid features)
   v
Residual MLP ensemble (5 seeds)  <--  two-stage CV search over 36 configs
   |  model comparison vs baselines (mean curve, ridge)
   v
Cumulative twin proxy -> history matching -> 20-year forecast -> deliverable
```

## 1. Data cleaning

- The organiser Excel files (very wide tables, mostly empty cells, `-999.25` placeholders) were converted once into clean CSVs / NumPy arrays (`cod/data_after_engineering/`).
- **Cases and split.** One row per case in `cases.csv`: the 4 uncertainty parameters, the split (train 1–70, validation 71–85, test 86–100) and a 5-fold `cv_fold` index for the train cases. The test split stays locked until the final score.
- **Curves and time axis.** The simulator curves were reshaped to long format and aligned to the 41 quarterly dates shared by all cases and the observed field history (1998–2008). Both reported rates and cumulative values are stored; comparisons use cumulative because history and cases follow different rate conventions (`*_rate_avg` columns are provided for rates).
- **Grid cleaning.** Each grid sheet has 90,365 cells × 29 columns. A cell counts as active when `porv > 0` → 42,512 active cells. 23 columns are identical in all 100 cases → stored once as a template; the 6 columns that vary (`porv, poro, permx, permy, tranx, trany`) are kept per case.
- **Scaling bounds** (`param_bounds.csv`) are computed from train cases only; missing values are stored as empty cells (no `-999.25`).
- **Checks.** Active-cell mask identical in every case; curves finite and monotone; the few parameter values slightly outside the train range (cases 75, 82, 87, 98) are kept and not clipped.

## 2. Feature selection

Two input sets were compared in the tuning search (36 configs screened on fold 0; the top 3 then run
through the full 5-fold CV):

| Feature set | Inputs | Fold-0 screen RMSE (rate units) | Full 5-fold CV |
|---|---|---|---|
| `p4` | fault transmissibility, porosity multiplier, permeability multiplier, aquifer pore volume (min/max scaled) | 55.0 | not run (outside the top 3) |
| `p4g8` (chosen) | `p4` + 8 grid summary features (z-scored) | **30.9** | **36.3** (winner) |

- The 8 grid features: total pore volume, aquifer / reservoir pore volume, mean porosity, mean permeability (global and in-zone), mean transmissibility (x and y).
- Rationale: every case's grid is a scaled copy of the base case (verified by direct comparison), so 8 summaries carry the geological difference that the 4 knobs create. The full 42,512-cell grid — and the template columns that never change — adds nothing for this model.
- `p4g8` beat `p4` on the screen at equal settings (30.9 vs 55.0, both 128×128, weight decay 1e-4, no log transform); the winner's full 5-fold CV score is 36.3.

## 3. Model selected

- **Target formulation — residual learning.** The network predicts the *difference* between a case's curve and the mean train curve (`PhaseScaler`), so it only learns case-to-case variation. Inputs are min/max (params) and z-score (grid) scaled; target residuals are standardized (a signed log1p variant was tested and rejected by the search).
- **Architecture.** A small MLP — 2 hidden layers, SiLU activations, dropout 0.05 — that outputs the full 41-date curve at once (3 phases × 41 dates = 123 numbers). At 128×128 it has ~34k weights: deliberately small, because there are only 70 training cases.
- **Training.** Adam (lr 1e-3), cosine LR decay, early stopping (patience 300, max 3000 epochs).
- **Hyperparameter search (two-stage, 36 configs).** Grid over feature set × log × hidden {32², 64², 128²} × weight decay {1e-4, 1e-3, 1e-2}: stage 1 screens all 36 configs on one fold; stage 2 re-runs the top 3 on all 5 folds. Winner (→ `best_config.json`): `p4g8`, hidden 128×128, weight decay 1e-4, no log — **CV RMSE 36.3** (fold spread 29–49).
- **Final fit.** A 5-seed ensemble trained on all 70 train cases, with a fixed epoch budget (median of the CV epochs used); predictions are the average of the 5 models → `final_model.pt`.
- **Protocol.** All scalers and models are fitted inside the train split only; validation (71–85) is reported, test (86–100) is opened once, at the end.

## 4. Model comparison

All candidates receive identical inputs and the same data protocol:

| Model | What it is | Test pooled RMSE (rate units) |
|---|---|---|
| **DNN ensemble (ours)** | tuned `p4g8` MLP, 5 seeds | **32.5** |
| Ridge | linear regression on the same 12 inputs | 135.8 |
| Mean curve | train average curve (uses no case information) | 763.5 |

- The ensemble wins on every split and every phase: ~**4.2× better than ridge**, ~23× better than the mean curve (test pooled `nRMSE` 0.0073 vs 0.0306 vs 0.1722).
- The same architecture was re-validated as the **cumulative twin proxy** (same inputs and protocol, cumulative targets) that drives history matching and forecasting — it also beats ridge (test pooled RMSE 23,097 vs 181,002).
- Downstream sanity checks: the 45,000-point Sobol history-match search slightly beats the best of the 100 provided cases, and the decline families used for the forecast were backtested on all 100 known cases.
- Honest caveat: the hardest cases are the few just outside the training parameter range (75, 82, 87, 98; largest errors 82, 87, 98).

## Where each step lives

| Step | Artifact |
|---|---|
| Data cleaning | `cod/notebook1_data_eng.ipynb` → `cod/data_after_engineering/` |
| Features, tuning, training, metrics | `cod/proxy_model.py`; outputs in `cod/proxy_outputs/` (`config_search.csv`, `best_config.json`, `metrics.csv`, `final_model.pt`) |
| Model story with plots | `cod/notebook2_sample_model1.ipynb` |
| Cumulative twin | `cod/forecast_pipeline.py --train-cumulative` → `cumulative_model.pt`, `cumulative_metrics.csv` |
| History matching | `cod/history_match.py` → `hm_*.csv`, `hm_summary.json` |
| Forecast + deliverable | `cod/forecast_pipeline.py --forecast --deliverable` |
