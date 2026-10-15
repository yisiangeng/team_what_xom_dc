# DWC26 workflow — quickstart (challenge steps 1–4)

Reproducible pipeline for the ExxonMobil DataWorks 2026 challenge: a PyTorch proxy of the
reservoir simulator (step 1), history matching against the observed 1998–2008 field
production (step 2), the history-matched models (step 3), and a 20-year forecast with the
10 best-performing cases written into the deliverable workbook (step 4).

For the *why* behind the methods (plain language, video-script friendly), see
`MODEL_CONCEPT.md` in the workspace root.

## 1. What you need

- Python 3.10+ and the dependencies in `requirements.txt` (numpy, pandas, scikit-learn,
  scipy, torch CPU, matplotlib, openpyxl).
- The organiser dataset — the nine Excel files `01`–`09`. It is **not** committed
  (confidential, and the large files exceed GitHub's 100 MB limit): copy them into a
  `dataset_ori/` folder next to the scripts.
- `data_after_engineering/` — the cleaned data (cases with train/val/test split,
  quarterly rate and cumulative curves, grid features, observed history, MANIFEST and a
  handoff guide). It ships with the repo; only rebuild it (below) if the raw data changes.

## 2. Setup (once)

```powershell
# from the folder that contains the scripts (cod/ in the local workspace)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

macOS/Linux: `python3 -m venv .venv` then `.venv/bin/pip install -r requirements.txt`.

Optional — rebuild the engineered data from the raw Excel files (~15 min, needs
`dataset_ori/`): run `notebook1_data_eng.ipynb` from the repo root. Otherwise use the
`data_after_engineering/` folder as-is.

## 3. Fast path — reproduce the deliverable without retraining

The trained models and search results ship in `proxy_outputs/`, so the final output can be
regenerated in well under a minute:

```powershell
.\.venv\Scripts\python.exe forecast_pipeline.py --forecast
.\.venv\Scripts\python.exe forecast_pipeline.py --deliverable
```

→ writes `10_Final_deliverables.xlsx` **one level above the scripts** (the workspace root).

## 4. Full workflow, step by step

Run everything from the folder that contains the scripts, with the venv interpreter.

### Step 1 — Train the production-rate proxy

```powershell
.\.venv\Scripts\python.exe proxy_model.py --train
```

- Maps the 4 uncertainty parameters (+8 optional grid features) to oil/gas/water rates at
  the 41 shared quarterly dates, for all 100 cases.
- Trains a 5-seed ensemble on cases 1–70; validation cases 71–85 and test cases 86–100
  are never fitted. Writes `proxy_outputs/final_model.pt`, `metrics.csv`, prediction
  CSVs and figures.
- `--tune --train` runs the full cross-validated hyper-parameter search first (much
  longer). Quick smoke fit: `--feature-set p4g8 --hidden 32 32 --epochs 300 --seeds 3`.
- Health check afterwards: `.\.venv\Scripts\python.exe check_checkpoint.py`.

### Step 2 — Grid-feature mapping + cumulative twin proxy

```powershell
.\.venv\Scripts\python.exe forecast_pipeline.py --check-grid
.\.venv\Scripts\python.exe forecast_pipeline.py --train-cumulative
```

- Matching and forecasting work on **cumulative** production (the observed history and
  the simulated cases report rates under different conventions).
- `--check-grid` validates the parameters → grid-features mapping used to feed the model
  for arbitrary parameter sets.
- `--train-cumulative` trains the cumulative twin proxy (same architecture and protocol;
  it only ships if it beats a ridge baseline on the held-out test split). Writes
  `proxy_outputs/cumulative_model.pt` and `cumulative_metrics.csv`.

### Step 3 — History matching (the search)

```powershell
.\.venv\Scripts\python.exe history_match.py
```

- Scores 40,000 space-filling Sobol parameter sets + 5,000 local polish points (~2 s on
  CPU) against the observed history, using the cumulative proxy.
- Writes `hm_candidates.csv` (best 5,000), `hm_matched_models.csv` (top-20 = the
  history-matched models), `hm_summary.json` and two diagnostic PNGs.
- Reference result: best misfit ≈ 0.0039, slightly better than the best of the 100
  provided cases — with a wide set of near-equally good matches (the history
  underdetermines: ~4,900 parameter sets sit within 25% of the best misfit).

### Step 4 — 20-year forecast, best-10 selection, deliverable

```powershell
.\.venv\Scripts\python.exe forecast_pipeline.py --forecast
.\.venv\Scripts\python.exe forecast_pipeline.py --deliverable
```

- `--forecast`: extends each matched case to 2028 with the decline family that won a
  backtest on all 100 known cases (Arps for oil/gas, Gompertz for water), then picks the
  10 best performers with k-means diversity (one per cluster). Writes `forecast_cum.csv`,
  `hm_top10_parameters.csv`, `family_backtest.csv`, `forecast_fan.png`. Takes ~30 s.
- `--deliverable`: fills `dataset_ori/09 Template Deliverable.xlsx` (10 sheets × 80
  quarters, Date column untouched) and writes `../10_Final_deliverables.xlsx`, then
  re-reads it to validate row counts, dates, finiteness and monotonicity.
- Both stages can be combined: `--forecast --deliverable`.

## 5. Verify

```powershell
.\.venv\Scripts\python.exe -m unittest test_proxy_model test_forecast_pipeline
```

17 tests (~7 s): data ordering, grid mapping, misfit math, curve families, case
selection, deliverable round-trip and a checkpoint smoke test.

## 6. What each step writes

| Step | Command | Main outputs |
|---|---|---|
| 1 | `proxy_model.py --train` | `proxy_outputs/final_model.pt`, `metrics.csv`, prediction CSVs, figures |
| 2 | `forecast_pipeline.py --check-grid --train-cumulative` | mapping report; `cumulative_model.pt`, `cumulative_metrics.csv` |
| 3 | `history_match.py` | `hm_candidates.csv`, `hm_matched_models.csv`, `hm_summary.json`, diagnostic PNGs |
| 4 | `forecast_pipeline.py --forecast --deliverable` | `forecast_cum.csv`, `hm_top10_parameters.csv`, `forecast_fan.png`, `10_Final_deliverables.xlsx` |

## 7. Notes and troubleshooting

- Always run with the project venv interpreter — the system Python may not have torch.
- The scripts resolve `data_after_engineering/`, `dataset_ori/` and `proxy_outputs/`
  **relative to their own folder** — keep them together (as in `cod/`). The deliverable
  is written one level above the scripts.
- Running from the team repo (`our_work/proxy_model_elton/` in a fresh clone): copy the repo's
  `data_after_engineering/` next to the scripts and place the organiser
  `09 Template Deliverable.xlsx` into a `dataset_ori/` subfolder — or rebuild the data
  there with `notebook1_data_eng.ipynb`.
- `--deliverable` needs `dataset_ori/09 Template Deliverable.xlsx` (the blank template).
- Useful checks: `check_checkpoint.py` (step-1 checkpoint vs `predictions.csv`),
  `hm_summary.json` (search bounds, model hash, benchmark vs the 100 cases).
- Companion notebooks walk through the same steps with plots:
  `notebook2_sample_model1.ipynb` (step 1) and
  `notebook3_history_match_forecast.ipynb` (steps 2–4); both are generated by their
  `build_notebook*.py` scripts.
