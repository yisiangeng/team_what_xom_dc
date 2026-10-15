# data_after_engineering: the cleaned datasets

> **Main material: read [`data_desc_engineered.md`](data_desc_engineered.md) to understand the data.** It is the full handoff guide from data engineering: where each file comes from, every column and its unit, how the grid was split, how to open the `.npz` and `.npy` files, the rules that hold in every file, and the cautions. This README is only a short map of the folder.

> **Confidential (DWC26).** Do not publish these files or put them in the code zip.

## What this folder is

The organiser's raw Excel files (`dataset_ori/`) are very wide, mostly empty and up to 330 MB each. `notebook1_data_eng.ipynb` (repo root) converted them once into clean CSV tables and NumPy arrays, which are saved here. Every later notebook reads from this folder and never touches the raw Excel files again.

**Two words used throughout:**
- **Case:** one simulation run with one set of the 4 uncertain parameters. There are 100 cases: train 1-70, validation 71-85, test 86-100.
- **History:** the field's observed production, quarterly from 1998-01-01 to 2008-01-01 (41 dates).

## Files at a glance

| File | Tier | What it holds |
|---|---|---|
| `cases.csv` | Core | 100 cases: `case_id`, `split`, `cv_fold`, the 4 parameters (`fault_trans`, `poro_mult`, `perm_mult`, `aquifer_pv`) and scaled copies `*_s` |
| `curves_quarterly_cum.npz` | Core | Arrays: `Y` (100 × 41 × 3 cumulative curves of the cases), `y_hist` (41 × 3, the history), `case_ids`, `dates`, `t_days`, `quantities`. **Phase order: gas, oil, water** |
| `history_observed.csv` | Core | The history as a table: cumulatives, reported rates, quarter-average rates and the water injection rate |
| `forecast_dates.csv` | Core | The 80 forecast dates, 2008-04-01 to 2028-01-01 |
| `param_bounds.csv` | Core | Smallest and largest value of each parameter among the train cases |
| `09 Template Deliverable.xlsx` | Core | The organiser's forecast template, unchanged |
| `curves_quarterly.csv` | Extra | The same curves as the `.npz` in one table, with rates and the split; easiest for plots and Power BI |
| `curves_native.csv` | Extra | Every simulator time step (finer than quarterly) |
| `grid_case_features.csv` | Optional | Per-case summaries of the grid (pore volume, mean porosity, permeability, transmissibility) |
| `grid_cells_template.csv`, `grid_varying_cells.npy`, `grid_active_cell_ids.npy` | Optional | The 3D grid: the parts that never change (stored once) and the 6 columns that change by case, for the 42,512 active cells |
| `grid_column_catalog.csv`, `grid_case_checks.csv`, `MANIFEST.csv` | Reference | Documentation and checks; do not model on them |
| `cache/` | Reference | Cache of the slow grid read in notebook 1. If you delete it, the next run of notebook 1 reads the grid again, which takes about 15 minutes |

Details of every column are in `data_desc_engineered.md`, section 4.

## What the team pipeline uses

| Notebook | Reads from here |
|---|---|
| `notebook2_proxy_model.ipynb` (proxy) | `cases.csv`, `curves_quarterly.csv` (rate model), `curves_quarterly_cum.npz` (cumulative model), `history_observed.csv`, `grid_case_features.csv` (the 8 grid features of configuration `p4g8`) |
| `notebook3_hist_match.ipynb` (history matching) | The same, through `src/proxy_lib.py`, plus the hand-over in `hist_match_input/` |
| `notebook4_forecast.ipynb` (forecast) | `curves_quarterly_cum.npz`, `history_observed.csv` (including `water_inj_rate`) and `forecast_dates.csv`. It fills the template from `dataset_ori/` |

## Five things to remember

1. **Key and index.** `case_id` (1-100) is the key in every file; the array index is `case_id - 1`.
2. **Time.** `t_days` counts days since 1998-01-01; history and cases end on 2008-01-01 (day 3,652). No simulated data exists after that date.
3. **Compare cumulatives, not reported rates.** The cases and the history use different rate conventions. If you need a rate, use `*_rate_avg`.
4. **Keep the test cases (86-100) locked** until the final score; `cv_fold = -1` marks them.
5. **Do not save these CSVs from Excel**: it changes numbers and dates. If you open one, click "Don't Convert".

## Rebuild

Run `notebook1_data_eng.ipynb` from the repo root with the raw files in `dataset_ori/`. It rewrites every file here. This README and `data_desc_engineered.md` are not generated, so update them by hand if the outputs change.
