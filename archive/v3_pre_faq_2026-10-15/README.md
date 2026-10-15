# Version v3 (15 October 2026): restore point before the FAQ alignment

This folder is a frozen copy of the full pipeline (proxy, history matching, forecast) **before** the changes made for the organiser's FAQ (`faq.md`). Keep this folder until the final submission. Do not edit files in here.

**What changed after this point (version 4):**
- **The 10 are chosen by history-match quality first (FAQ 9).** In this version, they were one per k-means cluster, so diversity came first.
- **Added physical checks (FAQ 5, 12-13):** zero quarters before production, cumulatives that never fall, and the forecast constants (tau, G, WOR slope).

See `summary_hist_mat.md`, section 8, in the repo root.

## What it contains

| Path here | Original location | What |
|---|---|---|
| `notebook2_proxy_model.ipynb`, `notebook3_hist_match.ipynb`, `notebook4_forecast.ipynb` | repo root | Executed notebooks, with outputs |
| `src/proxy_lib.py`, `src/forecast_lib.py` | `src/` | Model and forecast code (not changed in version 4) |
| `proxy_output/`, `hist_match_input/`, `hist_match_output/`, `forecast_output/` | repo root | All outputs, including the version 3 deliverable `forecast_output/10_Final_deliverables.xlsx` |
| `summary_hist_mat.md`, `summary_forecast.md`, `methodology_explained.md`, `action_plan.md`; `README_root.md` (the root `README.md`) | repo root | Write-ups of this version |
| `builders/` | session scratchpad | Scripts that generate notebooks 2-4 (`build_nb2.py`, `build_nb3.py`, `build_nb4.py`) and execute them (`run_nb.py`) |

## Settings of this version

- **Proxy (v2).** Elton's MLP (p4g8, 2 x 128) on oil + water, lr 3e-3, weight decay 1e-5, 20 seeds, cases 1-70. eps = 0.111 %.
- **History matching.** Oil + water cumulative, 37 dates; prior = train range; emcee 4 x 16 x 12,000. **Choice of 10: k-means into 10 clusters, one per cluster; a simulated case wins its cluster unless a new set beats it by more than eps.**
- **Forecast.** `crm_wor` (12-quarter window, WOR anchored on the last quarters); injection held at 9,327 STB/d; start from the observed 2008 values (A2).

## Key results of this version

| Item | Value |
|---|---|
| The 10 (sheet order) | case 6, 5 new sets, cases 97, 76, 11, 40 |
| Scores of the 10 | 0.494-0.537 %; rank by score among all 4,100 candidates: 3 to 2,393 |
| Automated checks, notebook 3 | 8 of 8 pass |
| 2028 oil, the 10 (P10 / P50 / P90) | 58.48 / 58.70 / 59.00 MMSTB |
| 2028 water, the 10 (P10 / P50 / P90) | 85.17 / 85.30 / 85.39 MMSTB |
| Wider range 2028 | oil 54.16-61.71, water 82.46-90.01 MMSTB |

## How to revert

From the repo root (PowerShell), copy this version back over the current files:

```powershell
$a = "archive\v3_pre_faq_2026-10-15"
Copy-Item "$a\notebook2_proxy_model.ipynb","$a\notebook3_hist_match.ipynb","$a\notebook4_forecast.ipynb" . -Force
Copy-Item "$a\summary_hist_mat.md","$a\summary_forecast.md","$a\methodology_explained.md","$a\action_plan.md" . -Force
Copy-Item "$a\README_root.md" README.md -Force
Copy-Item "$a\src\proxy_lib.py","$a\src\forecast_lib.py" src\ -Force
foreach ($d in "proxy_output","hist_match_input","hist_match_output","forecast_output") { Remove-Item -Recurse -Force $d; Copy-Item -Recurse "$a\$d" . }
```

The output folders' own `README.md` files are restored with the folders.

Re-running is not needed: the executed notebooks and all outputs are included.
