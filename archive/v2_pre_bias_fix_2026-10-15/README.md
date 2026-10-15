# Version v2 (15 October 2026): restore point before the 2008 bias-fix tests

This folder is a frozen copy of the full pipeline (proxy, history matching, forecast) **before** the tests that try to correct the 2008 rate bias. In 2008 the matches' oil rate is 12-15 % high and their water rate about 10 % low; see `hist_mat_summary.md`, "Main limits". Keep this folder until the final submission. Do not edit files in here.

The rule agreed with the team: if the bias-fix tests do not improve the forecast significantly, revert to this version.

## What it contains

| Path here | Original location | What |
|---|---|---|
| `notebook2_proxy_model.ipynb`, `notebook3_hist_match.ipynb`, `notebook4_forecast.ipynb` | repo root | Executed notebooks, with outputs |
| `src/proxy_lib.py`, `src/forecast_lib.py` | `src/` | Model and forecast code |
| `proxy_output/`, `hist_match_input/`, `hist_match_output/`, `forecast_output/` | repo root | All outputs, including the deliverable `forecast_output/10_Final_deliverables.xlsx` |
| `hist_mat_summary.md`, `forecast_summary.md`, `action_plan.md` | repo root | Write-ups of this version |
| `builders/` | session scratchpad | Scripts that generate notebooks 2-4 (`build_nb2.py`, `build_nb3.py`, `build_nb4.py`) and execute them (`run_nb.py`) |

## Settings of this version

- **Proxy (v2, adopted 15 Oct).** Elton's MLP (p4g8, 2 x 128, dropout 0.05) trained on oil + water over the 37 producing dates, lr 3e-3, weight decay 1e-5, 20 seeds, 1,145 epochs, cases 1-70.
  - sigma_proxy (validation): oil 0.074 %, water 0.138 %. eps = 0.111 %.
- **History matching.** Oil + water cumulative, 37 dates; prior = train range; emcee 4 x 16 x 12,000; hybrid choice of 10 (k-means, simulated cases preferred within eps). No model-error term.
- **Forecast.**
  - Method: `crm_wor`, a CRM liquid model driven by injection, with a 12-quarter fit window. Water is split off with the ln(WOR) vs Np trend, anchored on the last quarters. Gas = GOR x oil.
  - Injection after 2008 is held constant at 9,326.61 STB/d.
  - **Each case starts from its own 2008 values** (team decision 2).

## Key results of this version

| Item | Value |
|---|---|
| Best simulated case | case 6, score 0.494 % |
| MCMC | all 8 checks pass; max R-hat 1.024 |
| Posterior porosity multiplier (5 / 50 / 95 %) | 1.342 / 1.366 / 1.383 |
| Forecast backtest (crm_wor), median error | 2.94 % on the 100 cases, 3.95 % on the field |
| 2028 oil, the 10 cases (P10 / mean / P90) | 60.70 / 61.15 / 61.69 MMSTB |
| 2028 water, the 10 cases (P10 / P50 / P90) | 82.48 / 82.92 / 83.08 MMSTB |
| 2028, alternative A2 (observed 2008 start) | oil 58.74, water 85.29 MMSTB |
| 2028, field reference (no model) | oil 54.16, water 90.01 MMSTB |

**The chosen 10 (deliverable sheet order):**

| Rank | Source | Case | Score | Fault | Poro | Perm | Aquifer |
|---|---|---|---|---|---|---|---|
| 1 | simulated | 6 | 0.494 % | 0.0577 | 1.3584 | 2.4685 | 119.77 |
| 2 | new set | - | 0.495 % | 0.0543 | 1.3643 | 3.1946 | 164.84 |
| 3 | new set | - | 0.497 % | 0.0859 | 1.3701 | 2.7941 | 180.67 |
| 4 | new set | - | 0.499 % | 0.1026 | 1.3680 | 2.7769 | 160.12 |
| 5 | new set | - | 0.503 % | 0.1384 | 1.3645 | 3.3324 | 178.09 |
| 6 | new set | - | 0.510 % | 0.0895 | 1.3464 | 4.4192 | 121.07 |
| 7 | simulated | 97 | 0.522 % | 0.0519 | 1.3462 | 5.4085 | 153.89 |
| 8 | simulated | 76 | 0.528 % | 0.1079 | 1.3833 | 1.6405 | 182.20 |
| 9 | simulated | 11 | 0.533 % | 0.1250 | 1.3538 | 7.3377 | 188.08 |
| 10 | simulated | 40 | 0.537 % | 0.1443 | 1.3543 | 4.9613 | 132.52 |

## How to revert

From the repo root (PowerShell), copy this version back over the current files:

```powershell
$a = "archive\v2_pre_bias_fix_2026-10-15"
Copy-Item "$a\notebook2_proxy_model.ipynb","$a\notebook3_hist_match.ipynb","$a\notebook4_forecast.ipynb","$a\action_plan.md" . -Force
Copy-Item "$a\hist_mat_summary.md" summary_hist_mat.md -Force      # renamed in the repo on 15 Oct
Copy-Item "$a\forecast_summary.md" summary_forecast.md -Force
Copy-Item "$a\src\proxy_lib.py","$a\src\forecast_lib.py" src\ -Force
foreach ($d in "proxy_output","hist_match_input","hist_match_output","forecast_output") { Remove-Item -Recurse -Force $d; Copy-Item -Recurse "$a\$d" . }
```

Re-running is not needed: the executed notebooks and all outputs are included.
