# Baseline version v1 (15 October 2026): restore point

This folder is a frozen copy of the proxy + history-matching pipeline **before** the proxy-improvement tests (`nb3_imp_testing/`). Keep it until the final submission. Do not edit files in here.

## What it contains

| Path here | Original location | What |
|---|---|---|
| `notebook2_proxy_model.ipynb` | repo root | Proxy notebook, executed, with outputs |
| `notebook3_hist_match.ipynb` | repo root | History-matching notebook, executed, with outputs |
| `src/proxy_lib.py` | `src/` | Model code (Elton's model) |
| `proxy_output/` | repo root | Rate and cumulative models (`final_model.pt`, `cumulative_model.pt`), metrics, figures |
| `hist_match_input/` | repo root | Hand-over from notebook 2 to notebook 3 |
| `hist_match_output/` | repo root | The chosen 10, posterior, checks, figures |
| `hist_mat_summary.md` | repo root | Findings of this version |
| `builders/` | session scratchpad | Scripts that generate notebooks 2 and 3 (`build_nb2.py`, `build_nb3.py`) and execute them (`run_nb.py`) |

## Settings of this version

- **Proxy.** Elton's MLP, configuration `p4g8` (4 parameters + 8 grid features), 2 x 128 units, SiLU, dropout 0.05, Adam lr 1e-3, weight decay 1e-4, full batch, no log transform, 5-seed ensemble trained on cases 1-70.
  - Rate model: 2,255 epochs. Cumulative model: 743 epochs (CV median).
  - Grid mapper: linear fallback.
  - Target: all 41 dates x 3 phases (oil, gas, water).
- **Model hash.** `cumulative_model.pt` sha256 starts `d8fc2890d7ea`.
- **Proxy error passed to the history matching** (`sigma_proxy`, validation): oil 0.0753 %, water 0.1970 % of the final observed cumulative. In score units, eps = 0.149 %.
- **History matching.**
  - Matching: oil + water cumulative, 37 dates from 1999-01-01; prior = train range.
  - Sampler: emcee, 4 x 16 walkers x 12,000 steps.
  - Implausibility: impl_sd 1 %, cut-off 3.
  - Choice of 10: 4,000 posterior candidates + 100 cases, then k-means (10), simulated cases preferred within eps.

## Key results of this version

| Item | Value |
|---|---|
| Rate proxy test RMSE (pooled) | 32.5 (ridge 135.8) |
| Cumulative proxy test RMS error, oil / water | 0.071 % / 0.240 % of final |
| Best simulated case | case 6, score 0.494 % |
| MCMC | max R-hat 1.020, ESS ~4,037 |
| Posterior porosity multiplier (5 / 50 / 95 %) | 1.338 / 1.357 / 1.372 |

**The chosen 10:**

| Rank | Source | Case | Score | Fault | Poro | Perm | Aquifer |
|---|---|---|---|---|---|---|---|
| 1 | new set | - | 0.478 % | 0.0821 | 1.3735 | 3.0015 | 198.49 |
| 2 | new set | - | 0.480 % | 0.0684 | 1.3611 | 5.7534 | 189.07 |
| 3 | new set | - | 0.484 % | 0.0650 | 1.3756 | 2.3087 | 193.18 |
| 4 | new set | - | 0.485 % | 0.1027 | 1.3644 | 3.9084 | 184.75 |
| 5 | new set | - | 0.490 % | 0.1341 | 1.3696 | 2.8090 | 199.12 |
| 6 | simulated | 6 | 0.494 % | 0.0577 | 1.3584 | 2.4685 | 119.77 |
| 7 | simulated | 76 | 0.528 % | 0.1079 | 1.3833 | 1.6405 | 182.20 |
| 8 | simulated | 11 | 0.533 % | 0.1250 | 1.3538 | 7.3377 | 188.08 |
| 9 | simulated | 40 | 0.537 % | 0.1443 | 1.3543 | 4.9613 | 132.52 |
| 10 | simulated | 73 | 0.544 % | 0.0826 | 1.3538 | 1.4749 | 93.32 |

## How to revert

From the repo root (PowerShell), copy the baseline back over the current files:

```powershell
$a = "archive\v1_baseline_2026-10-15"
Copy-Item "$a\notebook2_proxy_model.ipynb","$a\notebook3_hist_match.ipynb" . -Force
Copy-Item "$a\hist_mat_summary.md" summary_hist_mat.md -Force      # renamed in the repo on 15 Oct
Copy-Item "$a\src\proxy_lib.py" src\ -Force
foreach ($d in "proxy_output","hist_match_input","hist_match_output") { Remove-Item -Recurse -Force $d; Copy-Item -Recurse "$a\$d" . }
```

Re-running is not needed: the executed notebooks and all outputs are included. To regenerate from scratch, run notebook 2 and then notebook 3.
