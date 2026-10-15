# Team What: ExxonMobil DataWorks Challenge 2026

**Topic:** machine learning-assisted history matching and production forecasting.

**Goal:** build a fast ML model of the reservoir simulator (a *proxy*), match it to the field's observed production history (1998-2008), and forecast production to 2028 for the 10 best-matching cases.

> **Confidential.** The data comes from the DWC26 dataset. Do not publish any of it, or anything derived from it, outside the team.

## Notes for Team What

1. **Before working, copy the organiser's original datasets (the nine Excel files `01` to `09`) into `dataset_ori/`.** They are git-ignored, so every member adds them locally. Notebook 4 needs `dataset_ori/09 Template Deliverable.xlsx` to write the deliverable.
2. **You do not need to rerun anything to read the results.** Every notebook is saved executed, and all outputs are in the folders below.
3. **To rerun, go in order: notebook 2 → 3 → 4.** Each one reads the previous one's output. Notebook 1 is needed only if the raw data changes. Its first run takes about 15 minutes, because it reads the 100 grid sheets.
4. **Environment.** Python 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, torch (CPU) and emcee. Run the notebooks from the repo root.

## The pipeline

```
 dataset_ori/ (organiser's Excel)
        |  notebook1_data_eng            -> data_after_engineering/   cleaned data
        |  notebook1.1_eda_commented     (exploration, no outputs used later)
        v
 notebook2_proxy_model                   -> proxy_output/              proxy models, accuracy
        |                                -> hist_match_input/          hand-over to notebook 3
        v
 notebook3_hist_match                    -> hist_match_output/         posterior, the 10 cases
        v
 notebook4_forecast                      -> forecast_output/           forecast to 2028
                                                                       10_Final_deliverables.xlsx
```

## Results in brief

- **Proxy.** It is based on Elton's MLP (4 parameters + 8 grid features) and was improved: trained on oil and water only, tuned, and with 20 seeds.
  - Test error is 0.064 % of the final observed oil and 0.142 % of water.
  - That is 20-40 % lower than the original (significant, p ~ 1e-9).
- **History matching** uses Bayesian MCMC on the proxy and a hybrid choice of 10. All 8 checks pass.
  - The history pins down porosity well (multiplier 1.34-1.38).
  - **The 10:** simulated cases 6, 97, 76, 11 and 40, plus 5 new parameter sets.
- **Forecast** uses a capacitance-resistance model plus the water-oil-ratio trend, chosen by a backtest (2.9 % error). Each forecast starts from the observed 2008 values, chosen by a blind hindcast.
  - **Oil in 2028:** 58.7 MMSTB (38.4 in 2008).
  - **Wider range** (start-point and model uncertainty): 54.2-61.7 MMSTB.
- **Main limitation.** The simulator family cannot reproduce the field's late water rise. See the summaries.

The full write-ups are **`summary_hist_mat.md`** (proxy and history matching) and **`summary_forecast.md`** (forecast).

**Versions.** Each improvement was kept only if it helped significantly. The version before each change is saved in `archive/`.

| Version | Change | Evidence | Restore point |
|---|---|---|---|
| v1 | Elton's proxy as is, plus the history match | - | `archive/v1_baseline_2026-10-15/` |
| v2 | Improved proxy (water error -41 % on the test cases), plus the forecast | `improve_proxy/` | `archive/v2_pre_bias_fix_2026-10-15/` |
| **v3 (current)** | The forecast starts from the observed 2008 values | `improve_forecast/` | - |

## Files and folders

**Notebooks and write-ups (repo root)**

| Path | What it is |
|---|---|
| `notebook1_data_eng.ipynb` | Data engineering: reads `dataset_ori/`, writes `data_after_engineering/` |
| `notebook1.1_eda_commented.ipynb` | Exploratory data analysis, annotated: data checks, parameter coverage, curves against the history, parameter effects, PCA, which parameters the history can identify |
| `notebook2_proxy_model.ipynb` | The proxy: reproduces Elton's setup exactly, then trains the improved proxy and exports it for history matching |
| `notebook3_hist_match.ipynb` | History matching: MCMC posterior, cross-checks, the choice of the 10 cases, a recovery test |
| `notebook4_forecast.ipynb` | Forecast to 2028: backtest of the methods, start-point hindcast, uncertainty, the deliverable |
| `summary_hist_mat.md` | Findings of the proxy and the history matching, including the proxy improvement tests (section 7) |
| `summary_forecast.md` | Forecast methods, evidence, results, limitations and sources |

**Data, code and outputs**

| Path | What it is |
|---|---|
| `dataset_ori/` | The organiser's original Excel files. Empty in the repo; add them locally |
| `data_after_engineering/` | Cleaned CSV and NumPy files. **Read `data_desc_engineered.md` inside it to understand the data** |
| `src/` | Shared code: `proxy_lib.py` (proxy models), `forecast_lib.py` (forecast methods), `grid_loader.py` (grid reader for notebook 1) |
| `proxy_output/` | Output of notebook 2: models, metrics, figures |
| `hist_match_input/` | Hand-over from notebook 2 to notebook 3: the adopted proxy, its error, the simulated curves |
| `hist_match_output/` | Output of notebook 3: the 10 cases, the posterior, the checks, figures |
| `forecast_output/` | Output of notebook 4, including **`10_Final_deliverables.xlsx`** (the submission) and the tables for Power BI |

Each of these folders has its own `README.md`.

**Background and team work**

| Path | What it is |
|---|---|
| `instruction/` | The competition brief: `instruction1.md` (files and submission requirements) and `instruction2_briefing_slides.md` (briefing slides) |
| `reference/` | The organisers' reference paper (Zhang et al. 2023, SPE Journal: Bayesian history matching with an LSTM proxy), as a PDF and as a text copy |
| `our_work/` | Each member's original work: Elton's and Si Jie's proxy models, and Yi Siang's and Zi Yang's history-matching notebooks. See its README |
| `archive/` | Frozen restore points of earlier versions (v1, v2), each with a revert command. See its README |

**Improvement studies (the evidence behind the current version)**

| Path | What it is |
|---|---|
| `improve_proxy/` | **Proxy improvement.** Eight proxy variants were compared with cross-validation, plus a follow-up check and a significance test. The tuned oil+water network with 20 seeds won: its out-of-fold error is about 20 % lower (p ~ 1e-9). Write-up: `summary_hist_mat.md`, section 7 |
| `improve_forecast/` | **Forecast start point.** A blind hindcast on 100 pseudo-fields and the real field shows that starting from the observed 2008 values halves the forecast error (p ~ 5e-37). Write-up: `summary_forecast.md`, section 5 |

Each has a README. Neither is needed to run the pipeline: the adopted changes are built into notebooks 2 and 4.

**Temporary (delete after the submission)**

| Path | What it is |
|---|---|
| `action_plan.md` | Working checklist |

`.gitignore` ignores the files in `dataset_ori/` but keeps the folder.

## Deadlines

- **16 Oct 2026:** code.
- **20 Oct 2026:** documents, Power BI and video.
