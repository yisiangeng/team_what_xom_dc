# our_work: the team's individual work

This folder holds each member's original work: the two proxy models and the history-matching drafts. The final pipeline at the repo root builds on it:

| Final step (repo root) | Built on |
|---|---|
| `notebook2_proxy_model.ipynb` | Elton's proxy (`proxy_model_elton/`), chosen as the team's proxy |
| `notebook3_hist_match.ipynb` | Yi Siang's history-matching v2, extended with a hybrid choice of the 10. Zi Yang's 10 and Elton's 10 are scored there for comparison |
| `notebook4_forecast.ipynb` | Built new; Elton's forecast (`forecast_pipeline.py`) is its baseline, but it failed the backtest |

The files here are kept as records and are **not** needed to run the final pipeline. Paths written inside them refer to where they were first run (for example `solution/...` or `cod/`), so some will not run from this folder without adjusting paths. The data is confidential (DWC26): do not share outside the team.

## Proxy models

| Folder | Owner | What it is |
|---|---|---|
| `proxy_model_elton/` | Elton | **The chosen proxy.** A PyTorch MLP that maps the 4 uncertain parameters plus 8 grid features to the quarterly production curves. It is a residual from the mean curve, an ensemble of 5 seeds, trained on cases 1-70. Elton's scripts cover challenge steps 1-4. The root notebook 2 reproduces his setup exactly; version 2 of the proxy is an improved retraining of the same network. See the file table below. |
| `proxy_model_sijie/` | Si Jie | **Reference only.** A packaged repository (`src/dwc26/`, `configs/`, `tests/`) built from the **raw** organiser data rather than the engineered data. It compares several proxies and freezes an LSTM as primary and a GRU as secondary; the LSTM's test error is 44 STB/d on oil and 34 STB/d on water. Its history matching and forecast (Tasks 4-5) were not started. Start with `summary.md`, a one-file review. |

**Files in `proxy_model_elton/`:**

| File | What |
|---|---|
| `README.md` | Quickstart for the 4 steps, with setup and troubleshooting |
| `MODEL_CONCEPT.md` | The reasoning behind the methods, in plain language |
| `proxy_model.py` | Step 1: the rate proxy, its config search and training. Writes `proxy_outputs/final_model.pt` |
| `forecast_pipeline.py` | Grid-feature mapping, the cumulative twin of the proxy, and step 4 (curve-family forecast and deliverable) |
| `history_match.py` | Steps 2-3: history matching with the proxy |
| `check_checkpoint.py` | Checks the saved model against its predictions |
| `test_proxy_model.py`, `test_forecast_pipeline.py` | Unit tests |
| `requirements.txt` | Python dependencies |
| `proxy_outputs/` | His results:<ul><li>models: `final_model.pt` (rates), `cumulative_model.pt`</li><li>tuning: `best_config.json`, `config_search.csv`</li><li>metrics and predictions</li><li>history-match outputs (`hm_*`; `hm_top10_parameters.csv` holds his 10, scored for comparison in notebook 3)</li><li>forecast (`forecast_cum.csv`, `family_backtest.csv`)</li><li>figures</li></ul>Notebook 2 compares against `best_config.json` and `config_search.csv` |
| `10_Final_deliverables.xlsx` | His deliverable, superseded by `forecast_output/10_Final_deliverables.xlsx` at the root |

**Main parts of `proxy_model_sijie/`:**

| File or folder | What |
|---|---|
| `summary.md` | Review document: what was built, results, deviations from the plan, limitations |
| `README.md` | Setup and commands |
| `PRD.md`, `CLAUDE.md` | Requirements and working rules |
| `docs/` | `decisions.md` (decision log), `references.md` (literature), `specs/` (task 0-3 specifications) |
| `src/dwc26/` | The package: `data/`, `models/`, `evaluation/`, `hm/`, `forecast/`, `cli.py` |
| `configs/`, `tests/` | Configuration (YAML) and tests |

## History matching

| File | Owner | What it is |
|---|---|---|
| `history_matching_guide.md` | Team note | Explains the three pipelines below, each in plain words and in technical terms. Read this first |
| `history_matching_v1_yisiang.ipynb` | Yi Siang | Draft pipeline v1, tested on 25 simulated cases. It has two methods: [A] rank the given cases by mismatch; [B] Bayesian MCMC over the 4 parameters with the proxy (emcee), as in the organisers' reference paper. It then chooses 10, checks them and writes the template |
| `history_matching_v2_yisiang.ipynb` | Yi Siang | v1 plus sections 10a-10g: direct sampling, implausibility, a parameter search, the oil-water trade-off, a hold-out check and a scorecard. **The base of the root notebook 3** |
| `history_matching_readme_yisiang.md` | Yi Siang | The input-folder README for v1/v2: what the proxy team had to hand over (`predictions.csv`, `case_params.csv`, `history.csv`) and the rules. The root `hist_match_input/` follows this contract |
| `history_matching_ziyang.ipynb` | Zi Yang | A full pipeline in 4 sections: (1) a GP-on-PCA proxy; (2) differential evolution to find the best parameters; (3) MCMC posterior sampling; (4) the choice of 10 diverse sets. His GP was tested as a proxy variant (`improve_proxy/`), and his 10 are scored for comparison in notebook 3 |

## Where the final results are

- **Write-ups (repo root):** `summary_hist_mat.md` and `summary_forecast.md`.
- **Outputs:** `proxy_output/`, `hist_match_output/` and `forecast_output/`, each with a README.
- **Deliverable:** `forecast_output/10_Final_deliverables.xlsx`.
