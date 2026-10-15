# Output folder of `notebook2_proxy_model.ipynb`

The proxy model: a fast stand-in for the reservoir simulator. It maps the 4 uncertain parameters (plus 8 grid features) to the production curves of a case. This folder holds the trained models, their accuracy and the figures. The findings are in `summary_hist_mat.md`, section 1 (repo root).

**The three models:**

| Model | File | Role |
|---|---|---|
| Rate proxy | `final_model.pt` | Elton's setup reproduced exactly: an MLP on the parameters and 8 grid features (`p4g8`), 2 × 128 units, 5 seeds, trained on cases 1-70. It predicts quarterly **rates** of gas, oil and water on 41 dates. Test RMSE 32.5, against 135.8 for a ridge baseline |
| Elton's cumulative twin | `cumulative_model.pt` | The same network trained on **cumulative** curves (Elton's step 2). This was version 1 of the history-matching proxy and is kept for comparison |
| **Improved cumulative proxy (adopted, version 2)** | `cumulative_model_improved.pt` | Trained on oil and water over the 37 producing dates, with lr 3e-3, weight decay 1e-5 and **20 seeds**. Notebook 2 copies it to `hist_match_input/cumulative_model.pt`, **the proxy used by the history matching and the forecast**. Test error is oil 0.064 % and water 0.142 % of the final observed value, against 0.071 % and 0.240 % for the twin |

**Units.** Rates are per day; cumulatives are STB (oil, water) and MSCF (gas). Columns ending in `%` are percentages of the field's final observed cumulative on 2008-01-01, the units the history matching uses.

## Models (PyTorch checkpoints)

Load them with `src/proxy_lib.py`: `load_checkpoint(path)` and `predict_from_params(ckpt, theta)`, or `load_hm_proxy(...)` for the history-matching interface. Each file holds the weights of every seed, the scalers, the configuration, the number of epochs and the grid mapper. The grid mapper turns the 4 parameters into the 8 grid features, so a new parameter set needs nothing else.

## Tuning

| File | What it holds |
|---|---|
| `config_search.csv` | Elton's two-stage configuration search: 36 configurations on one fold, then the top 3 on 5 folds, with their cross-validation RMSE |
| `best_config.json` | The chosen configuration: `p4g8`, 128 × 128, dropout 0.05, lr 1e-3, weight decay 1e-4. It is the same as Elton's |
| `config_check_cumulative.csv` | A check that the chosen configuration also suits the cumulative target: the top configurations rerun on 5 folds for cumulatives, with the error also as % of final oil |

## Accuracy

| File | What it holds |
|---|---|
| `metrics_rate.csv` | Rate proxy against baselines (ridge, mean curve) by split and phase: `rmse`, `mae`, `r2`, `nrmse` |
| `metrics_cumulative.csv` | The same for Elton's cumulative twin. It must beat ridge on the test cases before it is saved (a gate) |
| `metrics_cumulative_improved.csv` | The adopted improved proxy against Elton's twin (`model` = `improved`, `elton_twin`), same columns |
| `accuracy_hm_units.csv` | **The error that matters for history matching**, by split (train, validation, test). It gives the RMS and max error per phase in % of the final observed cumulative, for the adopted proxy and (`elton_*`) for the twin. The validation row gives `sigma_proxy`, the proxy error passed to the history matching |
| `hm_score_proxy_vs_simulator.csv` | For each of the 100 cases, its history-match score computed from the simulator and from the proxy. Spearman rank correlation 1.0: the proxy ranks the cases the same way |
| `proxy_summary.json` | The key numbers in one file: configuration, epochs, rate and ridge RMSE, the cumulative gate, accuracy in HM units, `sigma_proxy`, score Spearman, grid-mapper mode |

## Figures (`figs/`)

| File | What it shows |
|---|---|
| `fig0_train_curves.png` | Training data: rate curves of the 70 train cases and their mean (the model learns each case's difference from the mean) |
| `fig1_pred_vs_truth.png` | Rate proxy against the simulator, all 41 dates (dashed = perfect) |
| `fig2_curves_validation.png`, `fig2_curves_test.png` | Rate curves of the best, median and worst validation and test cases, simulator against proxy |
| `fig4_rmse_by_date.png` | Rate proxy error by date (solid = validation, dashed = test) |
| `fig5_hm_units.png` | The adopted cumulative proxy in history-matching units, including its error by date |

## Related folders

- **`hist_match_input/`:** the hand-over to notebook 3 (the adopted model, `predictions.csv`, `simulated.csv`, `case_params.csv`, `proxy_meta.json`, `proxy_error_by_date.csv`), with its own README.
- **`archive/v1_baseline_2026-10-15/proxy_output/`:** this folder before the proxy improvement.
- **`our_work/proxy_model_elton/proxy_outputs/`:** Elton's original results, which notebook 2 reproduces.

## Rerun

Run `notebook2_proxy_model.ipynb` from the repo root. It overwrites every output file here, but not this README. Then rerun notebooks 3 and 4, because they depend on the proxy.
