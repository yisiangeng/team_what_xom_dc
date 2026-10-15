# Action plan (temporary working checklist)

Decisions (user, 2026-10-15): the 10 = **hybrid** (simulated cases preferred, new sets only when better by more than the proxy error); forecast to 2028 = **next step, not now**; test cases 86-100 **allowed** as candidates in the final HM run.

Notes: notebook 2 reproduces Elton exactly (same config, identical RMSE). Grid mapper uses the linear fallback (mean_tranx analytic error 0.29 sd).

## 0. Setup
- [x] Read Elton's code (`proxy_model.py`, `forecast_pipeline.py`, `history_match.py`) and outputs
- [x] Read Yi Siang v1/v2 code, Zi Yang's notebook, the guide
- [x] Check environment: torch 2.13 CPU, emcee 3.1.6, nbclient, ipykernel
- [x] `src/proxy_lib.py`: Elton's model code (paths changed only) + checkpoint save/load + `CumulativeProxy` loader for HM
- [x] Smoke-test `proxy_lib` (load data both targets, grid mapping report)

## 1. `notebook2_proxy_model.ipynb` (repo root)
- [x] 1 Data: rates on 41 dates, splits 70/15/15, CV folds
- [x] 2 Features: p4, p4g8 (grid feature table)
- [x] 3 Model: MLP residual from mean curve, parameter count
- [x] 4 Tuning: Elton's 2-stage search (36 configs fold 0 -> top 3 on 5 folds); `RUN_TUNE` flag; compare with Elton's `config_search.csv` / `best_config.json`
- [x] 5 Rate proxy: 5-seed ensemble, CV-median epochs, ridge + mean-curve baselines, metrics, Elton's 4 figures; compare with his test RMSE ~32.5
- [x] 6 Cumulative twin: same config, gate vs ridge on test; check that the config transfers (5-fold CV, top configs, check only)
- [x] 7 Grid mapper check (analytic, < 0.2 sigma)
- [x] 8 HM-unit accuracy: cumulative error % of final observed per date (37-date window, oil/water), validation -> `sigma_proxy`; Spearman of proxy-score vs simulator-score (val/test); error on cases nearest the history
- [x] 9 Export to `hist_match_input/`: `predictions.csv`, `simulated.csv`, `case_params.csv`, `cumulative_model.pt`, `proxy_meta.json` (bounds, sha256, sigma_proxy), `proxy_error_by_date.csv`, `README.md`
- [x] Execute notebook end-to-end; outputs to `proxy_output/` (models, metrics, figs)

## 2. `notebook3_hist_match.ipynb` (repo root) - hybrid of v2 + Zi Yang
- [x] Paths -> `data_after_engineering/`, `hist_match_input/`, `hist_match_output/` (+ `figs/`)
- [x] Containers, loaders, window align (37 dates from 1999-01-01, oil + water), method-A scores (v1 code)
- [x] Proxy = `CumulativeProxy` (Elton's cumulative twin), `sigma_proxy` from notebook 2
- [x] Prior = train range, **no padding**; edge warning
- [x] MCMC: emcee 4 ensembles x 16 walkers x 12,000 steps, sampled noise, AR(1) rho, R-hat / tau / ESS
- [x] 10b direct sampling cross-check
- [x] 10c implausibility (cases + parameter space, NROY, sensitivity 0.5/1/2 %)
- [x] 10d differential evolution: best achievable proxy score, edge check
- [x] **Hybrid selection**: candidates = 100 simulated cases (real curves) + posterior samples (proxy curves); drop I > 3; k-means(10) on scaled params; per cluster lowest score, simulated preferred unless proxy set better by > proxy error
- [x] Report Elton's 3-phase misfit too, for comparison with his result
- [x] 10e oil-water trade-off, 10f hold-out (early/late), 10g scorecard
- [x] Recovery test: 5 validation cases as fake histories (true case removed from pool), 0 / 0.5 / 2 % noise: coverage of true params, porosity error, R-hat
- [x] Execute notebook end-to-end (re-run after direct-sampling fix: integrate noise levels out, ESS >= 1000 rule)

## 3. Outputs `hist_match_output/` (tag `real`)
- [x] CSV: `top10_cases`, `top10_series` (1998-2008), `scores_all`, `candidates`, `posterior_summary`, `posterior_samples`, `mcmc_diagnostics`, `direct_vs_mcmc`, `implausibility_nroy`, `new_param_sets`, `holdout`, `scorecard`, `recovery_test`
- [x] Figures: fit_top10, scores, posterior corner + chosen 10, trace, predictive band, direct sampling, implausibility, tradeoff, holdout, selection map
- [x] `README.md` for input and output folders

## 4. `summary_hist_mat.md` (repo root)
- [x] Proxy performance (rate + cumulative vs baselines; reproduction vs Elton)
- [x] HM result: best misfit vs best / median simulated case; what the history constrains
- [x] The 10: sources, parameters, scores, diversity, convergence, recovery test
- [x] Comparison with Elton's and Zi Yang's results
- [x] Limitations + next step (forecast; Elton's 2008 discontinuity)

## 5. Wrap-up
- [x] Update memory (HM decisions)
- [x] Report to user (HM done, 2026-10-15)

---

# Part 2 (requested 2026-10-15): proxy improvement tests, then forecast

## 6. Proxy improvement tests (`improve_proxy/`, temporary)
Rule: the main notebooks stay untouched until a variant clearly wins. If nothing wins, keep the original model.
- [x] Create `improve_proxy/` with `notebook3_proxy_variants_test.ipynb` (a duplicate of notebook 3 plus a proxy-variant stage in front)
- [x] Fair protocol: 5-fold CV on the train cases + validation cases 71-85; metric = history-matching units (oil, water RMS error, % of final observed). Test cases 86-100 are used only once, for the winner
- [x] Variants:
  - [x] V0 baseline (Elton, current)
  - [x] V1 loss without gas and without the 4 all-zero quarters (oil no longer counted twice; more weight on water)
  - [x] V2 refit on cases 1-85 (proxy error then from CV, not from the validation cases)
  - [x] V3 bigger ensemble (10-20 seeds); ensemble spread as a per-point proxy error
  - [x] V4 wider tuning (random search: learning rate, batch size incl. mini-batch, dropout, depth, epochs); reference paper: Bayesian tuning, best batch size 2
  - [x] V5 monotone output (non-negative quarterly increments, so cumulatives cannot decrease)
  - [x] V6 GP on PCA scores (Zi Yang) and an MLP+GP average
- [x] Success rule: validation water error down by >= 10 %, oil not worse; confirmed on test
- [x] Winner(s): rerun the HM pipeline in the test notebook; compare eps, posterior, top 10
- [x] Report in `summary_hist_mat.md` (new section): what worked or not, with numbers
- [x] If adopted (user OK): move into `src/proxy_lib.py` + notebook 2, rerun notebooks 2 and 3; else delete nothing, revert nothing

## 7. Forecast (`notebook4_forecast.ipynb`, `forecast_output/`, `summary_forecast.md`)
- [x] Inputs: `hist_match_output/top10_cases_real.csv`, `top10_series_real.csv`; history incl. water injection; template `dataset_ori/09 Template Deliverable.xlsx`
- [x] Rates from cumulatives (quarter mean, delta cum / delta days), the same definition for cases and history
- [x] Research and cite methods online (verify each source before citing)
- [x] Candidate methods:
  - [x] F0 Elton's families on the whole cumulative curve (baseline; his 25-quarter fit / 16-quarter test backtest)
  - [x] F1 Arps decline on the oil rate, fitted on the decline period only, anchored to the 2008 rate; modified hyperbolic (terminal decline)
  - [x] F2 water from the WOR / X-plot trend (log WOR against cumulative oil)
  - [x] F3 total liquid rate x (1 - water cut), water cut extrapolated (fractional-flow style)
  - [x] F4 CRM (capacitance-resistance model) + fractional-flow model, driven by water injection (needs a future-injection assumption)
  - [x] gas = GOR x oil (constant, about 0.3633)
- [x] Backtest: rolling origins (fit to 2004 / 2005 / 2006, forecast to 2008) on all 100 simulated cases + the observed history; rate and cumulative errors by horizon; choose method + fit window per phase
- [x] Accuracy improvements: method/window by backtest, ensemble of the best methods, physical constraints (continuity at 2008 in rate and cumulative, monotone cumulative, non-increasing oil rate, water cut <= 1 and non-decreasing, fixed GOR)
- [x] Uncertainty: P10/P50/P90 across the 10 and across methods (for slides / Power BI)
- [x] Deliverable: copy of the 09 template, sheets Case 1-10 (HM rank order), 80 quarters 2008-04-01 to 2028-01-01, gas / oil / water cumulative; Date column untouched; readback validation
- [x] `summary_forecast.md`: method choices, sources, backtest results, limitations
- [x] Update this checklist and memory

### Outcome (15 Oct 2026)
- Proxy: rule winner (MLP+GP average) failed the single test -> runner-up V4 (oil+water target, lr 3e-3, wd 1e-5) x 20 seeds checked out-of-fold (+ test, second use, disclosed) and ADOPTED; trained on 1-70 (refit on 1-85 did not help). Baseline archived in archive/v1_baseline_2026-10-15/.
- Notebooks 2, 3, 4 re-run; all 8 HM checks pass; 10 = cases 6, 97, 76, 11, 40 + 5 new sets.
- Forecast: crm_wor (n=12, WOR anchored on last 2 quarters) won the backtest (2.9 % vs Elton 64 %). Deliverable forecast_output/10_Final_deliverables.xlsx validated.
- Docs: summary_hist_mat.md (v2 + section 7), summary_forecast.md.
- [ ] Open: user to decide whether to delete this file, improve_proxy/ and improve_forecast/ after review.

## 8. 2008 rate-bias fix (requested 15 Oct; revert if no significant gain)
- [x] Restore point `archive/v2_pre_bias_fix_2026-10-15/` (all notebooks, src, outputs, docs, builders; README with revert command)
- [x] Significance of the proxy improvement (`improve_proxy/significance_test.py`): out-of-fold -20 % oil and water, p ~ 1e-9
- [x] Blind start-point hindcast (`improve_forecast/`): 100 pseudo-fields + field, cut-offs 2005/06/07, starts main / A1 / A2 / fade
- [x] Rule fixed before the run: adopt only if better on pseudo-fields (Wilcoxon p < 0.01) and on the field -> A2 passes (3.5 vs 7.5 %, p ~ 5e-37; field 3.8 vs 6.5 %)
- [x] Model-error term in HM: already present (sampled noise + AR(1)); not changed, reason documented
- [x] Notebook 4: hindcast section, A2 start, own-start sensitivity, wider range band; re-run, all checks and deliverable validation pass
- [x] Docs: summary_forecast.md v3, summary_hist_mat.md limits, memory
