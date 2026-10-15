# Input folder of `notebook_hist_matching.ipynb`

Put the proxy team's files here. Then run all cells of the notebook. The results appear in `solution/histmatch_output/`.

If this folder has no `predictions.csv`, the notebook runs a draft demo on 25 real simulated cases instead.

## Files

| File | Required | Columns | Notes |
|---|---|---|---|
| `predictions.csv` | yes | `case_id`, `date`, `oil_cum`, `water_cum`, and `gas_cum` (optional) | One row per case and date. Use cumulative values (STB for oil and water, MSCF for gas). Every case needs the same dates. Dates can run past 2008-01-01 (up to 2028). The history window (to 2008-01-01) is used for matching. Later dates are carried to the output. |
| `case_params.csv` | for method B | `case_id`, `fault_trans`, `poro_mult`, `perm_mult`, `aquifer_pv` | One row per case, in real units. |
| `history.csv` | no | `date`, `oil_cum`, `water_cum` (and `gas_cum`) | The observed history. If absent, `solution/output/history_observed.csv` is used. |

Date format: `YYYY-MM-DD`. One case in the file is allowed. The notebook then returns that case with its score.

## Rules

* Use the same time axis as the history (quarterly, from 1998-01-01), or any finer axis. The notebook interpolates cumulative values onto the history dates.
* Do not put test cases 86-100 here until the final run.
* The data is confidential (DWC26). Do not share it outside the team.

## Other datasets the notebook reads

The notebook reads these files from `solution/output/` (made by `notebook1_data_eng.ipynb`). You do not need to supply them.

| File | Rows | What it holds | Used for |
|---|---|---|---|
| `history_observed.csv` | 41 | The observed field history, quarterly from 1998-01-01 to 2008-01-01. Columns: `date`, `t_days`, cumulative `gas_cum`, `oil_cum`, `water_cum`, reported rates, average rates and `water_inj_rate`. | The target of the matching. Used when `history.csv` is absent. |
| `param_bounds.csv` | 4 | Minimum and maximum of each of the 4 parameters in the train cases. | Prior range for the MCMC (widened by 5 %), and the 0-1 scaling of parameters. |
| `cases.csv` | 100 | One row per simulation case: `case_id`, `split` (train, validation or test), `cv_fold`, the 4 parameters, and scaled copies (`*_s`). | Draft demo only. Gives the parameters and the split of each case. |
| `curves_quarterly_cum.npz` | 100 cases x 41 dates x 3 | Simulated cumulative gas, oil and water of every case on the quarterly history dates. | Draft demo only. This is the stand-in for `predictions.csv`. |
| `forecast_dates.csv` | 80 | The 80 quarterly forecast dates, 2008-04-01 to 2028-01-01. | Self-test of the template writer. |

## Results: datasets written to `solution/histmatch_output/`

Every file name ends with a tag. `_draft` means the demo run on 25 real simulated cases. `_real` means a run on the proxy team's `predictions.csv`.

In these CSV files, mismatch scores are **fractions** of the final observed cumulative. Example: `0.005` means 0.5 %.

| File | Rows | What it holds |
|---|---|---|
| `top10_cases_<tag>.csv` | 10 | **The answer.** The 10 chosen cases, ranked 1 to 10. Columns: the mismatch scores (`nrmse_oil`, `nrmse_water`, `score`, `max_rel_err`), the signed error on the last date (`err_final_oil`, `err_final_water`), `loglik` and `weight` (method A), `post_dist` (method B), the three ranks, `meets_threshold` (`score` at or below the 1 % provisional level), and the 4 parameters of the case. |
| `top10_series_<tag>.csv` | 10 cases x all dates | The full cumulative time series of the 10 chosen cases. Columns: `rank`, `case_id`, `date`, `gas_cum` [MSCF], `oil_cum` [STB], `water_cum` [STB], `gas_is_derived`. When `gas_is_derived` is `True`, gas is 0.3633 x oil. This file feeds template tabs `Case 1` to `Case 10`. |
| `scores_all_<tag>.csv` | one per input case | The scores and ranks of every input case, not only the chosen ones. `rank_A` ranks by curve mismatch. `rank_B` ranks by distance to the posterior. `rank_hybrid` is the mean of both. |
| `posterior_summary_<tag>.csv` | 6 | The result of the MCMC for each parameter and for the two noise levels: `mean`, `sd`, 5th, 50th and 95th percentiles. `sd_ratio_vs_prior` is the posterior spread divided by the prior spread. `well_determined` is `yes` when this ratio is below 0.5. |
| `mcmc_diagnostics_<tag>.csv` | 6 | Convergence checks. `r_hat` should be below 1.1. `autocorr_time` is the number of steps a chain needs to forget its start. The noise levels appear here as `log_sigma_*`. |
| `posterior_samples_<tag>.csv` | about 38,000 | Every 10th kept MCMC draw. Columns: the 4 parameters in real units, and `log_sigma_oil`, `log_sigma_water` (natural log). Use it to redraw the posterior plots or to sample new parameter sets. |

The MCMC files only exist when method B ran. Method B needs at least 8 cases and their parameters.

## Figures written to `solution/histmatch_output/figs/`

The notebook saves five PNG files. All use one colour scheme. Black is the observed history. Blue is our cases and the posterior. Orange is a known true value (tests only). In the draft figures there is no orange.

| File | What it shows | How to read it | What the draft run shows |
|---|---|---|---|
| `fit_top10_<tag>.png` | The chosen cases against the observed history. Left column: oil. Right column: water. Top row: cumulative curves in million STB. Bottom row: the residual (predicted minus observed) as % of the final observed value. | Thick black line = observed history. Dark blue = rank 1. Light blue = ranks 2 to 10. Grey = other supplied cases. A good case lies on the black line, and its residual stays near 0. | The 10 chosen curves lie on the history. Grey cases fall away, mainly in water. Rank 1 ends about +0.9 % off in oil and -2.6 % off in water. The weakest chosen case is off by about +3 % in oil and -9 % in water at the end. |
| `scores_<tag>.png` | Horizontal bars: the mismatch score (NRMSE, % of the final observed cumulative) of the 25 best cases. | A shorter bar is a better match. Dark blue = chosen. Light blue = not chosen. The dashed line is the provisional 1 % accept level. | Cases 11, 40, 50 and 20 are under 1 %. The next cases jump to about 1.7 % and above. The 10 chosen cases are 11, 40, 50, 20, 24, 47, 65, 19, 35 and 32. The pool of 25 has few good matches. |
| `posterior_<tag>.png` | A corner plot of the MCMC result for the 4 parameters. The diagonal shows a histogram of each parameter. Below the diagonal, each panel shows a scatter of two parameters. | A narrow histogram means the history fixes that parameter. A flat histogram means the history says little about it. Open circles are the chosen cases. Use it to see if the chosen cases sit inside the posterior cloud. | Only the porosity multiplier is fixed (about 1.33, 5-95 % range 1.30 to 1.36). Fault transmissibility, permeability multiplier and aquifer pore volume stay as wide as their prior. A weak link exists: a higher permeability multiplier goes with a slightly lower porosity multiplier. |
| `trace_<tag>.png` | The value of each parameter along the MCMC steps, for 12 walkers. One panel per parameter. | The dashed line marks the end of burn-in (the first half, which is thrown away). After that line the traces should look like steady noise with no drift. | After a short start-up, all four parameters wander steadily around their range. This agrees with `mcmc_diagnostics_draft.csv`: R-hat is 1.03 to 1.05 (below 1.1), and chains forget their start after about 160 to 200 steps. |
| `predictive_<tag>.png` | The posterior prediction against the history. Left: oil. Right: water. Both in million STB, over the matching window. | Black dashed line = observed history. Blue line = posterior median. Blue band = P10 to P90 range over 300 posterior draws. The line should follow the black dashes, and the band shows the uncertainty. | The median follows the history closely in oil and water. The band is very thin. Many parameter sets give nearly the same history, so the history alone cannot separate them. |

Note: the figures show the **matching window** (1998 to 2008-01-01). Forecast dates after 2008 are in `top10_series_<tag>.csv` but are not plotted.

## Version 2: `notebook_hist_matching_v2.ipynb`

v2 is a copy of the notebook with six additions (sections 10a to 10g). It reads the same input folder. It writes to `solution/histmatch_output_v2/`, so the v1 results stay as they were. That folder holds all the files and figures described above, plus the ones below.

### Extra datasets

| File | Rows | What it holds |
|---|---|---|
| `scorecard_<tag>.csv` | one per input case | **The comparison table.** One row per case, sorted by `rank_A`. Fit: `score_%`, `max_err_%`, `err_final_oil_%`, `err_final_water_%`. Against the best: `vs_best` (score divided by the lowest score). Ranks: `rank_A`, `rank_B`. Benchmark: `I_max` (worst implausibility) and `not_ruled_out` (`I_max` at or below 3). Hold-out: `early_%`, `late_%`. Flags: `pareto` (on the oil-water front) and `chosen` (in the top 10). |
| `direct_vs_mcmc_<tag>.csv` | 4 | The posterior of each parameter by MCMC and by direct sampling: mean, standard deviation, 5th and 95th percentiles. `shift_in_sd` is the difference of the means in MCMC standard deviations. `sd_ratio` is the ratio of the standard deviations. Values near 0 and 1 mean the two methods agree. |
| `implausibility_nroy_<tag>.csv` | 4 | For each parameter: the range of values that the history does **not** rule out (`nroy_min`, `nroy_max`) against the full range. `share_of_range_left` near 1 means the history does not restrict that parameter. |
| `new_param_sets_<tag>.csv` | up to 10 | **New candidate parameter sets** from the parameter search: the 4 parameters in real units, `proxy_score` (the proxy's *predicted* mismatch, as a fraction), `nearest_case` and `dist_to_nearest_case` (0-1 scaled distance to the closest supplied case), and `proxy_error` (the proxy's own error, in score units). Rank 1 is the best set. The others are spread over the region that scores within the proxy error of the best. These sets have no simulated curve. |
| `holdout_<tag>.csv` | one per input case | The hold-out test. `early` and `late` are the scores on the first 70 % and the last 30 % of the window. Ranks on each part. `chosen_on_early` marks the top 10 chosen on the early part only. |

### Extra figures (in `histmatch_output_v2/figs/`)

| File | What it shows | How to read it | What the draft run shows |
|---|---|---|---|
| `direct_sampling_<tag>.png` | The posterior of each parameter by MCMC (blue bars) and by direct sampling (black line). | The two should match. The black line is noisier because it has fewer effective samples. | They match for all 4 parameters. Verdict: agree. |
| `implausibility_<tag>.png` | For each parameter, the lowest implausibility reached at each parameter value. Shaded band = not ruled out (3 or less). | A curve that stays in the band means the history does not restrict that parameter. A curve that rises out of the band marks values that the history rules out. | Only the porosity multiplier is restricted, to about 1.26 to 1.39. The other three stay in the band everywhere. |
| `new_sets_<tag>.png` | The new candidate sets (dark diamonds) against the posterior cloud (light blue) and the chosen pool cases (open circles). Porosity on the x axis. | The diamonds should sit inside the cloud. They show how far the new sets differ from the pool cases in the three free parameters. | The sets spread over the full range of the free parameters, at porosity 1.30 to 1.38. Several are far from any pool case. |
| `new_sets_curves_<tag>.png` | The proxy's predicted curves of the new sets (light blue) against the history (black dashes). Oil left, water right. | The curves should lie on the history. | They lie on the history. The water curves spread a little at the end. |
| `tradeoff_<tag>.png` | Left: oil error against water error for every case (log scales) with the Pareto front. Right: the signed error on the last date. | Left: a point close to the lower left is good in both. Right: the quadrant shows the direction of the bias. | Oil and water errors move together. Only case 11 is on the front. The chosen cases split into "oil high, water low" and "oil low, water high". |
| `holdout_<tag>.png` | The score on the early period against the score on the late period (held out), for every case. | Points near a diagonal line mean a good early fit predicts a good late fit. Dark blue = chosen on the early period. | The points follow a line. The cases with the best early scores also have the best late scores. |
