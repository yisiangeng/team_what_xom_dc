# Output folder of `notebook3_hist_match.ipynb`

All files end with the tag `_real` (the run on the observed field history). Mismatch scores are **fractions** of the observed cumulative on 2008-01-01. Example: `0.005` means 0.5 %. Findings: `summary_hist_mat.md` (repo root).

## The answer

| File | Rows | What it holds |
|---|---|---|
| `top10_cases_real.csv` | 10 | **The 10 chosen cases**, ranked 1 to 10. `source` = `simulated` (an existing case, `case_id` given) or `new_set` (a posterior parameter set, curves from the proxy). Scores: `score` (headline NRMSE of oil and water), `nrmse_oil`, `nrmse_water`, `max_rel_err`, `err_final_oil`, `err_final_water` (signed, last date), `I_max` (implausibility; 3 or less = not ruled out), `proxy_score`. Selection: `in_band` (inside the equal-best band), `rank_all` (rank by score among all candidates), `reason`. The 4 parameters. `misfit_elton_3phase` (Elton's objective, for comparison) and the cumulatives on 2008-01-01. Physics check of the raw proxy curves (new sets only, empty for simulated cases): `raw_pre_production_max` (must be 0), `raw_largest_drop_STB` and `raw_largest_drop_%_of_final` (the dip removed by a running maximum in `top10_series_real.csv`) |
| `top10_series_real.csv` | 10 x 41 dates | The 1998-2008 cumulative curves of the 10: `rank`, `cand_id`, `source`, `case_id`, `date`, `gas_cum` [MSCF], `oil_cum` [STB], `water_cum` [STB], `curve_source` (`simulator` or `proxy`). **Input of the forecast step** |

## All candidates and the posterior

| File | Rows | What it holds |
|---|---|---|
| `candidates_real.csv` | 100 + 4,000 | Every candidate of the choice of 10 (simulated cases and posterior sets), with the same scores, `not_ruled_out` and `in_band` (inside the equal-best band; empty if ruled out) |
| `posterior_samples_real.csv` | ~19,000 | Every 10th kept MCMC draw: 4 parameters and `log_sigma_oil`, `log_sigma_water`. **Read by notebook 4** (posterior band) |
| `hm_run_summary_real.json` | - | The key numbers of the run in one file, including the checks and the selection rule |

## Supporting tables: `hm_details_real.xlsx`

One Excel workbook, one table per sheet. The first sheet, `contents`, lists the sheets. In Python: `pd.read_excel("hist_match_output/hm_details_real.xlsx", sheet_name="posterior", index_col=0)`.

| Sheet | Rows | What it holds |
|---|---|---|
| `simulated_cases` | 100 | Every simulated case, one row each:<ul><li>scores as fractions (`score`, `nrmse_*`, `err_final_*`), `I_max`, `rank_A`, `post_dist` (Mahalanobis distance to the posterior, 0-1 scaled), split, parameters</li><li>scorecard layers in % (`vs_best`, `not_ruled_out`, `early_%`, `late_%`, `pareto`, `chosen`)</li><li>hold-out: `early_score`, `late_score` (first 70 % / last 30 % of the window), `rank_early`, `rank_late`, `chosen_on_early`</li></ul> |
| `posterior` | 8 | One row per parameter:<ul><li>posterior `mean`, `sd`, `p05`, `p50`, `p95`; `sd_ratio_vs_prior` below 0.5 = the history determines it (`well_determined`)</li><li>MCMC `r_hat` (below 1.1) and `autocorr_time`; the `log_sigma_*` rows hold only these</li><li>direct (importance) sampling: `*_direct`, `shift_in_sd` near 0 and `sd_ratio` near 1 = agreement</li><li>range not ruled out by the history: `nroy_min`, `nroy_max`, `share_of_range_left`</li></ul> |
| `implausibility_sensitivity` | 3 | How much is not ruled out at an assumed error of 0.5, 1 and 2 % |
| `global_search` | 10 | Differential evolution on the proxy: best parameter set per run, its score, edge flags |
| `recovery_test` | 10 | Validation cases used as fake histories: coverage of the true parameters, porosity error, R-hat, fit of the chosen 10 against the pool's best 10 |
| `selection_comparison` | 5 | The adopted 10 against the version 3 k-means 10, "simulated only" (v1 method A), Elton's 10 and Zi Yang's 10, all scored with the same proxy: best / mean / worst score, how many not ruled out, how many outside the train range, mean pairwise distance (spread) |
| `other_teams_10` | 20 | Elton's and Zi Yang's 10 parameter sets with their scores under our proxy |

## Figures (`figs/`)

Black = observed history. Blue = our cases and the posterior (dark blue = rank 1 or chosen). Orange = a known true value (recovery test only). Solid line = simulator curve, dashed line = proxy curve (new set).

| File | What it shows |
|---|---|
| `fit_top10_real.png` | The 10 against the history (oil left, water right). Bottom row: residual in % of the final observed value |
| `scores_real.png` | Score bars: the 25 best simulated cases and the chosen new sets (hatched) |
| `selection_real.png` | The 10 (numbered) across the region the history does not rule out, porosity on the x axis |
| `posterior_real.png` | Corner plot of the posterior; circles = chosen simulated cases, diamonds = chosen new sets |
| `trace_real.png` | Chain traces with the end of burn-in |
| `predictive_real.png` | Posterior median and P10-P90 band against the history |
| `direct_sampling_real.png` | Posterior by MCMC (bars) and by direct sampling (line) |
| `implausibility_real.png` | Lowest implausibility along each parameter; shaded = not ruled out |
| `tradeoff_real.png` | Oil error against water error (Pareto front), and the direction of the error on the last date |
| `holdout_real.png` | Early score against late (held-out) score |
