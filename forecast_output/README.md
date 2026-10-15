# Output folder of `notebook4_forecast.ipynb`

The 20-year production forecast (2008-04-01 to 2028-01-01, 80 quarters) of the 10 history-matched cases from `hist_match_output/top10_cases_real.csv`. The method and findings are in `summary_forecast.md` (repo root).

**Settings of this run (version 3, 15 Oct 2026):**
- **Method.** `crm_wor`: a capacitance-resistance model (CRM) for the liquid rate, fitted on the last 12 quarters, split into oil and water with the ln(WOR) trend moved through the latest WOR. Gas = GOR × oil.
- **Injection.** Water injection is held at 9,326.61 STB/d after 2008.
- **Start.** Every forecast starts from the **observed** 2008-01-01 cumulatives and rates (A2), chosen by the start-point hindcast. The case supplies the shape of the forecast.
- **Previous version.** Version 2 started each case from its own 2008 values and is archived in `archive/v2_pre_bias_fix_2026-10-15/forecast_output/`.

**Units.** Oil and water in STB, gas in MSCF; rates per day. Columns ending in `_MM` are in millions (MMSTB, or million MSCF for gas). Errors ending in `%` are percentages, not fractions. Phase names are `oil`, `gas`, `water`.

## The deliverable

| File | What it holds |
|---|---|
| `10_Final_deliverables.xlsx` | **The submission.** A copy of `dataset_ori/09 Template Deliverable.xlsx` with sheets `Case 1` to `Case 10` (sheet k = history-matching rank k). <ul><li>Columns: Date, Gas production cumulative [MSCF], Oil production cumulative [STB], Water production cumulative [STB].</li><li>80 quarterly rows from 04/01/2008 to 01/01/2028. The values are rounded to whole numbers.</li><li>The Date column and headers are untouched from the template.</li><li>The file was read back and checked: 80 rows per sheet, finite, cumulatives never decrease, continuous with 2008, equal to the forecast.</li></ul> |

## The forecast

| File | Rows | What it holds |
|---|---|---|
| `forecast_summary_by_case.csv` | 10 | Per case: oil and water cumulative in 2008 and 2028 (`_MM`), gas in 2028 (`gas_2028_MMMSCF`), oil rate in 2028 (STB/d) and water cut in 2028 |
| `forecast_cases_long.csv` | 6,153 | **For Power BI and the slides.** One row per case × date × phase: `rank` (0 = observed field), `case`, `date`, `period`, `phase`, `cumulative`, `rate_per_day`. The `period` values are:<ul><li>`history (case)`: the case's own curve, 1998-2008</li><li>`history (observed)`: the field</li><li>`forecast`: the adopted forecast</li><li>`forecast, own start (sensitivity)`: the version 2 start</li></ul> |
| `forecast_bands.csv` | 1,200 (5 sets × 3 phases × 80 dates) | Uncertainty by date: `set`, `phase`, `date`, `P10`, `P50`, `P90` (cumulatives). The sets are:<ul><li>`10 cases`</li><li>`posterior (200 sets)`: 200 MCMC parameter sets, same method and start</li><li>`10 cases, own start (sensitivity)`</li><li>`field reference (history only)`: the method on the observed history, no model; P10 = P50 = P90</li><li>`wider range ...`: here **P10 = minimum, P90 = maximum** of the 10 cases, the own-start cases and the field reference, and P50 = the median of the 10 cases. Use this range to show the uncertainty, not the 1 % spread of the 10</li></ul> |
| `forecast_run_summary.json` | - | The key numbers in one file: method, start option, hindcast and backtest results, checks, oil and water in 2028 (P10 / mean / P90), own-start oil, the wider range, the field reference |

## How the method and the start were chosen

| File | Rows | What it holds |
|---|---|---|
| `backtest_all_settings.csv` | 57 | **Method backtest.** Each method × fit window (8/12/16 quarters) × WOR anchoring (`fit`/`last`) × cut-off (2005/2006/2007), forecast to 2008. The columns are:<ul><li>median absolute error on the 100 simulated cases (`cases_med_abs_*`)</li><li>median signed error (`cases_bias_*`)</li><li>error on the real field (`field_*`)</li></ul>All are % of the volume produced after the cut-off |
| `backtest_best_per_method.csv` | 4 | The best setting of each method, averaged over the cut-offs. `score` = mean of oil and water on the cases (lower is better); `field_score` = the same on the field. `crm_wor` wins (2.94 %) |
| `hindcast_start_all.csv` | 1,515 | **Start-point hindcast.** Each truth × cut-off × start option. The truths are the real field and each simulated case acting as a pseudo-field. The options are `main` (own start), `A1`, `A2`, `fade_365` and `fade_730`. The columns are `err_oil_%`, `err_water_%` and `abs_mean` (mean of the two absolute errors) |
| `hindcast_start_summary.csv` | 5 | Per start option:<ul><li>median and mean errors on the pseudo-fields</li><li>the field error at each cut-off and on average</li><li>the share of pseudo-fields where it beats `main`</li><li>the paired Wilcoxon p-value against `main`</li></ul>`A2` is best (3.52 % against 7.47 % for `main`; field 3.80 % against 6.54 %) |
| `alternative_start_comparison.csv` | 9 | Oil, water and gas in 2028 (P10 / mean / P90 of the 10, `_MM`) for the adopted start (A2), the own start and A1 (observed cumulative, own rates); `mean_change_vs_main_%` is against A2 |

## Figures (`figs/`)

Black = observed field. Blue = our cases (dark blue = adopted forecast). Orange = the version 2 own start. Dashed black = field reference.

| File | What it shows |
|---|---|
| `backtest.png` | Left: backtest error of each method's best setting at each cut-off (log scale, lower is better). Right: the real field cut at 2006-01-01, with the final method's oil and water forecast against what happened |
| `forecast_fan.png` | Top: cumulative oil, water and gas, 1998-2028, with the 10 cases, the posterior P10-P90 band (blue), the wider range (grey) and the field reference. Bottom: the rates. The cases' own 2008 rates differ from the start of the forecast because the forecast starts at the observed rates |
| `alternative_start.png` | Oil and water in 2028 for each of the 10 cases under the three starts, with the field reference as a dashed line |

## Rerun

Run `notebook4_forecast.ipynb` from the repo root after notebooks 2 and 3. It takes about 2 minutes, mostly for the backtest and the hindcast. It overwrites every output file here; this README is not touched.
