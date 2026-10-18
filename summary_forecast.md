# Forecast to 2028: methods and findings

15 October 2026, version 4. Version 3 started the forecast from the observed 2008 values. Version 4 uses the 10 cases chosen by match quality first (organiser FAQ 9, `summary_hist_mat.md` section 8) and adds physical checks of the forecast constants (section 6b). The method is unchanged. Version 3 is archived in `archive/v3_pre_faq_2026-10-15/`, version 2 (each case's own start) in `archive/v2_pre_bias_fix_2026-10-15/`.
- **Notebook:** `notebook4_forecast.ipynb`. **Code:** `src/forecast_lib.py`.
- **Outputs:** `forecast_output/`, including the deliverable `forecast_output/10_Final_deliverables.xlsx`.
- **Inputs:** the 10 history-matched cases of `notebook3_hist_match.ipynb` (`summary_hist_mat.md`).

## At a glance

- **Method.** Each of the 10 cases is extended from 2008-04-01 to 2028-01-01 (80 quarters) with a **capacitance-resistance model (CRM)** for the total liquid rate, split into oil and water by the **water-oil-ratio trend**.
  - The CRM is driven by the water injection rate, held at its 2008 value.
  - The water-oil-ratio trend is ln(WOR) linear in cumulative oil, moved through the latest WOR.
  - Gas = GOR × oil.
- **Why this method.** It won a backtest on all 100 simulated cases and on the real field: the last 1-3 years were hidden and forecast. Median error of the produced volume: **2.9 %** on the cases, 4.0 % on the field.
  - Arps decline + WOR: 4.5 %. Liquid trend + WOR: 5.4 %.
  - Elton's curve families: 64 %. They cannot be used for the forecast.
- **Start point: the observed 2008 values (new in version 3).** Each forecast starts from the field's observed 2008 cumulatives and rates. The case supplies the shape: the CRM time constant and gain, and the WOR slope.
  - **Why.** The 10 cases match the observed cumulatives, but in 2008 their oil rates are 11-15 % too high and their water rates about 10 % too low. No parameter set removes this gap.
  - **Evidence.** A blind hindcast (section 5) compared the two starts on 100 pseudo-fields and on the real field:
    - Starting from the observed state halved the error on the pseudo-fields (3.5 % against 7.5 %, p ~ 1e-36).
    - It also lowered the error on the real field (3.8 % against 6.5 %).
- **Result (10 cases, P10 / P50 / P90 on 2028-01-01).**
  - Oil: **58.6 / 58.7 / 59.0 MMSTB** (38.4 MMSTB in 2008).
  - Water: 85.0 / 85.3 / 85.5 MMSTB.
  - Gas: 21.3 / 21.3 / 21.4 million MSCF.
  - Oil rate: about 1,600 STB/d in 2028, at a water cut of 0.87-0.88.
- **Uncertainty.** The 10 cases agree within about 1 %, which understates the real uncertainty. The **wider range** (for slides and Power BI) is oil 54.2-61.7 MMSTB and water 81.8-90.0 MMSTB. It covers the cases, their own-start forecasts and the data-only field forecast.
- **Deliverable.** The 09 template filled for Case 1 to Case 10 (history-matching rank order), 80 quarters each. The Date column is left untouched. The file was read back and validated.

## 1. The problem and the organiser's instructions

The organiser asks for a forecast of field production over the next 20 years, with the "best 10 forecast cases" written into the template `09 Template Deliverable.xlsx` (kick-off slides 26, 31). Each of its 10 sheets has:
- **Columns:** Date | Gas production cumulative [MSCF] | Oil production cumulative [STB] | Water production cumulative [STB];
- **Rows:** 80 quarterly dates from 04/01/2008 to 01/01/2028.

Rock properties are static (slide 30), and there are no new wells after 2008 (team confirmation).

**The difficulty.** The simulator runs we were given stop at 2008-01-01, so no simulated or observed data exists after that date. The proxy maps parameters to the 41 dates it was trained on, so it cannot forecast. Each history-matched case therefore has to be extended by a forecasting method fitted to its own late history. That method must be chosen on evidence: the backtest.

## 2. Methods considered

| Code | Method | How it works | Source |
|---|---|---|---|
| `elton` | Cumulative curve families (Elton's step 4) | Arps-type cumulative curve (oil) and Gompertz curve (water) fitted to the whole 1998-2008 cumulative curve, then shifted to be continuous | `our_work/proxy_model_elton/forecast_pipeline.py` |
| `arps_wor` | Decline curve + WOR trend | Arps hyperbolic decline of the oil rate fitted on the last *n* quarters, b in [0, 1], switching to exponential at a terminal decline of 5 %/yr (modified hyperbolic). Water = WOR(Np) × oil | Arps (1945); terminal-decline remedy attributed to Robertson (1988) [2, 3]; WOR trend [4-6] |
| `liq_wor` | Liquid trend + WOR split | Total liquid rate with an exponential, non-increasing trend; oil = liquid / (1 + WOR), WOR from the trend | Water-cut / fractional-flow extrapolation [4, 5] |
| `crm_wor` | **CRM + WOR split** | Field-level CRM ("tank"): q_L(t) relaxes towards G × I with a time constant tau, i.e. q_L,k = q_L,k-1 e^(-dt/tau) + (1 - e^(-dt/tau)) G I_k. tau and G are fitted on the last *n* quarters; injection I is held at 9,327 STB/d after 2008. Split by the WOR trend | Sayarpour et al. (2007, 2009) [7]; CRM + fractional-flow split [8] |

**Not used: extending the proxy or a recurrent network in time.** There is no training data after 2008. Si Jie's recurrent networks, which can run past 2008, flattened their oil decline after 2010 (`our_work/proxy_model_sijie/summary.md`, limitation 1). The organisers' reference (Zhang et al. 2023) uses its LSTM only within the simulated period.

**Shared rules.**
- **Rates.** Quarter means, (cumulative - cumulative a quarter earlier) / days, the same for every curve.
- **Gas.** GOR × oil, with the GOR of the case's last 4 quarters (about 0.3633 in every case and in the field).
- **WOR anchoring.** `fit` uses the least-squares line. `last` keeps the fitted slope but moves the line through the last 2 quarters' WOR. Over 2002-2008 the field's ln(WOR) is linear in cumulative oil (r = 0.998, slope 0.10-0.13 per MMSTB), but a line fitted over a window lags behind the latest data.
- **Physical constraints.** Cumulatives only increase; the oil rate never increases (no new wells); water cut < 1. Each forecast is continuous with its 2008 start values, and its first-quarter rates are within 10 % of the start rates.

## 3. Backtest: how the method was chosen

**Design.**
- **Cut-offs.** 2005-01-01, 2006-01-01 and 2007-01-01. Each method sees only the data up to the cut-off and forecasts to 2008-01-01 (12, 8 and 4 quarters ahead).
- **Curves.** All **100 simulated cases** and the **observed field**. Elton's backtest used one split on the cases only. Hindcasting of this kind is the standard way to compare decline methods; the length of the history used is the most important factor [9].
- **Error.** The relative error of the volume produced between the cut-off and 2008, per phase. The median over the 100 cases is used for the choice, and the field error is reported next to it.
- **Settings searched.** Fit window 8, 12 or 16 quarters; WOR anchoring `fit` or `last`. Injection after the cut-off is held constant, as in the real forecast.

**Results** (best setting of each method, mean over the 3 cut-offs, error in % of the produced volume):

| Method | Window | WOR anchor | Cases: oil | Cases: water | Cases: bias oil / water | Field: oil / water | Score |
|---|---|---|---|---|---|---|---|
| **crm_wor** | 12 | last | **2.80** | **3.08** | -1.9 / -2.1 | 2.6 / 5.3 | **2.94** |
| arps_wor | 12 | last | 4.67 | 4.26 | -3.0 / -4.1 | 2.5 / 7.3 | 4.47 |
| liq_wor | 16 | fit | 4.85 | 6.02 | -4.2 / -2.3 | 4.1 / 8.1 | 5.44 |
| elton | - | - | 62.3 | 66.1 | +62.3 / -66.1 | 61.8 / 70.2 | 64.2 |

- **The ensemble does not help.** Averaging `crm_wor` and `arps_wor` scores 3.48 %, worse than `crm_wor` alone (2.94 %), so the final method is `crm_wor` alone.
- **Elton's method fails the backtest.** It over-forecasts oil by about 60 % and under-forecasts water by about 65 % within 1-3 years. Fitting a curve family to the whole cumulative curve, plateau included, does not carry the late decline forward. This is the same reason his forecast jumped at 2008 (oil 5,247 to 8,905 STB/d; water falling to about 0 by 2018).
- **The settings that won** (12-quarter window, WOR line through the latest data) are themselves accuracy improvements found by the backtest.

Figure: `forecast_output/figs/backtest.png`.

## 4. Forecast results (start: observed 2008 values)

All 10 start from the observed 2008-01-01 values: oil 38.38 MMSTB, water 12.81 MMSTB, oil rate 5,247 STB/d, water rate 7,782 STB/d.

| Case | Source | Oil 2028 [MMSTB] | Water 2028 [MMSTB] | Gas 2028 [MM MSCF] | Oil rate 2028 [STB/d] | Water cut 2028 |
|---|---|---|---|---|---|---|
| 1 | new set | 58.73 | 85.54 | 21.34 | 1,589 | 0.875 |
| 2 | new set | 58.73 | 85.52 | 21.34 | 1,589 | 0.875 |
| 3 | simulated case 6 | 58.69 | 85.29 | 21.33 | 1,587 | 0.875 |
| 4 | new set | 58.59 | 85.39 | 21.28 | 1,572 | 0.876 |
| 5 | new set | 58.73 | 85.53 | 21.34 | 1,589 | 0.875 |
| 6 | simulated case 97 | 58.95 | 85.23 | 21.42 | 1,620 | 0.873 |
| 7 | simulated case 76 | 58.91 | 85.25 | 21.39 | 1,615 | 0.873 |
| 8 | simulated case 11 | 59.03 | 85.14 | 21.45 | 1,632 | 0.872 |
| 9 | simulated case 40 | 59.00 | 85.17 | 21.43 | 1,628 | 0.872 |
| 10 | simulated case 73 | 58.52 | 84.13 | 21.26 | 1,581 | 0.874 |

**Uncertainty** (cumulative on 2028-01-01):

| Set | Oil [MMSTB] | Water [MMSTB] | Gas [MM MSCF] |
|---|---|---|---|
| The 10 cases (P10 / P50 / P90) | 58.58 / 58.73 / 59.00 | 85.04 / 85.27 / 85.53 | 21.28 / 21.34 / 21.43 |
| Posterior, 200 parameter sets from the MCMC, same start (P10 / P50 / P90) | 58.23 / 58.38 / 58.59 | 85.00 / 85.20 / 85.46 | 21.16 / 21.21 / 21.28 |
| The 10 cases, own start (sensitivity, P10 / P50 / P90) | 60.83 / 61.17 / 61.69 | 82.40 / 82.82 / 83.12 | 22.10 / 22.22 / 22.41 |
| Field reference (the same method on the observed history, no model) | 54.16 | 90.01 | 19.68 |
| **Wider range** (min - max of all of the above except the posterior) | **54.16 - 61.71** | **81.83 - 90.01** | **19.68 - 22.42** |

- **The spread between the 10 is narrow (about 1 %).** All 10 match the same history and now share the same start, so the method carries them along similar paths. The posterior band is similarly narrow.
- **These bands leave out the uncertainty of the start and of the model,** which are larger. In the hindcast, the 10 blind matches' range never contained the field's water (0 of 3 cut-offs).
- **Use the wider range** when presenting the uncertainty. It spans the case forecasts, the version 2 own-start forecasts and the data-only field forecast.

Figures: `forecast_fan.png` (cumulatives and rates, 1998-2028; the 10 cases, the posterior band, the wider range, the field reference).

## 5. Start point: own 2008 values or observed? (blind hindcast)

**The problem.** The 10 cases match the observed *cumulatives* within 1-3 %, but their 2008 *rates* are off:

| Rate on 2008-01-01 | Field | The 10 cases |
|---|---|---|
| Oil | 5,247 STB/d | 5,840-6,050 STB/d (11-15 % high) |
| Water | 7,782 STB/d | 6,940-7,040 STB/d (about 10 % low) |

No parameter set in the design removes this gap (`summary_hist_mat.md`, "Main limits"). A forecast that starts from each case's own state carries it for 20 years. Version 2 did that, and the team agreed to test fixes and keep one only if it improved the forecast significantly.

**The test** (`notebook4_forecast.ipynb` section 4; full version with diagnostics in `improve_forecast/hindcast_start_test.py`). The whole chain is repeated blind at an earlier date:
- **Truths.** The real field, and each of the 100 simulated cases as a pseudo-field (the other 99 are the candidates).
- **Blind match.** At each cut-off (2005, 2006, 2007), the 10 simulated cases with the lowest oil + water cumulative misfit on 1999 to the cut-off only.
- **Forecast.** Each of the 10 to 2008-01-01 with the final method, from each start:

| Start | What it does |
|---|---|
| `main` | The case's own cumulative and rates (version 2) |
| `A1` | The observed cumulative, the case's own rates |
| `A2` | The observed cumulative and rates; the case supplies tau, G and the WOR slope |
| `fade_T` | The observed rates, fading back to the case's own path over T days |

- **Error.** The mean of the 10 against the truth, in % of the volume produced after the cut-off.
- **Rule, set before the run.** Switch only if the new start is better on the pseudo-fields (paired Wilcoxon, p < 0.01) **and** on the real field.

**Results** (mean of |oil| and |water| error, %):

| Start | Pseudo-fields (300 pairs) | Better than `main` | Wilcoxon p | Field 2005 | Field 2006 | Field 2007 | Field mean |
|---|---|---|---|---|---|---|---|
| `main` (version 2) | 7.47 | - | - | **2.48** | 4.03 | 13.10 | 6.54 |
| `A1` | 3.96 | 83 % | 4e-36 | 3.32 | 4.95 | 10.11 | 6.13 |
| `fade_365` | 3.72 | 82 % | 2e-36 | 3.87 | 4.21 | 5.41 | 4.50 |
| `fade_730` | 3.65 | 82 % | 2e-36 | 4.25 | 3.86 | 4.25 | 4.12 |
| **`A2` (adopted)** | **3.52** | **83 %** | **5e-37** | 5.19 | **3.55** | **2.66** | **3.80** |

- **A2 wins in both worlds.** It is better on the pseudo-fields, which are in-family, and on the real field, which is probably not.
  - Most of the gain on the pseudo-fields comes from the cumulative (A1).
  - Anchoring the rates adds a further, significant gain: A2 is better than A1 in 73 % of the pairs (p ~ 3e-14).
- **The real field shows the 2008 problem directly.**
  - At the 2007 cut-off, the blind matches already run 8 % high on oil rate and 8 % low on water rate.
  - Their own-start forecast then misses the field's oil by about 11 % and its water by about 16 %. A2 misses them by 0.2 % and 5 %.
  - At the 2005 cut-off, before the gap opens, `main` is better. The gain comes from the late cut-offs, which are the ones most like 2008.
- **Fading the anchor out does not help.** The longer the fade, the closer it gets to A2, and A2 itself is best.
- **The simulator's own continuation confirms the field is outside the model family.** This is a diagnostic in `improve_forecast/` that uses the matched cases' real simulator curves after the cut-off.
  - On the pseudo-fields it is the most accurate start of all (1.5 %).
  - On the real field it is the least accurate: 8-9 % error, with oil 4-15 % high and water 5-14 % low.
  - The simulator's dynamics therefore over-produce oil and under-produce water compared with the real field, and the gap grows with time.
- **Limit.** There are only 3 field cut-offs, so the field result alone cannot be tested statistically. The decision rests on the pseudo-fields, where the result is significant, and on the field agreeing.

**Effect on the 2028 forecast** (mean of the 10; `forecast_details.xlsx`, sheet `alternative_start_comparison`):

| Start | Oil 2028 [MMSTB] | Water 2028 | Gas 2028 | Against the adopted A2 (oil / water) |
|---|---|---|---|---|
| **A2: observed cumulative + rates (adopted)** | **58.79** | **85.22** | **21.36** | - |
| A1: observed cumulative, own rates | 60.96 | 83.04 | 22.15 | +3.7 % / -2.6 % |
| Own start (version 2 deliverable) | 61.22 | 82.75 | 22.24 | +4.1 % / -2.9 % |
| Field reference (no model) | 54.16 | 90.01 | 19.68 | -7.9 % / +5.6 % |

Figure: `forecast_output/figs/alternative_start.png`. Tables: `forecast_output/forecast_details.xlsx`, sheets `hindcast_start_summary`, `hindcast_start_all` and `alternative_start_comparison`.

## 6. What was done to improve accuracy

1. **Method chosen by evidence,** on 100 simulated cases and the real field at 3 cut-offs, instead of one split on the cases only.
2. **Settings tuned by the backtest:** fit window and WOR anchoring. Both winners use the late history only (12 quarters) and follow the latest WOR.
3. **The method uses more of the data.** The CRM uses the water injection record, which the decline methods ignore. It cut the error by about a third against the best decline method.
4. **Physical constraints:** continuity at 2008, monotone cumulatives, no oil-rate increase, water cut < 1, fixed GOR.
5. **Ensembling tested.** Averaging the two best methods was worse, so it was not used.
6. **Uncertainty carried from the history match:** a band from 200 posterior parameter sets.
7. **Start-point bias corrected (version 3).** The forecast starts from the observed 2008 rates, chosen by a blind hindcast. This halved the hindcast error on the pseudo-fields and cut it by 40 % on the real field.
8. **Honest range.** The wider range adds the start and model uncertainty that the case bands leave out.

### 6b. Physical realism of the forecast (organiser FAQ 5)

FAQ 5 asks that the models "remain physically realistic". Two groups of checks run in notebook 4 (section 5 and 5b), and all pass.

**The forecast curves** (all 10 cases, 80 quarters):

| Check | Result |
|---|---|
| Cumulatives never fall | Pass |
| Oil rate never rises (no new wells, FAQ 2) | Pass |
| Water cut below 1 | Pass (0.872-0.876 in 2028) |
| First-quarter rates within 10 % of the 2008 rates (no jump) | Pass |

**The fitted constants** (`forecast_output/forecast_by_case.csv`), fitted on the last 12 quarters of each case and of the observed field:

| Constant | Meaning | Field | The 10 cases | Check |
|---|---|---|---|---|
| G | Liquid produced per barrel of water injected, at steady state | 1.365 | 1.342-1.366 | Pass: inside its fitting range (0.2-5) and within 2 % of the field. G above 1 means that the aquifer supports the pressure as well as the injection |
| G x injection | The liquid rate after 2008 | 12,728 STB/d (observed 2008: 13,029) | 12,519-12,741 STB/d | Pass: within 4 % of the observed 2008 rate, so no jump |
| tau | How fast the liquid rate follows injection | 5 days | 5 days | Reported, not tested. It sits at the lower bound of the fit in every case and in the field. With quarterly data, liquid / injection is 1.32-1.40 in every quarter of 2005-2008: the response is faster than one quarter, so tau cannot be resolved below that. With constant injection the forecast does not depend on it |
| WOR slope | How fast ln(water-oil ratio) rises with cumulative oil | 0.130 per MMSTB | 0.074-0.078 | Pass: positive (the water cut rises). **But the cases' slope is only 57-60 % of the field's** |

- **G and aquifer size.** Over the 10 cases, G rises with the aquifer volume (Spearman +0.50), as it should, but the range of G is narrow because all 10 match the same liquid history.
- **The WOR slope is the known gap, now measured.** The field's water rises with cumulative oil about 1.7 times faster than in any matched case. This is the late water rise that no simulated case reproduces (`summary_hist_mat.md`, limitation 1). A2 fixes the level in 2008, but the forecast then follows the cases' slower water rise. So the 10 cases probably sit on the high-oil, low-water side, and the field reference (oil 54.2, water 90.0 MMSTB) marks the other side of the wider range. This is why the wider range should be presented.

**Considered but not changed.**
- **A model-error term in the history match.** Notebook 3 already samples the noise level of the misfit (`log_sigma`) with AR(1) correlation, so the misfit the simulator cannot explain is already absorbed into the noise. A structured model-error term would widen the posterior, but it would not move the forecast. The wider range covers the same need more directly.
- **Rates in the history-match score.** No parameter set fits both the 2008 totals and the 2008 rates, so this would only trade one fit for the other.

## 7. Limitations

1. **Horizon.** The backtest and the hindcast cover at most 12 quarters; the forecast covers 80. They are evidence for the method and the start, not proof of 20-year accuracy.
2. **Operating assumptions.** Water injection is constant at 9,327 STB/d after 2008; no new wells; no change of operating conditions. A change in injection changes the CRM liquid rate directly.
3. **Injection in the simulated cases.** The backtest on the simulated cases uses the field's injection record, assuming the cases follow the field's injection schedule. The cases' files do not report injection.
4. **The model family does not reproduce the field's late behaviour.** Anchoring the start fixes the level in 2008, but the decline shape and WOR slope still come from the cases. The cases' WOR slope is only 57-60 % of the field's (section 6b), so the forecast is probably high on oil and low on water later on. The field reference (54.2 MMSTB of oil) marks that side of the wider range.
5. **The 10 sheets differ only in shape,** because they share the same start. Their spread (about 1 %) is not the forecast uncertainty; see the wider range.
6. **New sets.** Their 1998-2008 curves come from the proxy. Their quarterly rates carry the proxy's noise (about ±5 % between quarters), which the 12-quarter fit smooths.
7. **What the organisers score against is not known.** If they score against a continuation of this simulator, the own-start forecast would be the consistent one. In the hindcast, however, A2 was also better on the in-family pseudo-fields, so it is the better choice either way.
8. **The method choice has no significance test.** `crm_wor` won the backtest on the median error (2.9 % against 4.5 % for `arps_wor`), consistently over the cut-offs, but no paired test was run, unlike the start-point choice.
9. **Sources.** Several methods are cited through secondary sources (documentation, reviews, abstracts). The original papers were not all read.

## Sources

1. Arps, J. J. (1945). Analysis of decline curves. *Transactions of the AIME*, 160(1), 228-247. https://doi.org/10.2118/945228-G
2. IHS Harmony documentation, decline analysis theory: long-range hyperbolic extrapolation overstates reserves; the remedy attributed to Robertson (1988) is a switch to exponential decline. https://www.ihsenergy.ca/support/documentation_ca/Harmony_Enterprise/latest/content/html_files/ref_materials/analysis_method_theory/decline_analysis_theory.htm
3. Hart Energy, "Terminal decline in decline curve analysis": switch from hyperbolic to exponential at a terminal decline rate (Dmin). https://www.hartenergy.com/exclusives/terminal-decline-decline-curve-analysis-oil-wells-195380
4. Ershaghi, I. and Omoregie, O. (1978). A method for extrapolation of cut vs recovery curves. SPE 6977. Cited via lecture notes, Universidad Nacional de Cuyo: https://aulaabierta.ingenieria.uncuyo.edu.ar/pluginfile.php/231352/mod_page/content/8/Clase%20VI%20C-D-E-F-G%202025.pdf
5. Water-flooding characteristic curves, WOR type (Wright 1958, semi-log WOR against cumulative oil), reviewed in *Fuels* 7(7), MDPI: https://mdpi-res.com/d_attachment/fuels/fuels-07-00007/article_deploy/fuels-07-00007.pdf
6. Oil & Gas Journal, "Technique forecasts production from waterflooded reservoirs" (log WOR against cumulative oil): https://www.ogj.com/drilling-production/article/17255740/technique-forecasts-production-from-waterflooded-reservoirs
7. Sayarpour, M., Kabir, C. S. and Lake, L. W. (2007, 2009), CRM analytical solutions, including the field-level tank model (CRMT). Cited via Gan, Hartanto, Haynes and Sayarpour (2012), Windalia waterflood, *APPEA Journal*: https://publish.csiro.au/AJ/AJ11070
8. Oliveira, Horowitz and Rojas Tueros (2021). CRM combined with the Koval and Gentil fractional-flow models ("Kogen"), *Oil & Gas Science and Technology* 76: https://ogst.ifpenergiesnouvelles.fr/10.2516/ogst/2020090
9. Hindcasting decline-curve methods: the length of the history is the most critical factor. Louisiana State University: https://repository.lsu.edu/petroleum_engineering_pubs/465
10. Zhang, Z. et al. (2023). Robust method for reservoir simulation history matching using Bayesian inversion and LSTM network-based proxy. *SPE Journal* 28(3), 983-1007. https://doi.org/10.2118/203976-PA (organisers' reference; `reference/`)

## Files

| File | What it holds |
|---|---|
| `forecast_output/10_Final_deliverables.xlsx` | **The deliverable**: the 09 template, Case 1-10, 80 quarters of gas, oil and water cumulative (start: observed 2008 values) |
| `forecast_output/forecast_cases_long.csv` | History and forecast of every case and of the observed field: date, phase, cumulative, rate (for Power BI). Includes the own-start sensitivity as period `forecast, own start (sensitivity)` |
| `forecast_output/forecast_bands.csv` | P10 / P50 / P90 by date: the 10 cases, the posterior, the own-start sensitivity, the field reference, and the wider range (min / median / max) |
| `forecast_output/forecast_by_case.csv` | 2008 and 2028 values per case, and the fitted constants (tau, G, WOR slope) of each case and of the field |
| `forecast_output/forecast_details.xlsx` | Supporting tables, one sheet each: all backtest runs, the best setting per method, the start-point hindcast (every truth × cut-off × start, and the summary with the tests), the adopted forecast against the other starts |
| `forecast_output/forecast_run_summary.json` | Key numbers of the run |
| `forecast_output/figs/` | `backtest.png`, `forecast_fan.png`, `alternative_start.png` |
| `improve_forecast/` | The full hindcast with diagnostics (simulator continuation, cumulative errors, coverage) |
