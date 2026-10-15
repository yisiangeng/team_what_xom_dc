# Forecast to 2028: methods and findings

15 October 2026.
- **Notebook:** `notebook4_forecast.ipynb`. **Code:** `src/forecast_lib.py`.
- **Outputs:** `forecast_output/`, including the deliverable `forecast_output/10_Final_deliverables.xlsx`.
- **Inputs:** the 10 history-matched cases of `notebook3_hist_match.ipynb` (`hist_mat_summary.md`).

## At a glance

- **Method.** Each of the 10 cases is extended from 2008-04-01 to 2028-01-01 (80 quarters) with a **capacitance-resistance model (CRM)** for the total liquid rate, split into oil and water by the **water-oil-ratio trend**.
  - The CRM is driven by the water injection rate, held at its 2008 value.
  - The water-oil-ratio trend is ln(WOR) linear in cumulative oil, moved through the latest WOR.
  - Gas = GOR × oil.
- **Why this method.** It won a backtest on all 100 simulated cases and on the real field: the last 1-3 years were hidden and forecast. Median error of the produced volume: **2.9 %** on the cases, 4.0 % on the field.
  - Arps decline + WOR: 4.5 %. Liquid trend + WOR: 5.4 %.
  - Elton's curve families: 64 %. They cannot be used for the forecast.
- **Result (10 cases, P10 / P50 / P90 on 2028-01-01).**
  - Oil: **60.7 / 61.1 / 61.7 MMSTB** (38.6 MMSTB in 2008).
  - Water: 82.5 / 82.9 / 83.1 MMSTB.
  - Gas: 22.1 / 22.2 / 22.4 million MSCF.
  - Oil rate: about 1,700 STB/d in 2028, at a water cut of 0.87.
- **Deliverable.** The 09 template filled for Case 1 to Case 10 (history-matching rank order), 80 quarters each. The Date column is left untouched. The file was read back and validated.
- **Main caution: the start point.** The 10 cases match the observed *cumulative* production, but they end 2008 with oil rates about 12-15 % above the field's (5,900-6,050 against 5,247 STB/d), and water rates about 10 % below (6,980-7,040 against 7,782 STB/d).
  - Starting each forecast from the observed 2008 values instead gives 3.9 % less oil and 2.9 % more water in 2028.
  - A data-only forecast of the field itself gives 54.2 MMSTB of oil. The team's main forecast keeps each case's own start, as decided.

## 1. The problem and the organiser's instructions

The organiser asks for a forecast of field production over the next 20 years, with the "best 10 forecast cases" written into the template `09 Template Deliverable.xlsx` (kick-off slides 26, 31). Each of its 10 sheets has:
- **Columns:** Date | Gas production cumulative [MSCF] | Oil production cumulative [STB] | Water production cumulative [STB];
- **Rows:** 80 quarterly dates from 04/01/2008 to 01/01/2028.

Rock properties are static (slide 30), and there are no new wells after 2008 (team confirmation).

**The difficulty.** The simulator runs we were given stop at 2008-01-01, so no simulated or observed data exists after that date. The proxy maps parameters to the 41 dates it was trained on, so it cannot forecast. Each history-matched case therefore has to be extended by a forecasting method fitted to its own late history. That method must be chosen on evidence: the backtest.

## 2. Methods considered

| Code | Method | How it works | Source |
|---|---|---|---|
| `elton` | Cumulative curve families (Elton's step 4) | Arps-type cumulative curve (oil) and Gompertz curve (water) fitted to the whole 1998-2008 cumulative curve, then shifted to be continuous | `Elton Model/forecast_pipeline.py` |
| `arps_wor` | Decline curve + WOR trend | Arps hyperbolic decline of the oil rate fitted on the last *n* quarters, b in [0, 1], switching to exponential at a terminal decline of 5 %/yr (modified hyperbolic). Water = WOR(Np) × oil | Arps (1945); terminal-decline remedy attributed to Robertson (1988) [2, 3]; WOR trend [4-6] |
| `liq_wor` | Liquid trend + WOR split | Total liquid rate with an exponential, non-increasing trend; oil = liquid / (1 + WOR), WOR from the trend | Water-cut / fractional-flow extrapolation [4, 5] |
| `crm_wor` | **CRM + WOR split** | Field-level CRM ("tank"): q_L(t) relaxes towards G × I with a time constant tau, i.e. q_L,k = q_L,k-1 e^(-dt/tau) + (1 - e^(-dt/tau)) G I_k. tau and G are fitted on the last *n* quarters; injection I is held at 9,327 STB/d after 2008. Split by the WOR trend | Sayarpour et al. (2007, 2009) [7]; CRM + fractional-flow split [8] |

**Not used: extending the proxy or a recurrent network in time.** There is no training data after 2008. Si Jie's recurrent networks, which can run past 2008, flattened their oil decline after 2010 (`Si Jie's model/summary.md`, limitation 1). The organisers' reference (Zhang et al. 2023) uses its LSTM only within the simulated period.

**Shared rules.**
- **Rates.** Quarter means, (cumulative - cumulative a quarter earlier) / days, the same for every curve.
- **Gas.** GOR × oil, with the GOR of the case's last 4 quarters (about 0.3633 in every case and in the field).
- **WOR anchoring.** `fit` uses the least-squares line. `last` keeps the fitted slope but moves the line through the last 2 quarters' WOR. Over 2002-2008 the field's ln(WOR) is linear in cumulative oil (r = 0.998, slope 0.10-0.13 per MMSTB), but a line fitted over a window lags behind the latest data.
- **Physical constraints.** Cumulatives only increase; the oil rate never increases (no new wells); water cut < 1; each forecast starts at the case's own 2008-01-01 cumulative (no jump).

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

## 4. Forecast results

| Case | Source | Oil 2008 → 2028 [MMSTB] | Water 2008 → 2028 [MMSTB] | Gas 2028 [MM MSCF] | Oil rate 2028 [STB/d] | Water cut 2028 |
|---|---|---|---|---|---|---|
| 1 | simulated case 6 | 38.62 → 61.07 | 12.54 → 82.88 | 22.19 | 1,686 | 0.867 |
| 2 | new set | 38.60 → 60.86 | 12.56 → 83.08 | 22.11 | 1,667 | 0.869 |
| 3 | new set | 38.59 → 60.70 | 12.54 → 83.06 | 22.05 | 1,652 | 0.870 |
| 4 | new set | 38.61 → 60.82 | 12.53 → 82.97 | 22.09 | 1,661 | 0.869 |
| 5 | new set | 38.59 → 60.71 | 12.55 → 83.10 | 22.05 | 1,652 | 0.870 |
| 6 | new set | 38.64 → 61.12 | 12.56 → 82.98 | 22.20 | 1,688 | 0.867 |
| 7 | simulated case 97 | 38.62 → 61.42 | 12.58 → 82.76 | 22.31 | 1,726 | 0.864 |
| 8 | simulated case 76 | 38.71 → 61.46 | 12.48 → 82.69 | 22.31 | 1,720 | 0.865 |
| 9 | simulated case 11 | 38.72 → 61.71 | 12.48 → 82.46 | 22.42 | 1,744 | 0.863 |
| 10 | simulated case 40 | 38.72 → 61.69 | 12.48 → 82.49 | 22.41 | 1,739 | 0.863 |

**Uncertainty** (cumulative on 2028-01-01):

| Set | Oil P10 / P50 / P90 [MMSTB] | Water [MMSTB] | Gas [MM MSCF] |
|---|---|---|---|
| The 10 cases | 60.70 / 61.09 / 61.69 | 82.48 / 82.92 / 83.08 | 22.05 / 22.19 / 22.41 |
| Posterior (200 parameter sets from the MCMC) | 60.18 / 60.45 / 60.83 | 82.81 / 83.04 / 83.31 | 21.86 / 21.96 / 22.10 |
| Field reference (the same method on the observed history, no model) | 54.16 | 90.01 | 19.68 |

**The spread between the 10 is narrow (about 1 %).** All 10 match the same history and end 2008 with nearly the same rates, so the forecast method carries them along similar paths. The posterior band is similarly narrow. Neither band includes the uncertainty of the method or of the 2008 start (section 5). These are larger, so **the real uncertainty is wider than these bands.**

Figures: `forecast_fan.png` (cumulatives and rates, 1998-2028; the 10 cases, the posterior band, the field reference).

## 5. Does the start point matter? (team decision: tested)

| Start | Oil 2028, mean [MMSTB] | Water 2028 | Gas 2028 | Change vs main (oil / water) |
|---|---|---|---|---|
| **Main: each case's own 2008 values** | 61.16 | 82.85 | 22.21 | - |
| A1: observed 2008 cumulative (shift only) | 60.89 | 83.13 | 22.12 | -0.4 % / +0.3 % |
| A2: observed 2008 cumulative **and** rates (case supplies the shape) | 58.74 | 85.29 | 21.34 | **-4.0 % / +2.9 %** |
| Field reference (no model) | 54.16 | 90.01 | 19.68 | -11.4 % / +8.6 % |

- **A1 makes little difference.** The cumulatives already match within 1-2 %.
- **A2 makes a real difference.** The cases' 2008 oil rates are 12-15 % above the field's and their water rates about 10 % below. That gap carries forward for 20 years.
- **Reading.** The main forecast is the consistent choice: each sheet is one model continued from its own state. The gap at 2008 is a property of the simulator family (`hist_mat_summary.md`, limitation 1), not of the forecast.
  - If the organisers score the forecast against the simulator continued with the true parameters, the main forecast is the right one.
  - If they score it against the real field's future, A2 and the field reference suggest the main forecast is high on oil and low on water.
  - We report the main forecast in the deliverable and give A2 as a sensitivity.

Figure: `forecast_output/figs/alternative_start.png`. Table: `alternative_start_comparison.csv`.

## 6. What was done to improve accuracy

1. **Method chosen by evidence,** on 100 simulated cases and the real field at 3 cut-offs, instead of one split on the cases only.
2. **Settings tuned by the backtest:** fit window and WOR anchoring. Both winners use the late history only (12 quarters) and follow the latest WOR.
3. **The method uses more of the data.** The CRM uses the water injection record, which the decline methods ignore. It cut the error by about a third against the best decline method.
4. **Physical constraints:** continuity at 2008, monotone cumulatives, no oil-rate increase, water cut < 1, fixed GOR.
5. **Ensembling tested.** Averaging the two best methods was worse, so it was not used.
6. **Uncertainty carried from the history match:** a band from 200 posterior parameter sets.

## 7. Limitations

1. **Horizon.** The backtest covers at most 12 quarters; the forecast covers 80. The backtest is evidence for the method, not proof of 20-year accuracy.
2. **Operating assumptions.** Water injection is constant at 9,327 STB/d after 2008; no new wells; no change of operating conditions. A change in injection changes the CRM liquid rate directly.
3. **Injection in the simulated cases.** The backtest on the simulated cases uses the field's injection record, assuming the cases follow the field's injection schedule. The cases' files do not report injection.
4. **The 2008 rate gap** (section 5).
5. **New sets.** Their 1998-2008 curves come from the proxy. Their quarterly rates carry the proxy's noise (about ±5 % between quarters), which the 12-quarter fit smooths.
6. **The bands are narrow** because they leave out the method and start-point uncertainty.
7. **Sources.** Several methods are cited through secondary sources (documentation, reviews, abstracts). The original papers were not all read.

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
| `forecast_output/10_Final_deliverables.xlsx` | **The deliverable**: the 09 template, Case 1-10, 80 quarters of gas, oil and water cumulative |
| `forecast_output/forecast_cases_long.csv` | History and forecast of every case and of the observed field: date, phase, cumulative, rate (for Power BI) |
| `forecast_output/forecast_bands.csv` | P10 / P50 / P90 by date: the 10 cases, the posterior, the field reference |
| `forecast_output/forecast_summary_by_case.csv` | 2008 and 2028 values per case |
| `forecast_output/backtest_all_settings.csv`, `backtest_best_per_method.csv` | All backtest runs and the best setting per method |
| `forecast_output/alternative_start_comparison.csv` | Main forecast against the two alternative starts |
| `forecast_output/forecast_run_summary.json` | Key numbers of the run |
| `forecast_output/figs/` | `backtest.png`, `forecast_fan.png`, `alternative_start.png` |
