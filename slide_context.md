**Slide context: Team What, ExxonMobil DataWorks Challenge 2026 (DWC26)**

Prepared 2026-10-17 and re-checked 2026-10-18 against the repository at commit `ad390b1` ("check b4 submission"), plus the uncommitted documentation fixes of 17-18 Oct (`references.md`, `summary_hist_mat.md`, `proxy_output/README.md`). This file is shared by the whole team as the single source for the slides, the video and the Power BI dashboard. Every number was checked against a saved output file, except those marked "team estimate" or "team decision"; sources are in brackets. Anything not found in the repo is marked [NOT FOUND - please fill in]; none remain.

> **Team decisions (2026-10-17).**
> - **Use:** every number and figure in this file may be used in the slides and in the video.
> - **Headline oil number:** quote the **mean** of the 10 cases, **58.79 MMSTB (≈58.8)** of cumulative oil on 2028-01-01 [source: forecast_output/forecast_run_summary.json, oil_2028_MM.mean]. Note: README.md and the summary write-ups quote "58.7", which is the rounded P50 (58.73). Use 58.8 on slides, in the video and on the dashboard so that all three match.
> - **Audience:** the challenge judges only. Assume they know reservoir engineering, so keep the technical terms, but define each acronym once.
> - **New parameter sets:** the team is comfortable presenting 4 of the 10 cases as new parameter sets with no simulator run (their 1998-2008 curves come from the proxy). Say so openly on the slide and give the justification from `summary_hist_mat.md`, section 8:
>   - the proxy error is small (test: oil 0.06 %, water 0.14 %) and is part of the likelihood;
>   - the recovery test found the known answers (40 of 40);
>   - a new set never displaces an equally good simulated case.
> - **Framing:** keep the slides technical, with no economics (oil price, NPV). Do not claim a speed-up multiple over the simulator, because its run time is not known.

---

**0. Project Snapshot**

| Item | Detail |
|---|---|
| Title (approved 2026-10-18) | **"From 100 Reservoir Simulations to a 20-Year Production Forecast: Machine Learning-Assisted History Matching"**. Competition topic: Machine Learning-Assisted History Matching and the Challenge of Production Forecasting [source: instruction/instruction1.md] |
| One-sentence summary | We trained a fast neural-network copy of a reservoir simulator, used it to find the rock parameters that reproduce the field's 1998-2008 production, and forecast the 10 best-matching cases to 2028 [source: README.md] |
| Team | "Team What": Elton, Si Jie, Yi Siang and Zi Yang [source: README.md; our_work/README.md; git log] |
| Collaboration | Roles, who did what, how the team decided, and meeting snapshots: see section 12 |
| Timeline | Repo set up 2026-10-08; last commit 2026-10-16 [source: git log]. Deadlines: code 16 Oct 2026; documents, Power BI and video 20 Oct 2026 [source: README.md; instruction/instruction2_briefing_slides.md, slide 34] |
| Tech stack | Python 3.12, numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, PyTorch (CPU), emcee (MCMC sampler) [source: README.md]. Power BI for the dashboard: version 2.154.1260.0 (64-bit, May 2026) or earlier, with the Excel deliverable as its source (required by the brief) [source: instruction/instruction1.md] |
| Final deliverable | `forecast_output/10_Final_deliverables.xlsx`: 10 sheets (Case 1-10), 80 quarterly rows each, 2008-04-01 to 2028-01-01 [verified by reading the file] |

---

**1. Problem Statement**

Reservoir engineers use history matching to align reservoir simulations with the field's historical production. The resulting models are the foundation for production forecasts that drive critical investment and business decisions [source: instruction/instruction2_briefing_slides.md, slide 25]. History matching needs many runs of numerical simulators, so it is time-consuming and computationally expensive [same source]. In this challenge only 100 pre-run simulations exist and we cannot run the simulator at all [source: methodology_explained.md, section 1.1]. The organiser states the goal: forecast production "so that we can evaluate if the field will make money" [source: briefing slide 30].

**Core question:** Which values of the 4 uncertain rock parameters reproduce the field's 1998-2008 production, and what will the field produce to 2028?

---

**2. Executive Overview** (≤150 words)

The organiser gave us 100 simulator runs, each with different values of 4 uncertain rock parameters, and 10 years of real field production. We built a neural-network "proxy" that reproduces a simulator run in 0.5 ms with about 0.06 % (oil) and 0.14 % (water) error on unseen cases. Using Bayesian sampling on this proxy, we found which parameter values fit the history and chose the 10 best-matching cases, all within the proxy's own error of the best match; all 13 automated checks pass. A tank-type forecasting model, chosen by backtesting, extends them to 2028: about **58.8 million barrels of cumulative oil (mean of the 10; 38.4 in 2008)**, with an honest range of **54.2-61.7**. Main takeaway: the history pins down porosity, but the simulator family cannot reproduce the field's late water rise, so model-structure uncertainty, not parameter uncertainty, dominates the forecast.
[sources: README.md; summary_hist_mat.md; forecast_output/forecast_run_summary.json]

---

**3. Background and Context**

- **Domain terms.**
  - *Reservoir simulator:* physics software that predicts oil, gas and water production from a 3D rock model [source: briefing slide 24].
  - *Case:* one simulator run with one set of the 4 parameters. *History:* the field's observed production [source: data_after_engineering/README.md].
  - *History matching (HM):* finding parameter values whose simulated production agrees with the history [source: methodology_explained.md, section 2.1].
  - *Proxy:* a machine-learning model trained to imitate the simulator, so that thousands of parameter sets can be tested quickly [source: methodology_explained.md, section 1.1].
  - *Cumulative:* total produced up to a date. Units: STB (stock-tank barrels), MMSTB (million STB), MSCF (thousand standard cubic feet of gas).
- **The 4 uncertain parameters** [source: methodology_explained.md, section 0]: porosity multiplier (how much oil the rock holds), permeability multiplier (how easily fluid flows), fault transmissibility (how easily fluid crosses the fault), aquifer pore volume (size of the water body pushing in).
- **Prior approach (organisers' reference):** Zhang et al. 2023, SPE Journal: an LSTM (recurrent neural network) proxy plus Bayesian MCMC [source: references.md]. Its LSTM is used only inside the simulated period, so it does not solve the 20-year forecast on its own [source: summary_forecast.md, section 2].
- **Constraints** [source: faq.md; summary_forecast.md, section 1; instruction/instruction1.md]:
  - Only the provided data may be used (FAQ 3).
  - No simulator runs exist after 2008-01-01, so the proxy cannot forecast; a separate forecasting method is required.
  - Future operations are assumed unchanged (FAQ 2): water injection held at 9,327 STB/d, no new wells.
  - Submit exactly 10 forecast cases, chosen by history-match quality first (FAQ 7-9), from the "full population" (FAQ 10).
  - Evaluation is at field level (FAQ 11). Judging criteria: all 4 deliverables on time; accuracy of model output (based on output results and code review); submission quality (business understanding, technical understanding, communication and storytelling, insights, innovation, collaboration, bonus) [source: briefing slide 35].

---

**4. Study Objectives**

Main objective (organiser): build a proxy, find the parameters that minimise the mismatch with the history, and forecast field production over the next 20 years [source: briefing slide 26], submitted as the best 10 forecast cases [source: briefing slide 31; faq.md, FAQ 7].

| # | Sub-objective | Success criterion used in the project | Source |
|---|---|---|---|
| 1 | Build an accurate, fast proxy | Proxy error far below the HM misfit; ranks cases like the simulator (Spearman rank correlation) | summary_hist_mat.md, section 1 |
| 2 | Improve the proxy only if the gain is real | Agreed rule: validation water error ≥10 % lower, oil not higher; then a paired significance test | summary_hist_mat.md, section 7 |
| 3 | Find parameter sets that fit the history and choose 10 | 13 automated checks (convergence, independent cross-check, recovery of known answers, physical realism) | hm_run_summary_real.json, "checks" |
| 4 | Choose the forecast method on evidence | Lowest backtest error on 100 cases and the real field | summary_forecast.md, section 3 |
| 5 | Forecast to 2028 with honest uncertainty | Physical checks pass; uncertainty includes start-point and model-structure error | forecast_run_summary.json, "checks"; summary_forecast.md, section 4 |

---

**5. Methodological Approach**

**Data** [source: data_after_engineering/README.md; data_desc_engineered.md]
- 9 organiser Excel files (up to 330 MB each), converted once into clean CSV/NumPy files by notebook 1.
- 100 cases: train 1-70, validation 71-85, test 86-100. Each case: 4 parameters + a 3D grid of 42,512 active cells.
- History: quarterly, 1998-01-01 to 2008-01-01 (41 dates). The first 4 dates (1998) have zero cumulative production, which is valid data (FAQ 12-13); the matching window is 1999-01-01 to 2008-01-01 (37 dates) [source: summary_hist_mat.md, sections 2 and 8].
- Observed 2008 cumulatives: oil 38.38 MMSTB, water 12.81 MMSTB, gas 13.94 million MSCF [source: hist_match_input/proxy_meta.json].
- Forecast horizon: 80 quarters, 2008-04-01 to 2028-01-01 [source: data_after_engineering/forecast_dates.csv].

**Preprocessing and features**
- Cumulatives are matched instead of rates, because cases and history report rates differently [source: data_after_engineering/README.md].
- Inputs: 4 parameters + 8 grid summary features (config `p4g8`); a linear "grid mapper" computes the 8 features for new parameter sets [source: summary_hist_mat.md, section 1].
- Gas is rebuilt as GOR × oil (gas-oil ratio ≈ 0.3633) [source: summary_hist_mat.md, section 1].

**Models and why** [source: summary_hist_mat.md; summary_forecast.md]

| Stage | Technique | Why chosen |
|---|---|---|
| Proxy | MLP (multi-layer perceptron, 2×128 units) predicting the deviation from the mean curve; ensemble of 20 seeds; learning rate 3e-3, weight decay 1e-5 | The team chose Elton's proxy over Si Jie's LSTM [source: our_work/README.md], then improved it after testing 8 variants. The repo has no head-to-head test of the two on the same data |
| History matching | Bayesian MCMC (Markov-chain Monte Carlo, emcee: 4 ensembles × 16 walkers × 12,000 steps); Gaussian likelihood with AR(1) correlated errors and proxy error added | Follows the organisers' reference paper; gives the range of plausible parameters, not just one answer |
| Choice of 10 | "Quality first": keep candidates within eps (the proxy error, 0.111 %) of the best score; simulated cases first, then new posterior sets by score | Organiser FAQ 8-9 |
| Forecast | `crm_wor`: capacitance-resistance model (CRM, a "tank" model where liquid output follows water injection) + water-oil-ratio (WOR) trend to split oil and water | Won the backtest against 3 alternatives |
| Forecast start | "A2": start from the field's observed 2008 cumulatives and rates; the case supplies only the curve shape | Won a blind hindcast |

*Selection protocol (fixed before the tests):*
1. **Metric, and why it is used.** J = RMS of the oil and water errors in HM units (% of the final observed 2008 cumulative, over the 37 producing dates). It is the same quantity the history matching scores, so lowering J directly lowers the proxy error term `eps` in the likelihood. Gas is excluded (gas ≈ 0.3633 × oil, so it would count oil twice and hide water, the hard phase); rates are excluded because cases and history report rates under different conventions. The symmetric square-root-of-the-mean form stops a variant from winning by trading one phase for the other, and the rule adds the explicit "oil not worse" guard.
2. **Choose on cases 1-85 only.** 5-fold CV on the 70 train cases plus the 15 validation cases; test cases 86-100 are reserved for confirmation.
3. **Success rule (agreed).** Validation water at least 10 % lower than the baseline and validation oil not higher. **Consistency check (added after the first run):** CV J must not be higher than the baseline's either — the GP looked good on validation but was much worse in CV.
4. **Staged confirmation.** The runner-up (V4) was checked by out-of-fold error over all 85 cases (a larger sample than the 15 validation cases), then once on the test set (an openly stated second use of the test cases).
5. **Adoption.** V4 with 20 seeds, trained on cases 1-70: passes the rule (validation water −30 %, oil −1 %), beats the baseline by ≈20 % out-of-fold (p ≈ 1e-9) and on the test set (water −41 %, oil −10 %, J −38 %). The 5- and 20-seed versions are practically equivalent (test J 0.106 vs 0.110 %); 20 seeds was adopted because 5 seeds missed the strict oil clause by 1 %, within the noise of 15 cases.
6. **Lesson.** 15 validation cases are a small sample: the GP average looked best there and failed the test. The decision therefore rested on CV/out-of-fold results over more cases. Because the test cases were used three times in the project, the test p-values are optimistic; the out-of-fold result does not use them.

**Final proxy model selected.** V4 with a 20-seed ensemble — the "improved cumulative proxy" (adopted 15 Oct 2026). An MLP (2 × 128, SiLU, dropout 0.05) takes 12 inputs (4 parameters + 8 grid features) and predicts oil and water on the 37 producing dates as deviations from the mean train curve; gas is rebuilt as GOR × oil (≈ 0.3633) and the 4 pre-production dates are 0. Training: full-batch Adam, lr 3e-3, weight decay 1e-5, cosine decay, 1,145 epochs (CV median), cases 1-70 only; 20 seeds, averaged and clipped at 0. Accuracy in HM units: validation oil 0.074 %, water 0.138 %; test oil 0.064 %, water 0.142 %; ranks cases exactly like the simulator (Spearman 1.000); `eps` = 0.111 % (v1: 0.149 %); 0.5 ms per call with all 20 networks batched. It ships as `hist_match_input/cumulative_model.pt` for notebook 3 and supplies the curves of the 4 new parameter sets in the final 10; the rate proxy is comparison only [source: summary_hist_mat.md, section 1; notebook2_proxy_model.ipynb, section 6b].

**Validation strategy**
- Proxy: 5-fold cross-validation (CV) on cases 1-70, validation set 71-85, test set 86-100; out-of-fold check on 85 cases with Wilcoxon signed-rank test [source: summary_hist_mat.md, section 7].
- HM: R-hat convergence, independent direct sampling, implausibility screening, hold-out of the late history, recovery test on 5 known cases [source: summary_hist_mat.md, sections 3-4].
- Forecast: backtest at cut-offs 2005, 2006, 2007 on 100 cases and the field; blind hindcast of start options with a rule fixed before the run [source: summary_forecast.md, sections 3, 5].

**Metrics**
- HM score: NRMSE (normalised root-mean-square error) of oil and water cumulatives, as % of the observed 2008 cumulative [source: summary_hist_mat.md, intro].
- Forecast error: relative error of the volume produced after the cut-off [source: summary_forecast.md, section 3].

**Pipeline (for a flow diagram)** [source: README.md]
1. Raw Excel (9 files) → notebook 1: clean data.
2. Notebook 2: train the proxy (parameters → production curves).
3. Notebook 3: MCMC on the proxy → posterior → choose the 10 best cases.
4. Notebook 4: backtest forecast methods → forecast the 10 to 2028 from the observed 2008 start.
5. Output: `10_Final_deliverables.xlsx` → Power BI dashboard.

---

**6. Key Findings and Results**

**Headline metrics**

| Metric | Value | Baseline / comparison | Source |
|---|---|---|---|
| Proxy test error, oil (% of final observed cumulative) | 0.064 % | Elton's original: 0.071 % | proxy_output/proxy_summary.json |
| Proxy test error, water | 0.142 % | Elton's original: 0.240 % (−41 %) | proxy_output/proxy_summary.json |
| Proxy out-of-fold error reduction (85 cases) | −21 % oil, −20 % water; p = 1e-9 / 6e-9 | Elton's original | improve_proxy/results/significance_summary.csv |
| Proxy ranking agreement with simulator (Spearman) | 1.000 (validation, test) | - | proxy_output/proxy_summary.json |
| Rate proxy, test RMSE (Elton's step-1 model, for comparison only; not used in the history match) | Oil 41.2 STB/d, water 35.7 STB/d, gas 13.9 MSCF/d | Ridge regression: 190.8 / 119.5 / 68.4. Mean curve: 1,015.9 / 762.1 / 369.0 | proxy_output/metrics_rate.csv (test rows); units from data_desc_engineered.md |
| Proxy speed | 0.5 ms per call (20 networks) | Simulator run time not known, so no speed-up multiple is claimed | summary_hist_mat.md, section 1 |
| Full pipeline run time (notebooks 1-4) | About 30-40 min | - | Team estimate, 2026-10-18 (not recorded in the repo). The repo gives about 15 min for the first run of notebook 1 and about 2 min for notebook 4 [README.md; forecast_output/README.md] |
| History-match score of the 10 chosen (lower is better) | 0.491-0.544 % | Median of 100 cases: 6.58 % | hm_run_summary_real.json |
| MCMC convergence (max R-hat; <1.1 = converged) | 1.024 | - | hm_run_summary_real.json |
| Recovery test: true parameters inside 5-95 % interval | 40 of 40 | - | hm_details_real.xlsx, sheet recovery_test |
| Automated checks passed | 13 of 13 | - | hm_run_summary_real.json |
| Forecast method backtest error | 2.94 % (crm_wor) | Arps + WOR 4.47 %; liquid + WOR 5.44 %; Elton's 64.2 % | forecast_run_summary.json |
| Start-point hindcast error (pseudo-fields / real field) | 3.52 % / 3.80 % (A2) | Own start: 7.47 % / 6.54 %; p = 5e-37 | summary_forecast.md, section 5 |
| **Oil 2028, mean of the 10 cases (headline)** | **58.79 MMSTB** | 38.38 MMSTB in 2008 | forecast_run_summary.json |
| Oil 2028, 10 cases (P10 / P50 / P90) | 58.58 / 58.73 / 59.00 MMSTB | - | summary_forecast.md, section 4 |
| Oil 2028, wider range | 54.16-61.71 MMSTB | - | forecast_run_summary.json |
| Water 2028, 10 cases (P10 / P50 / P90); wider range | 85.04 / 85.27 / 85.53; 81.83-90.01 MMSTB | 12.81 MMSTB in 2008 | summary_forecast.md, section 4 |
| Oil rate and water cut, 2028 | ≈1,570-1,630 STB/d; 0.872-0.876 | 2008: 5,247 STB/d | forecast_output/forecast_by_case.csv |

**Key findings**
1. **The proxy is accurate enough to replace the simulator for matching.** Its test errors (0.06 % oil, 0.14 % water) are far below the best-case misfit (≈0.49 %), and it orders cases exactly as the simulator does (Spearman 1.000) [source: proxy_summary.json; summary_hist_mat.md].
2. **The history fixes porosity but leaves other parameters open.** Porosity multiplier 1.366 (5-95 %: 1.342-1.383), posterior spread 6 % of prior; permeability and aquifer partly constrained; fault transmissibility not constrained (spread 99 % of prior) [source: hm_details_real.xlsx, sheet posterior].
3. **The 10 chosen are statistically tied.** 6 simulated cases (6, 97, 76, 11, 40, 73) + 4 new parameter sets; scores span 0.053 %, less than the proxy error eps = 0.111 % [source: hm_run_summary_real.json; summary_hist_mat.md, section 3].
4. **The simulator family cannot reproduce the field's late water rise.** In the last quarter of the history (to 2008-01-01) the 10 cases produce oil 11-15 % too fast (5,844-6,051 vs 5,247 STB/d) and water about 10 % too slowly (6,943-7,043 vs 7,782 STB/d). The field's WOR slope (0.130/MMSTB) is ≈1.7× the cases' (0.074-0.078) [source: recomputed from hist_match_output/top10_series_real.csv and history_observed.csv; matches summary_forecast.md, section 5; forecast_by_case.csv].
   - Interpretation: whatever controls water breakthrough (e.g. relative permeability) is probably not among the 4 uncertain parameters, as the team's own write-up suggests.
5. **Starting the forecast from the observed 2008 state halves the error.** Blind hindcast: 3.52 % vs 7.47 % on 100 pseudo-fields (better in 83 % of pairs, p = 5e-37) and 3.80 % vs 6.54 % on the real field. It lowers the mean 2028 oil of the 10 from 61.22 to 58.79 MMSTB [source: summary_forecast.md, section 5; forecast_run_summary.json].

**Comparisons**

| Way of choosing 10 (same proxy) | Mean score | Worst score | Outside train range | Source |
|---|---|---|---|---|
| Quality first (adopted) | 0.513 % | 0.544 % | 0 | hm_details_real.xlsx, selection_comparison |
| Hybrid k-means (v3) | 0.512 % | 0.537 % | 0 | same |
| Simulated cases only | 0.626 % | 0.925 % (4 ruled out) | 0 | same |
| Elton's 10 | 0.531 % | 0.572 % | 9 | same |
| Zi Yang's 10 | 0.513 % | 0.520 % | 0 | same |

**Surprising or negative results**
- The best-looking proxy variant on validation (MLP + Gaussian process average, validation J 0.063 %) failed the single test: test oil error 0.30 % vs 0.07 % for the baseline. It was rejected; 15 validation cases were too few to reveal the problem [source: summary_hist_mat.md, section 7].
- Elton's original forecast fails the backtest (≈64 % error: oil ≈60 % too high, water ≈65 % too low) [source: forecast_run_summary.json].
- Averaging the two best forecast methods was worse (3.48 %) than CRM alone (2.94 %) [source: summary_forecast.md, section 3].
- The 10 cases agree within ≈1 % on 2028 oil, which understates the real uncertainty; in the hindcast, the blind matches' range never contained the field's water (0 of 3 cut-offs) [source: summary_forecast.md, section 4].
- The improved proxy changed the history-match outcome only slightly (eps 0.149 % → 0.111 %; porosity and best case unchanged) [source: summary_hist_mat.md, section 7].

---

**7. Limitations and Assumptions**

- **Model structure.** No parameter set in the design range fits both the 2008 totals and the 2008 rates; the forecast fixes the 2008 level but inherits the cases' slower water rise, so it is probably high on oil and low on water later [source: summary_forecast.md, section 7.4].
- **Horizon.** Backtest and hindcast cover at most 12 quarters; the forecast covers 80. They support the method, not 20-year accuracy [source: summary_forecast.md, section 7.1].
- **Operating assumptions.** Injection constant at 9,327 STB/d, no new wells, rock properties static [source: summary_forecast.md, section 7.2; briefing slide 30].
- **Simulated-case injection.** The backtest assumes the cases follow the field's injection schedule; case files do not report injection [source: summary_forecast.md, section 7.3].
- **New parameter sets have no simulator run;** their curves come from the proxy [source: summary_hist_mat.md, section 5].
- **Test cases were reused** (three times) in proxy tests, so test p-values are optimistic; the out-of-fold result does not use them [source: summary_hist_mat.md, section 7].
- **Assumed 1 % error** in the implausibility screen is provisional [source: summary_hist_mat.md, section 5].
- **No significance test** for the forecast method choice (unlike the start-point choice) [source: summary_forecast.md, section 7.8].
- **Unknown scoring target:** what the organisers score the forecast against is not known [source: summary_forecast.md, section 7.7].
- **What the results do NOT prove:** that 58.8 MMSTB is the most likely 2028 value, that the order of the 10 is meaningful, or that any economic decision follows (no economics were modelled).

---

**8. Conclusions and Next Steps**

| Objective | Achieved? | Evidence |
|---|---|---|
| 1. Accurate, fast proxy | Yes | Test error 0.06 % oil / 0.14 % water; Spearman 1.000 [proxy_summary.json] |
| 2. Improve only on evidence | Yes | −20 % out-of-fold error, p ≈ 1e-9; failed variant rejected [significance_summary.csv] |
| 3. Fit history, choose 10 | Yes, with a known gap | 13/13 checks pass; late water rise not reproduced [hm_run_summary_real.json] |
| 4. Evidence-based forecast method | Yes | Backtest 2.94 % vs 4.47 % next best [forecast_run_summary.json] |
| 5. Forecast with honest uncertainty | Yes | 58.8 MMSTB (mean), range 54.2-61.7; all physical checks pass [forecast_run_summary.json] |

- **Practical implications.** Interpretation: decision-makers should plan on the 54.2-61.7 MMSTB range, not the 1 % spread of the 10 cases; the low end (field-only trend) reflects the faster water rise the simulator misses.
- **Next steps (from the repo).** Power BI dashboard, documents and video are due 20 Oct [source: summary_hist_mat.md, section 6]. Considered but not done [source: summary_forecast.md, section 6b]:
  - a structured model-error term: it would widen the posterior but not move the forecast;
  - rates in the HM score: no parameter set fits both the 2008 totals and rates, so it would only trade one fit for the other.
- **Further next steps.** Interpretation (not in the repo): add the parameters that control water breakthrough (e.g. relative permeability) to the uncertain set and rerun the simulator on the posterior new sets; add economics to turn the range into a value decision. Confirm before presenting.

---

**9. Visual Assets Inventory**

| File | What it shows | Takeaway it supports |
|---|---|---|
| hist_match_output/figs/fit_top10_real.png | The 10 vs history (oil, water) with the 100 cases in grey; residuals below | The 10 track the history; on 2008-01-01 water is 1.9-2.6 % low and oil 0.1-0.9 % high [top10_cases_real.csv, err_final_*] |
| forecast_output/figs/forecast_fan.png | Cumulatives and rates 1998-2028: 10 cases, posterior band, wider range, field reference | Headline forecast and honest range |
| forecast_output/figs/backtest.png | Backtest error per method and cut-off; field cut at 2006 | CRM + WOR wins; Elton's method fails |
| forecast_output/figs/alternative_start.png | 2028 oil and water per case under 3 start options | Start choice moves oil by ≈4 % |
| hist_match_output/figs/posterior_real.png | Corner plot of the parameter posterior | Porosity fixed; fault not constrained |
| hist_match_output/figs/selection_real.png | The 10 across the not-ruled-out region | How the 10 were chosen |
| hist_match_output/figs/scores_real.png | Score bars, 25 best cases + new sets | The 10 are tied |
| hist_match_output/figs/trace_real.png | MCMC chain traces | Sampler converged |
| hist_match_output/figs/direct_sampling_real.png | MCMC vs direct sampling | Independent check agrees |
| hist_match_output/figs/holdout_real.png | Early vs late (held-out) score | Early match predicts late (Spearman 0.993) |
| hist_match_output/figs/tradeoff_real.png | Oil vs water error (Pareto front) | Oil-water fit trade-off |
| hist_match_output/figs/implausibility_real.png, predictive_real.png | Ruled-out regions; posterior predictive band | Supporting / backup |
| proxy_output/figs/fig5_hm_units.png | **The adopted cumulative proxy.** Left: HM score from proxy curves vs from simulator curves (train, validation, test on the 1:1 line). Right: validation error by date, oil and water | The proxy ranks cases exactly like the simulator; its error stays below 0.12 % (oil) and 0.26 % (water) at every date |
| proxy_output/figs/fig1_pred_vs_truth.png | Rate proxy (Elton's reproduced step-1 model, not the adopted proxy) vs simulator, all 41 dates | Backup only: the network architecture predicts well |
| proxy_output/figs/fig2_curves_test.png (and _validation) | Rate curves of the best, median and worst test (validation) cases, simulator vs rate proxy | Backup only; label it as the rate proxy |
| proxy_output/figs/fig4_rmse_by_date.png, fig0_train_curves.png | Rate-proxy error by date; rate curves of the 70 train cases and their mean | Backup |
| improve_proxy/results/figs/variants_validation.png | Oil and water validation error of 9 proxy variants (V0 baseline to V6), with the water target line | Evidence-based model choice; note the V6 average looks best here but failed the test (section 6) |
| Tables: summary_hist_mat.md (top-10 table), summary_forecast.md (per-case 2028 table, hindcast table) | - | Slide-ready tables |

Notebook 1.1 (EDA) figures exist only inside the notebook, not as saved files [source: file listing].

**Missing visuals to create:** pipeline flow diagram; problem/data overview (100 cases fanning around the history; could crop fit_top10_real.png); simple parameter explainer graphic; proxy before/after bar chart (v1 vs v2 error); hindcast bar chart (own start vs A2); version timeline (v1→v4); team/collaboration slide with space for meeting snapshots (section 12). Power BI screenshots are not used in the slides (team decision).

---

**10. Suggested Slide Storyline**

| # | Conclusion-style title | Key point | Best visual / number |
|---|---|---|---|
| 1 | From 100 Reservoir Simulations to a 20-Year Production Forecast: Machine Learning-Assisted History Matching (title slide) | Title, team | Team What: Elton, Si Jie, Yi Siang, Zi Yang |
| 2 | Investment decisions depend on simulators that are slow to calibrate | Problem and core question | Briefing slides 25-26 |
| 3 | We had 100 simulator runs, 4 uncertain parameters and 10 years of history | Data and constraints | 100 cases fan vs history (crop of fit_top10_real.png) |
| 4 | A four-step pipeline turns 100 runs into a 20-year forecast | Approach overview | Pipeline diagram (to create) |
| 5 | A neural-network proxy imitates the simulator to within 0.06 % oil and 0.14 % water | Proxy accuracy | fig5_hm_units.png; table of errors |
| 6 | We kept a proxy upgrade only after it proved a significant 20 % gain | 8 variants tested under a rule fixed in advance (section 5); 20-seed tuned MLP adopted; the GP average looked best on validation but failed the test | significance_summary.csv; variants_validation.png |
| 7 | The production history pins down porosity but not the fault | Posterior | posterior_real.png; porosity 1.342-1.383 |
| 8 | Ten cases match the history equally well, and all 13 checks pass | The 10 and validation; state openly that 4 are new parameter sets (proxy curves) and why that is defensible | fit_top10_real.png; 0.491-0.544 % vs median 6.58 %; recovery test 40 of 40 |
| 9 | The simulator cannot reproduce the field's late water rise | Model-structure gap | 2008 rates: oil 11-15 % high, water about 10 % low; field WOR slope 1.7× the cases' |
| 10 | A tank model plus water-oil trend forecasts best in backtests | Method choice | backtest.png; 2.94 % vs 4.47 % vs 64 % |
| 11 | Starting from the observed 2008 state halves the forecast error | Hindcast | 3.52 % vs 7.47 %, p = 5e-37; alternative_start.png |
| 12 | The field should reach about 58.8 MMSTB of oil by 2028, within 54.2-61.7 | Headline result (mean of the 10) | forecast_fan.png |
| 13 | The real uncertainty is the model, not the parameters | Limitations | 1 % spread vs wider range |
| 14 | Next: widen the uncertain parameters and link the forecast to value | Conclusions, next steps | Objectives table (section 8) |
| 15 | Four members' work was compared and merged into one tested pipeline | Collaboration (a judging criterion) | Roles table (section 12); version table v1-v4 (README.md); space for meeting snapshots |

---

**11. Open Questions for the Team**

Resolved on 2026-10-17:
- All numbers and figures may be used in the video.
- `references.md` now gives the correct ensemble size (20 networks).
- The headline oil number is the mean, 58.79 MMSTB.
- No team title yet; one is suggested in section 0.
- Members are shown by first name: Elton, Si Jie, Yi Siang, Zi Yang.
- Si Jie creates the dashboard. All four members make the slides and present.
- The audience is the judges only.
- The kick-off date, the judging weightings and Power BI screenshots are left out.
- The team is comfortable presenting the 4 new parameter sets.

Resolved on 2026-10-18:
- The suggested title now names the domain ("Reservoir", "Production").
- Who records, edits and uploads the video is left out of this file.
- Rate-proxy units confirmed from the data: oil and water in STB/d, gas in MSCF/d. Errors are always reported by phase (section 6), never pooled across phases, because the units differ. The pooled figures in the repo (rate 32.5, cumulative 23,097) mix units and must not be quoted; `summary_hist_mat.md` and `proxy_output/README.md` were corrected to per-phase values. The rate proxy is a comparison model only.
- The full pipeline (notebooks 1-4) takes about 30-40 min (team estimate). The simulator run time is not known, so the slides make no "N times faster" claim.
- The slides stay technical; no economics.

- The title is approved: "From 100 Reservoir Simulations to a 20-Year Production Forecast: Machine Learning-Assisted History Matching".
- The team write-ups (`summary_hist_mat.md`, `summary_forecast.md`, `methodology_explained.md`, `improve_forecast/README.md`) were corrected to the current 2008 gaps: oil rate 11-15 % high, cumulative water 1.9-2.6 % low, cumulative oil 0.1-0.9 % high.

Still open: none.

---

**12. Collaboration**

How the work was split [sources: our_work/README.md; README.md; git log]. Each member's original work is kept in `our_work/`; the final pipeline (notebooks 1-4) merges the best parts after head-to-head comparisons.

| Role | Member | Jobs | Evidence in the repo |
|---|---|---|---|
| Data Engineer | Yi Siang | Converted the 9 organiser Excel files (up to 330 MB each) into clean CSV/NumPy files; wrote the data guide for the team | `notebook1_data_eng.ipynb`, `data_after_engineering/`, `data_desc_engineered.md` [git log: 2026-10-08, 2026-10-10] |
| Data Analyst (EDA) | Zi Yang | Exploratory data analysis: data checks, parameter coverage, curves against the history, parameter effects, PCA, which parameters the history can identify | `notebook1.1_eda_commented.ipynb` [git log: ziyanglim29, 2026-10-09, "Document EDA insights in EDA_commented.ipynb"] |
| Data Scientist: proxy model | Elton | Built the proxy the team chose (PyTorch MLP, 4 parameters + 8 grid features, 5-seed ensemble; the final version is a retuned 20-seed retraining of the same network), plus a first history match and forecast (challenge steps 1-4) | `our_work/proxy_model_elton/` |
| Data Scientist: alternative proxy | Si Jie | Built a tested, packaged proxy repository from the raw data comparing several models; froze an LSTM (primary) and GRU (secondary). Kept as a reference after the team chose Elton's proxy | `our_work/proxy_model_sijie/` |
| Data Scientist: history matching | Yi Siang | History-matching drafts v1/v2 (score, likelihood, MCMC, cross-checks), which became notebook 3 | `our_work/history_matching_v1_yisiang.ipynb`, `_v2_` |
| Data Scientist: history matching | Zi Yang | Full alternative pipeline: GP-on-PCA proxy, differential evolution, MCMC, choice of 10. The GP was tested as a proxy variant, and the idea of using posterior sets as candidates was adopted | `our_work/history_matching_ziyang.ipynb`; `improve_proxy/` (variant V6) |
| Integration and pipeline lead | Yi Siang | Merged the team's work into notebooks 2-4; ran the proxy-improvement and forecast-start studies; built the forecast; archived each version (v1-v4); aligned the pipeline with the organiser FAQ | `notebook2-4`, `improve_proxy/`, `improve_forecast/`, `archive/` [git log: 2026-10-15 to 2026-10-16] |
| Dashboard Creator (Power BI) | Si Jie | Build the Power BI dashboard with `10_Final_deliverables.xlsx` as its source (required by the brief); extra tables for history, bands and the wider range are in `forecast_output/` (`forecast_cases_long.csv`, `forecast_bands.csv`). Use oil 58.8 MMSTB (mean) as the headline | Team decision, 2026-10-17 |
| Slides and documentation | All four (Elton, Si Jie, Yi Siang, Zi Yang) | Build the PowerPoint from this file | Team decision, 2026-10-17 |
| Presenters (video, 7 minutes) | All four (Elton, Si Jie, Yi Siang, Zi Yang) | Present in the video | Team decision, 2026-10-17 |

**How the team decided** (useful for the collaboration slide):
- **Proxy:** two proxies were built independently, and the team chose Elton's; Si Jie's is kept as a reference [source: our_work/README.md]. Eight improvement variants were then tested under a success rule agreed in advance [source: summary_hist_mat.md, section 7].
- **Choice of the 10:** Yi Siang's, Zi Yang's and Elton's sets of 10 were all scored with the same proxy (table in section 6) [source: hm_details_real.xlsx, sheet selection_comparison].
- **Forecast:** Elton's forecast was used as the baseline and replaced after it failed the backtest [source: summary_forecast.md, section 3].
- **Versioning:** every change was kept only if it helped significantly, with a restore point for each version (v1-v4) [source: README.md].

**Meeting snapshots (placeholder).** Leave room on the collaboration slide for 2-4 photos or screenshots of team meetings, each with a short caption (date and topic, e.g. "proxy comparison" or "choosing the 10"). The team will attach them; none are in the repo.

| Snapshot | Date | Topic / caption |
|---|---|---|
| [PLACEHOLDER - attach snapshot 1] | [to add] | [to add] |
| [PLACEHOLDER - attach snapshot 2] | [to add] | [to add] |
| [PLACEHOLDER - attach snapshot 3] | [to add] | [to add] |
| [PLACEHOLDER - attach snapshot 4] | [to add] | [to add] |
