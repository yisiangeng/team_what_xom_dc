**Slide context: Team What, ExxonMobil DataWorks Challenge 2026 (DWC26)**

Prepared 2026-10-17 and re-checked 2026-10-18 against the repository at commit `ad390b1` ("check b4 submission"), plus the uncommitted documentation fixes of 17-18 Oct (`references.md`, `summary_hist_mat.md`, `proxy_output/README.md`). All values re-checked on 2026-10-19 against the saved outputs at commit `9bdd640` (later commits change documentation only). This file is shared by the whole team as the single source for the slides, the video and the Power BI dashboard. Every number was checked against a saved output file, except those marked "team estimate" or "team decision"; sources are in brackets. Anything not found in the repo is marked [NOT FOUND - please fill in]; none remain.

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
| Collaboration | Roles, who did what, how the team decided, and meeting snapshots: see section 13 |
| Timeline | Repo set up 2026-10-08; last code commit 2026-10-16; commits on 2026-10-18 change documentation only [source: git log]. Deadlines: code 16 Oct 2026; documents, Power BI and video 20 Oct 2026 [source: README.md; instruction/instruction2_briefing_slides.md, slide 34] |
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
- **How the simulator works (for a diagram).** A reservoir simulation model represents the reservoir as many grid cells, each assigned rock and fluid properties, and solves partial differential equations derived from mass balance to predict fluid flow, pressure and production [source: briefing slide 24]. The organiser does not name the simulator software.

  ```
  Inputs (per grid cell)                                  Simulator                          Outputs (field level)
  - Geologic: porosity ×mult*, permeability ×mult*,       Solves mass-balance flow           Oil / water / gas
    fault transmissibility*, aquifer volume*         →    equations (oil, water, gas)   →    rates + cumulatives,
  - Fluid properties: viscosity, density, ... (fixed)     in 42,512 active cells,            quarterly 1998-2008
  - Rock-fluid: relative permeability kr,                 stepping through time              (41 dates)
    capillary pressure Pc (fixed)
  - Wells: field water injection schedule
                    * = the 4 uncertain parameters, varied across the 100 runs
  ```

  - *Input groups:* slide 24 sorts the inputs into "geologic" properties (porosity, permeability), fluid properties (viscosity, density, formation volume factor, solution gas ratio) and displacement properties that depend on rock-fluid interaction (relative permeability kr and capillary pressure Pc, functions of saturation) [source: briefing slide 24].
  - *Where our parameters sit:* all 4 uncertain parameters are geologic or aquifer properties; the fluid and rock-fluid properties are the same in all 100 runs. Interpretation: this is why the water-rise gap (section 5.ii, finding 4) cannot be removed by the 4 parameters; water breakthrough is governed largely by relative permeability, which is fixed.
  - *Why a proxy:* one run solves the equations in every cell at every time step, and history matching needs thousands of runs; only 100 pre-run cases exist and the simulator cannot be run [source: methodology_explained.md, section 1.1]. Do not quote a run time or speed-up: the simulator's run time is not known (team decision).
  - *Why a separate forecast:* the runs stop at 2008-01-01, so no simulated production exists after that date [source: summary_forecast.md, section 1].
  - *The reference paper* (Zhang et al. 2023) models a different reservoir (the SPE10 Tarbert model in CMG-IMEX, with 10 uncertain parameters including relative-permeability terms). Use it only for the history-matching loop (its Fig. 2: prior → proxy → MCMC → check), not as a description of our simulator [source: reference/robust-history-matching-bayesian-lstm.md].
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

The work has three stages, each in its own notebook: **(i) a proxy model** that imitates the simulator, **(ii) history matching** on that proxy, and **(iii) a forecast** of the 10 chosen cases to 2028. Each part below gives the stage's methodological approach, key findings and results. Section 6 collects the headline numbers of all three stages.

**Shared inputs** [source: data_after_engineering/README.md; data_desc_engineered.md]
- 9 organiser Excel files (up to 330 MB each), converted once into clean CSV/NumPy files by notebook 1.
- 100 cases: train 1-70, validation 71-85, test 86-100. Each case: 4 parameters + a 3D grid of 42,512 active cells.
- History: quarterly, 1998-01-01 to 2008-01-01 (41 dates). The first 4 dates (1998) have zero cumulative production, which is valid data (FAQ 12-13); the matching window is 1999-01-01 to 2008-01-01 (37 dates) [source: summary_hist_mat.md, sections 2 and 8].
- Observed 2008 cumulatives: oil 38.38 MMSTB, water 12.81 MMSTB, gas 13.94 million MSCF [source: hist_match_input/proxy_meta.json].
- Forecast horizon: 80 quarters, 2008-04-01 to 2028-01-01 [source: data_after_engineering/forecast_dates.csv].
- Common unit for errors and scores: % of the observed 2008 cumulative ("HM units"). Example: 0.49 % means the curve is, on average, 0.49 % of the final cumulative away from the history [source: summary_hist_mat.md, intro].

**Pipeline (for a flow diagram)** [source: README.md]
1. Raw Excel (9 files) → notebook 1: clean data.
2. Notebook 2 (part i): train the proxy (parameters → production curves).
3. Notebook 3 (part ii): MCMC on the proxy → posterior → choose the 10 best cases.
4. Notebook 4 (part iii): backtest forecast methods → forecast the 10 to 2028 from the observed 2008 start.
5. Output: `10_Final_deliverables.xlsx` → Power BI dashboard.

Full pipeline run time (notebooks 1-4): about 30-40 min [team estimate, 2026-10-18; not recorded in the repo. The repo gives about 15 min for the first run of notebook 1 and about 2 min for notebook 4: README.md; forecast_output/README.md].

---

**5.i Proxy model** (notebook 2; code in `src/proxy_lib.py`)

*Methodological approach*

- **Goal.** Replace the simulator inside the history match: map the 4 parameters to oil and water cumulative curves fast enough to evaluate millions of times.
- **Preprocessing and features.**
  - Cumulatives are matched instead of rates, because cases and history report rates differently [source: data_after_engineering/README.md].
  - Inputs: 4 parameters + 8 grid summary features (config `p4g8`); a linear "grid mapper" computes the 8 features for new parameter sets (within 0.002 sd on 30 held-out cases) [source: summary_hist_mat.md, section 1].
  - Gas is rebuilt as GOR × oil (gas-oil ratio ≈ 0.3633) [source: summary_hist_mat.md, section 1].
- **Model and why.** MLP (multi-layer perceptron, 2×128 units) predicting the deviation from the mean curve; ensemble of 20 seeds. The team chose Elton's proxy over Si Jie's LSTM [source: our_work/README.md], reproduced it exactly, then improved it after testing 8 variants. The repo has no head-to-head test of the two original proxies on the same data.
- **Validation strategy.** 5-fold cross-validation (CV) on cases 1-70, validation set 71-85, test set 86-100; out-of-fold check on 85 cases with a Wilcoxon signed-rank test and a case-bootstrap interval [source: summary_hist_mat.md, section 7].

*Selection protocol (fixed before the tests):*
1. **Metric, and why it is used.** J = RMS of the oil and water errors in HM units (% of the final observed 2008 cumulative, over the 37 producing dates). It is the same quantity the history matching scores, so lowering J directly lowers the proxy error term `eps` in the likelihood. Gas is excluded (gas ≈ 0.3633 × oil, so it would count oil twice and hide water, the hard phase); rates are excluded because cases and history report rates under different conventions. The symmetric square-root-of-the-mean form stops a variant from winning by trading one phase for the other, and the rule adds the explicit "oil not worse" guard.
2. **Choose on cases 1-85 only.** 5-fold CV on the 70 train cases plus the 15 validation cases; test cases 86-100 are reserved for confirmation.
3. **Success rule (agreed).** Validation water at least 10 % lower than the baseline and validation oil not higher. **Consistency check (added after the first run):** CV J must not be higher than the baseline's either, so that a validation result is confirmed on more cases. The GP average still passed on CV J (0.119 % vs 0.180 %); its weakness showed in CV oil (0.096 % vs 0.074 %; the GP alone 0.178 %) and the test then rejected it (item 6) [source: improve_proxy/results/variant_comparison.csv].
4. **Staged confirmation.** The runner-up (V4) was checked by out-of-fold error over all 85 cases (a larger sample than the 15 validation cases), then once on the test set (an openly stated second use of the test cases).
5. **Adoption.** V4 with 20 seeds, trained on cases 1-70: passes the rule (validation water −30 %, oil −1 %), beats the baseline by ≈20 % out-of-fold (p ≈ 1e-9) and on the test set (water −41 %, oil −11 %, J −38 %). The 5- and 20-seed versions are practically equivalent (test J 0.106 vs 0.110 %); 20 seeds was adopted because 5 seeds missed the strict oil clause by 1 %, within the noise of 15 cases.
6. **Lesson.** 15 validation cases are a small sample: the GP average looked best there and failed the test. The decision therefore rested on CV/out-of-fold results over more cases. Because the test cases were used three times in the project, the test p-values are optimistic; the out-of-fold result does not use them.

**Final proxy model selected.** V4 with a 20-seed ensemble — the "improved cumulative proxy" (adopted 15 Oct 2026). An MLP (2 × 128, SiLU, dropout 0.05) takes 12 inputs (4 parameters + 8 grid features) and predicts oil and water on the 37 producing dates as deviations from the mean train curve; gas is rebuilt as GOR × oil (≈ 0.3633) and the 4 pre-production dates are 0. Training: full-batch Adam, lr 3e-3, weight decay 1e-5, cosine decay, 1,145 epochs (CV median), cases 1-70 only; 20 seeds, averaged and clipped at 0. It ships as `hist_match_input/cumulative_model.pt` for notebook 3 and supplies the curves of the 4 new parameter sets in the final 10; the rate proxy is comparison only [source: summary_hist_mat.md, section 1; notebook2_proxy_model.ipynb, section 6b].

*Key findings*
1. **The proxy is accurate enough to replace the simulator for matching.** Its test errors (0.06 % oil, 0.14 % water) are far below the best-case misfit (≈0.49 %), and it orders cases exactly as the simulator does (Spearman 1.000) [source: proxy_summary.json; summary_hist_mat.md].
2. **The improvement is real and significant.** About 20 % lower error on both phases out-of-fold over 85 cases (p ≈ 1e-9), and 41 % lower water error on the test cases [source: improve_proxy/results/significance_summary.csv].
3. **Its effect on the history match is modest.** The proxy error passed to the history match (`eps` = 0.111 %) was already far below the best-case misfit; after the upgrade, porosity, convergence and the best simulated match (case 6) are unchanged. The history match is robust to the proxy [source: summary_hist_mat.md, section 7].
4. **15 validation cases are too few to pick a winner.** The best-looking variant on validation (MLP + Gaussian process average, J 0.063 %) failed the test (oil 0.30 %, against 0.064 % for the adopted proxy) and was rejected [source: summary_hist_mat.md, section 7].
5. **The ensemble spread is not a reliable error estimate.** It ranks validation errors only weakly (Spearman 0.48) and is 2.5× smaller than the actual error, so the measured validation error is used as the proxy error instead [source: summary_hist_mat.md, section 7].

*Results*

| Metric (adopted proxy; HM units) | Value | Source |
|---|---|---|
| Validation error, oil / water | 0.074 % / 0.138 % | proxy_output/proxy_summary.json |
| Test error, oil / water | **0.064 % / 0.142 %** | same |
| Out-of-fold error, 85 cases, oil / water | 0.070 % / 0.227 % | improve_proxy/results/significance_summary.csv |
| Ranking agreement with simulator (Spearman, validation and test) | 1.000 | proxy_output/proxy_summary.json |
| Proxy error passed to the history match (`eps`) | 0.111 % | summary_hist_mat.md, section 1 |
| Speed | 0.5 ms per call (20 networks batched) | summary_hist_mat.md, section 1 (team write-up; no saved timing output) |

No speed-up multiple over the simulator is claimed, because its run time is not known. Figures: `proxy_output/figs/fig5_hm_units.png`, `improve_proxy/results/figs/variants_validation.png`.

---

**5.ii History matching** (notebook 3)

*Methodological approach* [source: summary_hist_mat.md, sections 2-4 and 8]

- **Score.** NRMSE (normalised root-mean-square error) of the oil and water cumulatives in HM units, over the 37 producing dates. Gas is left out (≈ 0.3633 × oil). The 4 zero quarters of 1998 are valid data, but every candidate matches them exactly, so they cannot change a score (checked).
- **Bayesian set-up.** Follows the organisers' reference paper (Zhang et al. 2023); gives the range of plausible parameters, not just one answer.
  - Prior: uniform on the train range, no padding, so the proxy is never used outside its training range.
  - Likelihood: Gaussian with the noise level sampled, AR(1) correlated errors (estimated rho = 0.80), and the proxy error added to the variance.
  - Sampler: emcee MCMC (Markov-chain Monte Carlo), 4 ensembles × 16 walkers × 12,000 steps, started across the whole prior box, half dropped as burn-in.
- **Choice of 10: "quality first"** (organiser FAQ 8-10).
  - Candidates: the 100 simulated cases + 4,000 posterior sets.
  - Rule out candidates with implausibility `I_max > 3`.
  - Keep the equal-best band: scores within `eps` (the proxy error, 0.111 %) of the best.
  - Inside the band, simulated cases first by score (their curves are exact), then new sets by score; near-copies (distance < 0.05) are skipped.
- **Validation (13 automated checks).**
  - Convergence: R-hat (< 1.1 = converged), autocorrelation time, effective sample size.
  - Independent cross-check: direct sampling (400,000 draws, noise integrated out exactly).
  - Implausibility screening (assumed 1 % observation + model error).
  - Hold-out: choose on the early history (to 2005-04-01), score on the late history.
  - Recovery test: 5 validation cases used as fake histories at 0.5 % and 2 % noise, each removed from the pool before the full pipeline is run.
  - Physical realism (FAQ 5): parameters inside the range, zero before first production, cumulatives never fall.
- **Origin.** A hybrid of Yi Siang's drafts v1/v2 (score, likelihood, MCMC, checks) and Zi Yang's idea of using posterior sets as candidates.

*Key findings*
1. **The history fixes porosity but leaves other parameters open.** Porosity multiplier 1.366 (5-95 %: 1.341-1.383), posterior spread 6 % of prior; permeability and aquifer partly constrained; fault transmissibility not constrained (spread 99 % of prior) [source: hm_details_real.xlsx, sheet posterior].
2. **The simulated cases already contain matches as good as any parameter set.** The best score the proxy can reach anywhere in the range is 0.484 %, only 0.01 % better than case 6 (0.494 %) and far below `eps` [source: hm_run_summary_real.json].
3. **The 10 chosen are statistically tied.** 6 simulated cases (6, 97, 76, 11, 40, 73) + 4 new parameter sets; scores span 0.053 %, less than `eps` = 0.111 %, so their order is not meaningful. The simulated cases rank 1,512-2,661 of 4,100 candidates but are still chosen, because the gap to the best is within the proxy's own error [source: hm_run_summary_real.json; summary_hist_mat.md, section 3].
4. **The simulator family cannot reproduce the field's late water rise.** In the last quarter of the history (to 2008-01-01) the 10 cases produce oil 11-15 % too fast (5,844-6,051 vs 5,247 STB/d) and water about 10 % too slowly (6,943-7,043 vs 7,782 STB/d), while their totals stay close (water 1.9-2.6 % low, oil 0.1-0.9 % high). The same bias appears in every good simulated case, every posterior set and the global best match, so it is a limit of the model family, not of the matching method [source: recomputed from hist_match_output/top10_series_real.csv and history_observed.csv; summary_hist_mat.md, "Main limits"].
   - Interpretation: whatever controls water breakthrough (e.g. relative permeability) is probably not among the 4 uncertain parameters, as the team's own write-up suggests.
   - The sampled noise levels (posterior medians: oil 0.10 %, water 0.42 %) are larger than the proxy error, which confirms the remaining mismatch is model error, not proxy error.
5. **The 4 new parameter sets are defensible** (team decision, see the box at the top): the proxy error is small and part of the likelihood; the recovery test found the known answers (40 of 40); a new set never displaces an equally good simulated case [source: summary_hist_mat.md, section 8].

*Results*

| Metric | Value | Comparison | Source |
|---|---|---|---|
| Score of the 10 chosen (lower is better) | **0.491-0.544 %** | Best simulated case (6): 0.494 %; median of 100 cases: 6.58 % | hm_run_summary_real.json |
| Equal-best band | Score ≤ 0.601 %: 3,786 candidates (6 simulated, 3,780 new sets) | - | same |
| 2008 cumulatives of the 10 | Oil 38.42-38.72 MMSTB; water 12.48-12.58 MMSTB | Observed 38.38 / 12.81 | summary_hist_mat.md, section 3 |
| MCMC convergence (max R-hat) | 1.024 | Acceptance 0.43; effective sample size ≈3,160 | hm_run_summary_real.json |
| Direct-sampling cross-check | Agrees: every mean within 0.04 posterior sd, every sd within 2 % | Effective sample size 1,326 | same; summary_hist_mat.md, section 3 |
| Implausibility screen (1 % error) | 7.0 % of the parameter box not ruled out; 8 of 100 cases survive | Only porosity restricted (1.270-1.399) | same |
| Hold-out (early → late history) | Spearman 0.993; chosen 10 score 1.07 % on the late history | Best possible 1.05 %; median case 10.2 % | summary_hist_mat.md, section 3 |
| Recovery test: true parameters inside 5-95 % interval | **40 of 40** | Porosity error of the posterior mean 0.5 % (max 1.3 %) | hm_details_real.xlsx, sheet recovery_test |
| Automated checks passed | **13 of 13** | - | hm_run_summary_real.json |

**Posterior** [source: summary_hist_mat.md, section 3; hm_details_real.xlsx, sheet posterior]

| Parameter | Prior (train range) | Posterior 5 / 50 / 95 % | sd / prior sd | Reading |
|---|---|---|---|---|
| Porosity multiplier | 0.801-1.494 | 1.341 / 1.366 / 1.383 | 0.06 | Fixed by the history |
| Permeability multiplier | 0.54-9.50 | 0.71 / 1.99 / 4.83 | 0.52 | Partly constrained, leaning low |
| Aquifer pore volume | 54-200 | 116 / 162 / 195 | 0.58 | Partly constrained, leaning high |
| Fault transmissibility | 0.051-0.150 | 0.057 / 0.105 / 0.146 | 0.99 | Not constrained |

**Comparison of ways of choosing the 10** (all scored with the same proxy) [source: hm_details_real.xlsx, sheet selection_comparison]

| Way of choosing 10 | Mean score | Worst score | Outside train range |
|---|---|---|---|
| Quality first (adopted) | 0.513 % | 0.544 % | 0 |
| Hybrid k-means (v3) | 0.512 % | 0.537 % | 0 |
| Simulated cases only | 0.626 % | 0.925 % (4 ruled out) | 0 |
| Elton's 10 | 0.531 % | 0.572 % | 9 |
| Zi Yang's 10 | 0.513 % | 0.520 % | 0 |

Figures: `fit_top10_real.png`, `posterior_real.png`, `selection_real.png`, `scores_real.png`, `trace_real.png`, `direct_sampling_real.png`, `holdout_real.png` (all in `hist_match_output/figs/`).

---

**5.iii Forecast** (notebook 4; code in `src/forecast_lib.py`)

*Methodological approach* [source: summary_forecast.md, sections 1-5]

- **Why a separate method.** No simulator runs exist after 2008-01-01, and the proxy maps parameters only to the 41 dates it was trained on, so it cannot forecast. Each of the 10 cases is therefore extended by a forecasting method fitted to its own late history. Si Jie's recurrent networks, which can run past 2008, flattened their oil decline after 2010; the organisers' reference paper uses its LSTM only inside the simulated period.
- **Four candidate methods.**

| Code | Method | How it works |
|---|---|---|
| `elton` | Cumulative curve families (Elton's step 4) | Arps-type (oil) and Gompertz (water) curves fitted to the whole 1998-2008 cumulative curve |
| `arps_wor` | Decline curve + WOR trend | Modified hyperbolic Arps decline of the oil rate (switch to exponential at 5 %/yr); water = WOR × oil |
| `liq_wor` | Liquid trend + WOR split | Exponential, non-increasing liquid rate; oil = liquid / (1 + WOR) |
| **`crm_wor`** | **CRM + WOR split (adopted)** | Field-level capacitance-resistance model, a "tank" where the liquid rate relaxes towards G × injection with time constant tau: q_L,k = q_L,k-1 e^(-dt/tau) + (1 − e^(-dt/tau)) G I_k. tau and G fitted on the last 12 quarters; split into oil and water by the WOR trend |

- **Shared rules.** ln(WOR) linear in cumulative oil, moved through the last 2 quarters' WOR ("last" anchor); gas = GOR × oil (GOR of the last 4 quarters, ≈ 0.3633); water injection held at 9,327 STB/d, no new wells (FAQ 2).
- **Physical constraints.** Cumulatives only increase; oil rate never increases; water cut < 1; first-quarter rates within 10 % of the 2008 start rates.
- **Method choice: backtest.** Hide the last 1-3 years (cut-offs 2005, 2006, 2007; 12, 8 and 4 quarters ahead) on all 100 simulated cases and the real field; error = relative error of the volume produced after the cut-off; median over the 100 cases decides. Settings searched: fit window 8, 12 or 16 quarters; WOR anchor `fit` or `last`.
- **Start-point choice: blind hindcast.** The whole chain is repeated blind at each cut-off, with each of the 100 cases as a pseudo-field (the other 99 are candidates) and with the real field as truth; the 10 best blind matches are forecast from 4 start options (`main` = own state, `A1`, `A2`, `fade_T`). **Rule set before the run:** switch only if the new start is better on the pseudo-fields (paired Wilcoxon, p < 0.01) and on the real field.
  - Adopted **"A2"**: start from the field's observed 2008 cumulatives and rates; the case supplies only the curve shape (tau, G and the WOR slope).
- **Uncertainty.** P10 / P50 / P90 of the 10 cases; a posterior band from 200 MCMC parameter sets; and a **wider range** (min-max of the 10 cases, their own-start forecasts and a data-only field forecast) that adds start-point and model-structure uncertainty.

*Key findings*
1. **A tank model plus water-oil trend forecasts best.** `crm_wor` wins the backtest (2.94 %) against `arps_wor` (4.47 %) and `liq_wor` (5.44 %). It uses the water injection record, which the decline methods ignore, and cuts the error by about a third against the best decline method. Averaging the two best methods was worse (3.48 %) [source: forecast_run_summary.json; summary_forecast.md, section 3].
2. **Elton's original forecast fails the backtest** (≈64 % error: oil ≈62 % too high, water ≈66 % too low within 1-3 years), so it was replaced [source: forecast_run_summary.json].
3. **Starting the forecast from the observed 2008 state halves the error.** Blind hindcast: 3.52 % vs 7.47 % on 100 pseudo-fields (better in 83 % of pairs, p = 5e-37) and 3.80 % vs 6.54 % on the real field. It lowers the mean 2028 oil of the 10 from 61.22 to 58.79 MMSTB [source: summary_forecast.md, section 5; forecast_run_summary.json].
   - The simulator's own continuation is the most accurate start on the pseudo-fields (1.5 %) but the least accurate on the real field (8-9 %): further evidence that the field lies outside the model family [source: summary_forecast.md, section 5].
4. **The 10 cases agree within ≈1 % on 2028 oil, which understates the real uncertainty.** In the hindcast, the blind matches' range never contained the field's water (0 of 3 cut-offs), so the wider range is the one to present [source: summary_forecast.md, section 4].
5. **The late water-rise gap carries into the forecast.** The field's WOR slope (0.130/MMSTB) is ≈1.7× the cases' (0.074-0.078). A2 fixes the 2008 level, but the forecast then follows the cases' slower water rise, so it is probably high on oil and low on water later; the data-only field forecast (oil 54.2 MMSTB) marks the other side of the range [source: forecast_output/forecast_by_case.csv; summary_forecast.md, section 6b].

*Results*

| Metric | Value | Comparison | Source |
|---|---|---|---|
| Backtest error, adopted method (`crm_wor`, 12 quarters, WOR "last") | 2.94 % (cases); 3.95 % (field) | `arps_wor` 4.47 / 4.91 %; `liq_wor` 5.44 / 6.09 %; `elton` 64.2 / 66.0 % | forecast_run_summary.json |
| Start-point hindcast error, A2 (pseudo-fields / real field) | 3.52 % / 3.80 % | Own start: 7.47 % / 6.54 %; p = 5e-37 | same |
| **Oil 2028, mean of the 10 cases (headline)** | **58.79 MMSTB** | 38.38 MMSTB in 2008 | same |
| Oil 2028, 10 cases (P10 / P50 / P90) | 58.58 / 58.73 / 59.00 MMSTB | Posterior band (200 sets): 58.23 / 58.38 / 58.59 | summary_forecast.md, section 4 |
| Oil 2028, wider range | **54.16-61.71 MMSTB** | Own start: 61.22 (mean); field reference: 54.16 | forecast_run_summary.json; summary_forecast.md, section 5 |
| Water 2028, 10 cases (P10 / P50 / P90); wider range | 85.04 / 85.27 / 85.53; 81.83-90.01 MMSTB | 12.81 MMSTB in 2008 | summary_forecast.md, section 4 |
| Gas 2028, 10 cases (P10 / P50 / P90); wider range | 21.28 / 21.34 / 21.43; 19.68-22.42 million MSCF | 13.94 million MSCF in 2008 | same |
| Oil rate and water cut, 2028 | ≈1,570-1,630 STB/d; 0.872-0.876 | 2008: 5,247 STB/d | forecast_output/forecast_by_case.csv |
| Fitted CRM gain G (liquid per barrel injected) | 1.342-1.366 | Field 1.365 (G > 1: aquifer support as well as injection) | same |
| Physical checks (forecast curves; fitted constants) | 5 of 5; 4 of 4 pass | - | forecast_run_summary.json, "checks", "realism_checks" |

The fitted tau sits at its 5-day lower bound in every case and in the field: the liquid response is faster than one quarter, so tau cannot be resolved from quarterly data, and with constant injection the forecast does not depend on it [source: summary_forecast.md, section 6b]. Figures: `forecast_output/figs/backtest.png`, `alternative_start.png`, `forecast_fan.png`.

---

**6. Key Findings and Results**

Headline results at a glance. Each row is the number to quote; the method, the full findings and the supporting results are in section 5.

| Stage | Headline result | Details |
|---|---|---|
| Proxy | Test error **0.064 % oil, 0.142 % water** (% of the observed 2008 cumulative); ranks cases exactly like the simulator (Spearman 1.000); 0.5 ms per call | 5.i |
| History matching | The 10 chosen score **0.491-0.544 %** (median of the 100 cases: 6.58 %), statistically tied; **13 of 13** checks pass; recovery test 40 of 40 | 5.ii |
| What the history pins down | Porosity multiplier **1.366** (5-95 %: 1.341-1.383); fault transmissibility not constrained | 5.ii |
| Model-structure gap | In 2008 the 10 produce oil 11-15 % too fast and water about 10 % too slowly; field WOR slope ≈1.7× the cases' | 5.ii finding 4; 5.iii finding 5 |
| Forecast method | CRM + WOR backtest error **2.94 %** (next best 4.47 %); starting from the observed 2008 state halves the hindcast error (3.52 % vs 7.47 %, p = 5e-37) | 5.iii |
| **Oil 2028 (headline)** | **58.79 MMSTB, mean of the 10** (38.38 in 2008); wider range **54.16-61.71** | 5.iii |
| Water 2028 | P50 85.27 MMSTB (12.81 in 2008); wider range 81.83-90.01 | 5.iii |

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

**8. Conclusions**

| Objective | Achieved? | Evidence |
|---|---|---|
| 1. Accurate, fast proxy | Yes | Test error 0.06 % oil / 0.14 % water; Spearman 1.000 [proxy_summary.json] |
| 2. Improve only on evidence | Yes | −20 % out-of-fold error, p ≈ 1e-9; failed variant rejected [significance_summary.csv] |
| 3. Fit history, choose 10 | Yes, with a known gap | 13/13 checks pass; late water rise not reproduced [hm_run_summary_real.json] |
| 4. Evidence-based forecast method | Yes | Backtest 2.94 % vs 4.47 % next best [forecast_run_summary.json] |
| 5. Forecast with honest uncertainty | Yes | 58.8 MMSTB (mean), range 54.2-61.7; all physical checks pass [forecast_run_summary.json] |

- **Practical implications.** Interpretation: decision-makers should plan on the 54.2-61.7 MMSTB range, not the 1 % spread of the 10 cases; the low end (field-only trend) reflects the faster water rise the simulator misses.

---

**9. Next Steps**

Team proposal for the field's engineers (interpretation, not in the repo); confirm before presenting. Each step comes from a result in this file. The forecast holds operations constant (FAQ 2), so it is the do-nothing baseline each step is measured against; it does not give the size of each step's gain.

**9.1 Restore and hold water injection (2008-2009).**
- *What the data shows:* liquid production follows injection within one quarter. Liquid/injection was 1.32-1.40 in every quarter of 2005-2008, and the fitted CRM gain G is 1.365 [source: summary_forecast.md, section 6b; forecast_output/forecast_by_case.csv]. Injection fell 21 % (from a 2000-2003 average of 11,858 to 9,327 STB/d in 2008), and production fell with it [source: data_after_engineering/history_observed.csv].
- *Action:* check reservoir pressure. If it is falling, raise injection in steps of 1,000 STB/d and measure the response after each step.
- *Why now:* each extra 1,000 STB/d injected gives about 550 STB/d of oil at the 2008 water cut, about 270 by 2018 and about 170 by 2028 [computed: G × oil share of liquid; 2008 from history_observed.csv, later years from the forecast water cut in forecast_cases_long.csv].
- *Limit:* the gas-oil ratio has stayed at 0.3633 MSCF/STB. If it starts rising, pressure has dropped too far and injection in that area must go up.

**9.2 Find and block the water that bypasses oil (2008-2011).**
- *What the data shows:* the field's WOR slope (0.130/MMSTB) is ≈1.7× the cases' (0.074-0.078), so injected water is reaching producers through high-permeability layers (section 5.ii, finding 4; section 5.iii, finding 5).
- *Action:* run injection profile logs and inter-well tracers to locate those layers, then block them with gel treatments in injectors or water shut-off in producers.
- *Why it matters:* this gap is the difference between the headline forecast (58.8 MMSTB by 2028) and the data-only field trend (54.2 MMSTB).

**9.3 Drill for bypassed oil early (2010-2014).**
- *What the data shows:* the porosity multiplier is 1.366, so the rock holds about a third more pore volume than the base geology assumed. The fast water rise means part of it is being swept poorly (section 5.ii, findings 1 and 4).
- *Action:* drill infill wells and sidetracks where the updated model and saturation logs show remaining oil.
- *Why early:* nearly two-thirds (64 %) of the oil forecast for 2008-2028 is produced by 2018 [computed from forecast_output/forecast_bands.csv, 10 cases P50: 51.48 MMSTB in 2018, 58.73 in 2028].

**9.4 Track actual production against the forecast (ongoing).**
- *Checkpoint:* by 2013 the water cut should be about 0.73 on the 10-case path, or 0.78 on the field-trend path [computed from forecast_cases_long.csv and forecast_bands.csv]. If the field follows the field-trend path, bring 9.2 and 9.3 forward.
- *Yearly:* rerun notebooks 1-4 (about 30-40 min) with the new data.

---

**10. Visual Assets Inventory**

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
| improve_proxy/results/figs/variants_validation.png | Oil and water validation error of 9 proxy variants (V0 baseline to V6), with the water target line | Evidence-based model choice; note the V6 average looks best here but failed the test (section 5.i, finding 4) |
| Tables: summary_hist_mat.md (top-10 table), summary_forecast.md (per-case 2028 table, hindcast table) | - | Slide-ready tables |

Notebook 1.1 (EDA) figures exist only inside the notebook, not as saved files [source: file listing].

**Missing visuals to create:** pipeline flow diagram; simulator flow diagram (inputs → simulator → outputs, with the 4 uncertain parameters highlighted and kr/Pc marked as fixed; layout in section 3; it also serves as the parameter explainer); problem/data overview (100 cases fanning around the history; could crop fit_top10_real.png); proxy before/after bar chart (v1 vs v2 error); hindcast bar chart (own start vs A2); version timeline (v1→v4); team/collaboration slide with space for meeting snapshots (section 13). Power BI screenshots are not used in the slides (team decision).

---

**11. Suggested Slide Storyline**

14 slides for the 7-minute video. The "Part" column follows the structure of section 5 (i proxy, ii history matching, iii forecast), so each part can be handed to a presenter as a block. Timings are suggestions and add up to 7:00, weighted towards the three technical parts: setup 0:55; (i) proxy 1:15; (ii) history matching 1:50; (iii) forecast 1:50; opening and close 1:10.

| # | Part | Conclusion-style title | Key point | Best visual / number | Time |
|---|---|---|---|---|---|
| 1 | Opening | From 100 Reservoir Simulations to a 20-Year Production Forecast: Machine Learning-Assisted History Matching (title slide) | Title, team | Team What: Elton, Si Jie, Yi Siang, Zi Yang | 0:10 |
| 2 | Setup | Investment decisions depend on simulators that are slow to calibrate | Problem and core question in one line (section 1) | Briefing slides 25-26 | 0:15 |
| 3 | Setup | We had 100 simulator runs, 4 uncertain parameters and 10 years of history | Data, constraints and how the simulator works; the 4 parameters are geologic, while kr and Pc are fixed (section 3) | Simulator flow diagram (section 3, to create); 100 cases fan vs history (crop of fit_top10_real.png) | 0:25 |
| 4 | Setup | A three-part pipeline turns 100 runs into a 20-year forecast | Roadmap only: (i) proxy → (ii) history matching → (iii) forecast; details come in each part | Pipeline diagram with the three parts labelled (to create) | 0:15 |
| 5 | i Proxy | A neural-network proxy imitates the simulator to within 0.06 % oil and 0.14 % water | Accuracy far below the best-case misfit (≈0.49 %); ranks cases exactly like the simulator (5.i) | fig5_hm_units.png; 5.i results table | 0:35 |
| 6 | i Proxy | We kept a proxy upgrade only after it proved a significant 20 % gain | 8 variants tested under a rule fixed in advance; 20-seed tuned MLP adopted; the GP average looked best on validation but failed the test (5.i) | variants_validation.png; significance_summary.csv | 0:40 |
| 7 | ii History matching | The production history pins down porosity but not the fault | Posterior (5.ii, finding 1) | posterior_real.png; porosity 1.341-1.383 | 0:35 |
| 8 | ii History matching | Ten cases match the history equally well, and all 13 checks pass | The 10 and their validation; state openly that 4 are new parameter sets (proxy curves) and why that is defensible (5.ii, findings 3 and 5) | fit_top10_real.png; 0.491-0.544 % vs median 6.58 %; recovery test 40 of 40 | 0:40 |
| 9 | ii History matching | The simulator cannot reproduce the field's late water rise | Model-structure gap: totals match, rates do not (5.ii, finding 4); point back to slide 3: water breakthrough is governed by kr, which is fixed | 2008 rates: oil 11-15 % high, water about 10 % low; field WOR slope 1.7× the cases' | 0:35 |
| 10 | iii Forecast | A tank model plus water-oil trend forecasts best in backtests | Method choice on 100 cases and the field (5.iii, finding 1) | backtest.png; 2.94 % vs 4.47 % vs 64 % | 0:35 |
| 11 | iii Forecast | Starting from the observed 2008 state halves the forecast error | Blind hindcast with a rule fixed before the run (5.iii, finding 3) | alternative_start.png; 3.52 % vs 7.47 %, p = 5e-37 | 0:35 |
| 12 | iii Forecast | The field should reach about 58.8 MMSTB of oil by 2028; plan on 54.2-61.7, because the model, not the parameters, sets the uncertainty | Headline result (mean of the 10) and honest range; the 10 agree within 1 %, which understates the uncertainty; backtests cover at most 12 of the 80 quarters (5.iii, findings 4-5; section 7) | forecast_fan.png; 1 % spread vs wider range | 0:40 |
| 13 | Close | Next: restore injection, block the water channels, drill for bypassed oil, track the forecast | Conclusions and the four next steps (sections 8-9) | Objectives table (section 8); steps 9.1-9.4 | 0:30 |
| 14 | Close | Four members' work was compared and merged into one tested pipeline | Collaboration (a judging criterion) | Roles A-D (section 13); version table v1-v4 (README.md); space for meeting snapshots | 0:30 |

---

**12. Open Questions for the Team**

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
- Rate-proxy units confirmed from the data: oil and water in STB/d, gas in MSCF/d. Errors are always reported by phase (section 5), never pooled across phases, because the units differ. The pooled figures in the repo (rate 32.5, cumulative 23,097) mix units and must not be quoted; `summary_hist_mat.md` and `proxy_output/README.md` were corrected to per-phase values. The rate proxy is a comparison model only.
- The full pipeline (notebooks 1-4) takes about 30-40 min (team estimate). The simulator run time is not known, so the slides make no "N times faster" claim.
- The slides stay technical; no economics.

- The title is approved: "From 100 Reservoir Simulations to a 20-Year Production Forecast: Machine Learning-Assisted History Matching".
- The team write-ups (`summary_hist_mat.md`, `summary_forecast.md`, `methodology_explained.md`, `improve_forecast/README.md`) were corrected to the current 2008 gaps: oil rate 11-15 % high, cumulative water 1.9-2.6 % low, cumulative oil 0.1-0.9 % high.

Still open: none.

---

**13. Collaboration**

Each member's original work is kept in `our_work/`; the final pipeline (notebooks 1-4) merges the best parts after head-to-head comparisons [sources: our_work/README.md; README.md; git log].

**Role A: Si Jie, Business Analyst.** Leads the research direction, frames the problem, and turns the organiser's brief and FAQ into the team's working rules. Builds the Power BI dashboard to present the forecast and its uncertainty.

**Role B: Yi Siang, Data Engineer.** Owns the data architecture, cleans the organiser's Excel files, and manages the hand-over files between notebooks. Supports proxy modelling and history matching with clean, checked inputs.

**Role C: Elton, Data Scientist.** Trains the proxy model, tests improvements to it, and builds the 2028 forecast. Picks the forecast method by backtesting on past data.

**Role D: Zi Yang, Reservoir Analyst.** Explores the data to find which parameters drive production. Leads the history match, selects the 10 best cases, and owns the checks that prove the results hold up.

All four build the slides from this file and present in the video (team decision, 2026-10-17).

**How the team decided** (useful for the collaboration slide):
- **Proxy:** two proxies were built independently, and the team chose Elton's; Si Jie's is kept as a reference [source: our_work/README.md]. Eight improvement variants were then tested under a success rule agreed in advance [source: summary_hist_mat.md, section 7].
- **Choice of the 10:** Yi Siang's, Zi Yang's and Elton's sets of 10 were all scored with the same proxy (table in section 5.ii) [source: hm_details_real.xlsx, sheet selection_comparison].
- **Forecast:** Elton's forecast was used as the baseline and replaced after it failed the backtest [source: summary_forecast.md, section 3].
- **Versioning:** every change was kept only if it helped significantly, with a restore point for each version (v1-v4) [source: README.md].

**Meeting snapshots (placeholder).** Leave room on the collaboration slide for 2-4 photos or screenshots of team meetings, each with a short caption (date and topic, e.g. "proxy comparison" or "choosing the 10"). The team will attach them; none are in the repo.

| Snapshot | Date | Topic / caption |
|---|---|---|
| [PLACEHOLDER - attach snapshot 1] | [to add] | [to add] |
| [PLACEHOLDER - attach snapshot 2] | [to add] | [to add] |
| [PLACEHOLDER - attach snapshot 3] | [to add] | [to add] |
| [PLACEHOLDER - attach snapshot 4] | [to add] | [to add] |
