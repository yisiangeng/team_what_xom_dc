# History matching: summary of findings

15 October 2026, **version 2** (improved proxy adopted). The history-match results here are also the current ones in repo version 3, because version 3 changed only the forecast start (`summary_forecast.md`).
- **Notebooks:** `notebook2_proxy_model.ipynb` (proxy) and `notebook3_hist_match.ipynb` (history matching).
- **Shared model code:** `src/proxy_lib.py`. **Results:** `hist_match_output/` (see its `README.md`).
- **Version 1 (Elton's cumulative proxy) is archived** in `archive/v1_baseline_2026-10-15/`, with its results and a revert command. Section 7 records why version 2 replaced it.

Scores are **NRMSE as a share of the observed cumulative on 2008-01-01**, oil and water, over the 37 producing quarters (1999-01-01 to 2008-01-01). Example: 0.49 % means the curve is, on average, 0.49 % of the final cumulative away from the history.

## At a glance

- **The proxy works, and was improved.**
  - Elton's setup is reproduced exactly (same tuning result, same errors).
  - A tested improvement, which learns only oil and water and has tuned training settings, cuts the water error on unseen cases by about 40 % (test: 0.14 % against 0.24 % of the final cumulative), with oil unchanged or better.
  - The gain is statistically significant: about 20 % lower error on both phases in an out-of-fold check on 85 cases (p ~ 1e-9). Its effect on the history match is small, because the proxy error was already far below the misfit (section 7).
  - The proxy ranks the validation and test cases exactly like the simulator (Spearman 1.000).
- **The history match works.** All 8 automated checks pass:
  - the MCMC converged (R-hat 1.024 or less);
  - an independent direct-sampling check agrees;
  - the recovery test found the true parameters of 5 known cases in all 40 parameter-runs.
- **The 10 chosen** are 5 simulated cases (6, 97, 76, 11, 40) and 5 new parameter sets from the posterior.
  - Scores are 0.494-0.537 %. The best simulated case scores 0.494 % and the median case 6.6 %.
  - All 10 lie inside the train range, and none is ruled out by the history.
- **What the history pins down.** Porosity multiplier 1.366 (5-95 %: 1.342-1.383). Permeability multiplier and aquifer volume are only partly constrained; fault transmissibility is not constrained.
  - **What this means for the forecast.** The parameters the history leaves open hardly change the forecast. The 10 cases, and 200 posterior sets, differ by only about 1 % in 2028 oil. The larger uncertainty comes from the model's structure (next point).
- **Main limits:**
  - **The good matches are drifting away from the field at the end of the history.**
    - *Totals (cumulative) are close.* In 2008 every good match is 1.9-2.7 % below the field on total water and 0.5-0.9 % above it on total oil. The score uses totals only, so these cases score well.
    - *Rates are not.* In 2008 the matches produce about 5,900-6,050 STB/d of oil against the field's 5,250 (12-15 % high) and about 7,000 STB/d of water against 7,780 (about 10 % low). A total adds up ten years of production, so a recent rate error moves it only a little. A gap in rate means the total error is still growing.
    - *Physically,* water is taking over faster in the field than in the model: the field's water cut rises sooner, so its oil falls and its water climbs earlier.
  - **This is a limit of the simulation model, not of the history-matching method.**
    - The same bias appears in all the good simulated cases, in the new sets from the posterior and in the global search for the best possible match. No combination of the four parameters within their ranges removes it.
    - Whatever sets how fast water arrives is probably not among the uncertain parameters, for example the relative permeability curves or the well completions. Adding rates to the score would only trade fit on totals for fit on rates.
    - If the history were a run of this same simulator with hidden true parameters, some parameter set should fit both. That none does suggests the history comes from outside this model family.
  - **It matters for the forecast, and the forecast now corrects for it.**
    - A forecast that starts from each case's own 2008 values starts with too much oil and too little water and carries that bias to 2028. That was the version 2 forecast.
    - A blind hindcast showed that starting from the field's observed 2008 cumulatives and rates (option A2) halves the forecast error on 100 pseudo-fields (p ~ 1e-36) and lowers it on the real field (3.8 % against 6.5 %).
    - The forecast therefore now uses A2. Oil in 2028 is 58.7 MMSTB, against 61.2 with the own start. The data-only field forecast gives 54.2. See `summary_forecast.md`, section 5.
  - The 10 are statistically tied, so their order is not meaningful.

## 1. Proxy model (notebook 2)

### Reproduction of Elton's setup (unchanged from version 1)

| Item | Result |
|---|---|
| Tuning | Same grid of 36 configurations and the same two-stage 5-fold CV search. The same best config was chosen: `p4g8` (4 parameters + 8 grid features), 128x128 units, weight decay 1e-4, no log transform. The CV table matches Elton's within 0.002 |
| Rate proxy, test RMSE (pooled) | **32.5** (oil 41.2, gas 13.9, water 35.7). Ridge baseline: 135.8. Mean curve: 763.5. Identical to Elton's saved run on every split |
| Elton's cumulative twin, test RMSE (pooled) | 23,097 against ridge 181,002. Identical to his saved run |
| Grid mapping for new parameter sets | Elton's `GridMapper` uses the linear fallback, because the analytic formula misses `mean_tranx` by 0.29 sd. The fallback is within 0.002 sd on the 30 held-out cases |

### The adopted cumulative proxy (notebook 2, section 6b)

It keeps Elton's network (2 x 128, SiLU, dropout 0.05), inputs, grid mapper and training protocol. Changes:
- **Target:** oil and water on the 37 producing dates. Gas is rebuilt as GOR x oil; the 4 pre-production dates are 0.
- **Training:** learning rate 3e-3 and weight decay 1e-5, from a random search.
- **Ensemble:** 20 seeds.
- **Epochs:** 1,145 (CV median), trained on cases 1-70.

**Accuracy in history-matching units** (RMS error as a share of the final observed cumulative, 37-date window):

| Split | Oil: v2 (v1) | Water: v2 (v1) |
|---|---|---|
| Train | 0.049 % (0.066 %) | 0.143 % (0.229 %) |
| Validation | 0.074 % (0.075 %) | **0.138 % (0.197 %)** |
| Test | 0.064 % (0.071 %) | **0.142 % (0.240 %)** |

- **Ranking.** The history-matching score from the proxy's curves against the simulator's: Spearman 1.000 on train, validation and test. The mean absolute difference is 0.05 % (version 1: 0.09 %).
- **Proxy error passed to the history matching** (`sigma_proxy`, validation cases, same method as version 1): oil 0.074 %, water 0.138 %. In score units **eps = 0.111 %** (version 1: 0.149 %).
- **Speed.** The 20 networks are evaluated together in one batched product: 0.5 ms per call, identical to member-by-member evaluation within 5e-8 of the final value.

## 2. History-matching pipeline (notebook 3)

The pipeline is a hybrid of the team's drafts:
- **Yi Siang's v1/v2:** score, window, likelihood, MCMC, checks 10b-10g.
- **Zi Yang's approach:** choose from the posterior with k-means.
- **Team decision:** prefer simulated cases; test cases are allowed as candidates.

| Choice | Setting |
|---|---|
| Matched | Cumulative oil and water, 37 dates. Gas is left out (it is about 0.3633 x oil) |
| Prior | Uniform on the **train range**, no padding |
| Likelihood | Gaussian. Noise level sampled. AR(1) correlation (estimated rho = 0.80). The proxy error is added to the variance |
| Sampler | emcee, 4 ensembles x 16 walkers x 12,000 steps, starting across the whole prior box, half dropped as burn-in |
| Choice of 10 | Candidates = the 100 simulated cases + 4,000 posterior sets. Rule out `I_max > 3`. k-means into 10 clusters, one per cluster. Within a cluster, a simulated case wins unless a new set beats it by more than eps |
| Recovery test | 5 validation cases x 2 noise levels, 2 ensembles x 10,000 steps |

## 3. Results

### Mismatch of the simulated cases

| | Score |
|---|---|
| Best simulated case (case 6, train) | **0.494 %** |
| Next best | 97 (test) 0.522 %, 76 (validation) 0.528 %, 11 0.533 %, 40 0.537 %, 73 (validation) 0.544 % |
| Median of the 100 cases | 6.58 % |
| Cases not ruled out (`I_max <= 3`, assumed error 1 %) | 8 of 100 |
| Best score the proxy can reach (global search, train range) | 0.484 %, at the low edge of fault transmissibility and the high edge of permeability (aquifer 152) |

The best possible match is only 0.01 % better than case 6, far below the proxy error. **The simulated cases already contain matches as good as any parameter set in the design range.**

### Posterior

| Parameter | Prior (train range) | Posterior 5 / 50 / 95 % | sd / prior sd | Reading |
|---|---|---|---|---|
| Porosity multiplier | 0.801 - 1.494 | 1.342 / 1.366 / 1.383 | 0.06 | **Fixed by the history** |
| Permeability multiplier | 0.54 - 9.50 | 0.71 / 1.99 / 4.83 | 0.52 | Partly constrained, leaning low |
| Aquifer pore volume | 54 - 200 | 116 / 162 / 195 | 0.58 | Partly constrained, leaning high |
| Fault transmissibility | 0.051 - 0.150 | 0.057 / 0.105 / 0.146 | 0.99 | Not constrained |
| Noise, oil / water | | 0.10 % / 0.42 % | | Larger than the proxy error: the mismatch is model error, not proxy error |

- **Convergence.** Acceptance 0.43. Maximum R-hat 1.024. Autocorrelation time 122 steps or less (the 6,000 kept steps are about 49 autocorrelation times). Effective sample size about 3,160.
- **Direct sampling.** 400,000 draws, with the noise levels integrated out exactly, give an effective sample size of 1,326. Every mean is within 0.04 posterior sd of the MCMC, and every sd within 2 %. **They agree.** (Yi Siang's draft v2 drew the noise levels at random and reached an effective sample size below 100 even with 3.2 million draws, so the check was changed.)
- **Implausibility.** 7.0 % of the train-range box is not ruled out at an assumed error of 1 %. Only porosity is restricted, to 1.270-1.399. At 0.5 % no simulated case survives; at 2 %, 21 do.
- **Hold-out.** The 10 cases chosen on the early history (to 2005-04-01) score 1.07 % on the held-out late history. The best possible 10 score 1.05 % and the median case 10.2 %. Spearman correlation between early and late scores: 0.993.
- **Version 1 for comparison.** Porosity 1.357 (1.338-1.372), permeability 3.44 (1.51-6.09), aquifer 167 (125-195). The more accurate proxy moves permeability lower; porosity stays within its interval.

### The 10 chosen

| Rank | Source | Case | Score | Oil NRMSE | Water NRMSE | I_max | Fault | Poro | Perm | Aquifer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | simulated | 6 (train) | 0.494 % | 0.21 % | 0.67 % | 2.16 | 0.058 | 1.358 | 2.47 | 119.8 |
| 2 | new set | - | 0.495 % | 0.22 % | 0.67 % | 2.01 | 0.054 | 1.364 | 3.19 | 164.8 |
| 3 | new set | - | 0.497 % | 0.21 % | 0.67 % | 2.13 | 0.086 | 1.370 | 2.79 | 180.7 |
| 4 | new set | - | 0.499 % | 0.21 % | 0.67 % | 2.21 | 0.103 | 1.368 | 2.78 | 160.1 |
| 5 | new set | - | 0.503 % | 0.21 % | 0.68 % | 2.02 | 0.138 | 1.365 | 3.33 | 178.1 |
| 6 | new set | - | 0.510 % | 0.21 % | 0.69 % | 1.94 | 0.090 | 1.346 | 4.42 | 121.1 |
| 7 | simulated | 97 (test) | 0.522 % | 0.23 % | 0.70 % | 1.86 | 0.052 | 1.346 | 5.41 | 153.9 |
| 8 | simulated | 76 (validation) | 0.528 % | 0.24 % | 0.71 % | 2.60 | 0.108 | 1.383 | 1.64 | 182.2 |
| 9 | simulated | 11 (train) | 0.533 % | 0.24 % | 0.71 % | 2.63 | 0.125 | 1.354 | 7.34 | 188.1 |
| 10 | simulated | 40 (train) | 0.537 % | 0.24 % | 0.72 % | 2.65 | 0.144 | 1.354 | 4.96 | 132.5 |

- **Why 5 new sets.** These 5 clusters of the posterior contain no simulated case that is not ruled out. In the other 5 clusters, a simulated case was within eps of the best new set and was kept.
- **The order inside the 10 is not meaningful.** The scores span 0.043 %, less than the proxy error eps (0.111 %). Treat the 10 as equally good matches. The new sets' scores come from the proxy; the simulated cases' are exact.
- **Values at 2008-01-01.** Oil 38.59-38.72 MMSTB (observed 38.38); water 12.48-12.58 MMSTB (observed 12.81). Elton's 3-phase misfit of the 10: 0.0040-0.0044.
- **Changes from version 1.**
  - Simulated cases 6, 76, 11 and 40 are kept; case 73 is replaced by case 97 (the best test case).
  - The 5 new sets are new draws.
  - The mean score is 0.512 % (version 1: 0.505 %; the difference is within the proxy error).

Curves: `hist_match_output/top10_series_real.csv`. Figure: `figs/fit_top10_real.png`.

### Comparison with other ways of choosing 10 (all scored with the same proxy)

| Choice | Simulated cases | Mean score | Worst score | Not ruled out | Outside train range | Spread (mean pairwise distance) |
|---|---|---|---|---|---|---|
| **Hybrid (this pipeline)** | 5 | 0.512 % | 0.537 % | 10 | 0 | 0.57 |
| Simulated only (v1 method A) | 10 | 0.626 % | 0.925 % | 6 | 0 | 0.72 |
| Elton's 10 | 0 | 0.531 % | 0.572 % | 10 | **9** | **0.43** |
| Zi Yang's 10 | 0 | 0.513 % | 0.520 % | 10 | 0 | 0.73 |

- **Simulated only** has to go down to cases scoring 0.93 %, and 4 of its 10 are ruled out.
- **Elton's 10** lie outside the train range in 9 cases out of 10 (aquifer up to 207 against 199.8), where the proxy extrapolates. They are also the least diverse.
- **Zi Yang's 10** are similar to ours in score and spread. Ours replace half of them with simulated cases, whose curves are exact.

## 4. Does the pipeline work? Recovery test

We took 5 validation cases (78, 79, 81, 83, 85) as fake histories, with 0.5 % and 2 % noise, and removed each one from the pool before running the full pipeline.

| Check | Result |
|---|---|
| True parameters inside the posterior 5-95 % interval | **40 of 40** |
| Porosity error of the posterior mean | 0.5 % on average (absolute), 1.3 % at most |
| Convergence | R-hat 1.080 or less in every run |
| Fit of the chosen 10 to the clean fake history | 0.09-0.32 %, against 0.69-1.45 % for the best 10 simulated cases of the pool (optimistic for the new sets: their curves are the proxy's) |
| Fallback | In 3 of the 5 runs at 2 % noise, fewer than 10 candidates passed the implausibility cut (0, 15 and 0). The pipeline then clustered the best 419 candidates by score instead. The recovery still worked in these runs |

All 8 automated checks in notebook 3 pass.

## 5. Limitations

1. **Water is under-matched at the end of the history.**
   - Every good match ends 2008 with cumulative water 1.9-2.7 % low and oil 0.5-0.9 % high.
   - Their last-quarter rates are about 10 % low on water and 12-15 % high on oil.
   - No parameter set in the design range removes this, so it is a limit of the simulator family against the real field.
   - The forecast corrects the 2008 level by starting from the observed values, chosen by a blind hindcast (`summary_forecast.md`, section 5). The decline shape still comes from the cases.
2. **Edge of the design range.** The proxy's single best point sits on two edges of the box (lowest fault transmissibility, highest permeability). The aquifer posterior leans high. The pipeline does not leave the train range, where the proxy is valid.
3. **The 10 are tied.** Score differences are smaller than the proxy error.
4. **New sets have no simulator run.** Their 1998-2008 curves come from the proxy (error about 0.07 % oil, 0.14 % water).
5. **Assumed error.** The implausibility cut uses an assumed observation + model error of 1 % (provisional). It is used only to rule out poor candidates.
6. **The choice of 10 depends on how the proxy error is measured.** In the tests, using an out-of-fold error (larger) instead of the validation error changed which clusters got new sets. The posterior did not change.

## 6. From the history match to the forecast

The forecast to 2028 is done: `notebook4_forecast.ipynb`, `summary_forecast.md`, deliverable `forecast_output/10_Final_deliverables.xlsx`.
- It uses the 10 cases above, in rank order (sheet `Case k` = rank k).
- Because of limitation 1, each forecast starts from the observed 2008 values. The case gives only the shape.
- Result: oil 58.7 MMSTB in 2028 (10 cases within about 1 %). The wider range, which adds the start-point and model uncertainty, is 54.2-61.7 MMSTB.

Remaining work: the documents, the Power BI dashboard and the video (due 20 Oct).

## 7. Proxy improvement tests (15 October 2026)

**Where:** `improve_proxy/`.
- `notebook3_proxy_variants_test.ipynb`: variants, and a copy of notebook 3 run on the chosen proxy.
- `followup_runner_up.py`: the check that led to adoption.
- `proxy_variants.py`: the variant code.
- `results/`: CSV and figures.

**Restore point:** `archive/v1_baseline_2026-10-15/`.

### Protocol

- **Metric.** RMS error in history-matching units (above), oil and water; J = sqrt(mean of the two squares).
- **Choice on cases 1-85 only.** 5-fold CV on the 70 train cases (Elton's folds and protocol), and the 15 validation cases (model trained on 1-70).
- **Success rule (agreed).** Validation water at least 10 % lower and validation oil not higher than the baseline.
- **Consistency check (added after the first run, see below).** CV J must not be higher than the baseline's.
- **Test cases 86-100** are reserved for confirming the winner.

### Variants and results (validation / CV, % of the final observed cumulative)

| Variant | What changed | Val oil | Val water | Val J | CV J | Passes |
|---|---|---|---|---|---|---|
| V0 baseline | Elton's twin | 0.075 | 0.197 | 0.149 | 0.180 | - |
| V1 | Oil + water on the producing dates (gas rebuilt) | 0.070 | 0.160 | 0.124 | 0.156 | yes |
| V5 | Monotone output (non-negative increments) | 0.148 | 0.385 | 0.292 | 0.316 | no |
| V1+V5 | Both | 0.167 | 0.352 | 0.275 | 0.271 | no |
| V4 | V1 + random search (16 settings): lr 3e-3, wd 1e-5 | 0.076 | 0.130 | 0.106 | 0.125 | no (oil +1.0 %) |
| V3 (10 seeds) | V4 with a 10-seed ensemble | 0.077 | 0.131 | 0.107 | 0.125 | no (oil +2.5 %) |
| **V3 (20 seeds), adopted** | **V4 with a 20-seed ensemble** | **0.075** | **0.138** | **0.111** | **0.125** | **yes** |
| V6 | PCA + Gaussian process (Zi Yang), 8 scores | 0.090 | 0.094 | 0.092 | 0.175 | no (oil +19 %) |
| V6 average | V4 + GP | 0.042 | 0.079 | 0.063 | 0.119 | yes |

**Other findings from the variants.**
- **Mini-batch training** (batch 16) did not beat full batch in the search.
- **Ensemble spread as a per-point error estimate.** The spread of the 20 members ranks the validation errors only weakly (Spearman 0.48) and is 2.5 times smaller than the actual error. So it was not used as the proxy error.
- **More data.** In a learning curve on the validation cases, the error fell from 52 to 70 training cases, but a refit on cases 1-85 did not improve the test cases (below), so the adopted proxy is trained on 1-70.

### What happened

1. **The rule picked the MLP + GP average** (best validation J), refitted on 1-85.
2. **The single test rejected it.** Test oil 0.30 % against 0.07 % for the baseline, J 0.24 % against 0.18 %. The honest out-of-fold error over cases 1-85 agrees (oil 0.18 % against 0.09 %). The GP half generalises poorly on oil; the 15 validation cases happened not to show it. The notebook therefore kept the baseline, and its copy of notebook 3 reproduced the version 1 posterior.
3. **The runner-up, the tuned MLP (V4), was then checked** with the out-of-fold error over cases 1-85 (5 folds, models trained like the final model). This check does not use the test cases. Openly, it was also scored on the test cases, a **second use** of them (`followup_runner_up.py`, `results/followup_runner_up.csv`):

| Variant (trained on 1-70 unless stated) | Out-of-fold over 1-85: oil / water / J | Test: oil / water / J | Test, refit on 1-85: J |
|---|---|---|---|
| V0 baseline | 0.089 / 0.284 / 0.210 | 0.071 / 0.240 / 0.177 | 0.174 |
| V1 | 0.083 / 0.263 / 0.195 | 0.068 / 0.187 / 0.141 | 0.150 |
| V4, 5 seeds | 0.068 / 0.222 / 0.164 | 0.062 / 0.137 / 0.106 | 0.116 |
| **V4, 20 seeds (adopted)** | **0.070 / 0.227 / 0.168** | **0.064 / 0.142 / 0.110** | 0.117 |

4. **Adopted: V4 with 20 seeds, trained on cases 1-70.**
   - It passes the agreed rule (water -30 %, oil -1 % on validation) and the consistency check.
   - It beats the baseline in the larger out-of-fold check (J -20 %) and on the test cases (J -38 %, water -41 %, oil -10 %).
   - The 5-seed version is about as good, but it misses the strict "oil not worse" clause by 1 % (0.0760 against 0.0753), which is within the noise of 15 cases.
   - Refitting on 1-85 did not help on the test cases.
5. **Adoption.**
   - The model code is in `src/proxy_lib.py` (`WindowScaler`, `improved_cv`, `improved_train`, `IMPROVED_CONFIG`).
   - Notebook 2 trains it in section 6b and exports it to `hist_match_input/`.
   - Notebooks 2, 3 and 4 were re-run.
   - Elton's twin is still trained and reported in notebook 2 for comparison.

### Is the improvement significant? (`significance_test.py`, `results/significance_summary.csv`)

This is a paired comparison on the same cases. For each case, the error is the RMS over the 37 window dates, as a % of the final observed cumulative. The table gives the Wilcoxon signed-rank test on the per-case differences, and a 95 % case-bootstrap interval for the ratio of pooled RMS (adopted / baseline).

| Cases | Phase | Baseline → adopted | Ratio (95 % CI) | Cases better | Wilcoxon p |
|---|---|---|---|---|---|
| Out-of-fold, 1-85 (n = 85) | oil | 0.089 → 0.070 | 0.79 (0.73-0.87) | 72 / 85 | 1e-9 |
| Out-of-fold, 1-85 (n = 85) | water | 0.284 → 0.227 | 0.80 (0.70-0.90) | 66 / 85 | 6e-9 |
| Test, 86-100 (n = 15) | oil | 0.071 → 0.064 | 0.89 (0.71-1.04) | 11 / 15 | 0.12 |
| Test, 86-100 (n = 15) | water | 0.240 → 0.142 | 0.59 (0.46-0.73) | 14 / 15 | 1e-4 |

- **Statistically significant.** The out-of-fold check on 85 cases shows a 20 % error reduction for both phases, with p < 1e-8. On the test cases, water is significant; oil improves but is not significant with 15 cases.
- **The practical effect on the history match is modest.**
  - eps falls from 0.149 % to 0.111 %, against best-case misfits of about 0.49-0.54 %.
  - Porosity, convergence and the best simulated match (case 6) are unchanged.
  - Four of the five simulated cases in the 10 are the same as in version 1.
- **Caveats.**
  - The test cases have now been used three times, so the test p-values are optimistic. The out-of-fold result does not use them.
  - The seeds are fixed, so the test treats the two trained ensembles as given. It does not include the variation from retraining with other seeds.

### Lessons

- **15 validation cases are a small sample.** The GP average looked best there and was worst on the test cases. The CV and out-of-fold checks on more cases are what exposed it.
- **The history match is robust to the proxy:** porosity, the convergence and the best matches stay the same. The proxy mostly affects which posterior clusters get new sets, and the size of eps.
