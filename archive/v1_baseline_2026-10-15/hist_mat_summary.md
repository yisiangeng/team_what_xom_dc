# History matching: summary of findings

15 October 2026. Notebooks: `notebook2_proxy_model.ipynb` (proxy) and `notebook3_hist_match.ipynb` (history matching). Shared model code: `src/proxy_lib.py`. Results: `hist_match_output/` (see its `README.md`).

Scores are **NRMSE as a share of the observed cumulative on 2008-01-01**, oil and water, over the 37 producing quarters (1999-01-01 to 2008-01-01). Example: 0.49 % means the curve is, on average, 0.49 % of the final cumulative away from the history.

## At a glance

- **The proxy works.** Elton's setup is reproduced exactly. The cumulative proxy's error is 0.07 % (oil) and 0.20-0.24 % (water) of the final cumulative, and it ranks the validation and test cases in the same order as the simulator (Spearman 1.000).
- **The history match works.** The MCMC converged (R-hat 1.02 or less). An independent direct-sampling check agrees, and the recovery test found the true parameters of 5 known cases in all 10 runs.
- **The 10 chosen** are 5 simulated cases (6, 76, 11, 40, 73) and 5 new parameter sets from the posterior. Scores are 0.48-0.54 %. The best simulated case scores 0.49 % and the median case 6.6 %. All 10 lie inside the train range, and none is ruled out by the history.
- **What the history pins down:** porosity multiplier 1.356 (5-95 %: 1.338-1.372). Permeability multiplier and aquifer volume are only partly constrained, and fault transmissibility is not constrained at all.
- **Main limits:**
  - Every good match, simulated or new, ends about 2 % low on water and 0.1-0.9 % high on oil in 2008. The simulator family cannot do better.
  - The aquifer volume leans to the top of the design range.
  - The 10 are statistically tied, so their order is not meaningful.

## 1. Proxy model (notebook 2)

### Reproduction of Elton's setup

| Item | Result |
|---|---|
| Tuning | Same grid of 36 configurations and the same two-stage 5-fold CV search. The same best config was chosen: `p4g8` (4 parameters + 8 grid features), 128x128 units, weight decay 1e-4, no log transform. The CV table matches Elton's within 0.002 |
| Rate proxy, test RMSE (pooled) | **32.5** (oil 41.2, gas 13.9, water 35.7). Ridge baseline: 135.8. Mean curve: 763.5. Identical to Elton's saved run on every split |
| Cumulative twin, test RMSE (pooled) | 23,097 against ridge 181,002. Passes Elton's gate. Identical to his saved run |
| Config check on cumulatives (new) | The rate-tuned config is also the best of the top 3 on the cumulative target (5-fold CV) |
| Grid mapping for new parameter sets | Elton's `GridMapper` uses the linear fallback, because the analytic formula misses `mean_tranx` by 0.29 sd. The fallback is within 0.002 sd on the 30 held-out cases |

### Accuracy in history-matching units

| Split | Oil RMS | Oil max | Water RMS | Water max |
|---|---|---|---|---|
| Train | 0.066 % | 0.35 % | 0.229 % | 1.69 % |
| Validation | 0.075 % | 0.23 % | 0.197 % | 0.78 % |
| Test | 0.071 % | 0.30 % | 0.240 % | 0.88 % |

(Proxy error as a share of the final observed cumulative, 37-date window.)

- **Ranking.** The history-matching score from the proxy's curves, against the score from the simulator's curves, has Spearman 0.999 (train), 1.000 (validation) and 1.000 (test). The mean absolute difference is 0.09-0.11 %.
- **`sigma_proxy`.** The proxy error passed to notebook 3 is 0.075 % (oil) and 0.197 % (water), taken from the validation cases. In score units this is **eps = 0.149 %**.
- **Verdict.** The proxy error is about one third of the best achievable mismatch (0.49 %), so the proxy can separate good matches from poor ones. It cannot rank parameter sets whose scores differ by less than about 0.15 %.
- The proxy error grows with time: water reaches about 0.4 % by 2008 (`proxy_output/figs/fig5_hm_units.png`).

## 2. History-matching pipeline (notebook 3)

The pipeline is a hybrid of the team's drafts:
- **Yi Siang's v1/v2:** score, window, likelihood, MCMC, and checks 10b-10g.
- **Zi Yang's approach:** choose from the posterior with k-means.
- **Team decision:** prefer simulated cases, and allow test cases as candidates.

| Choice | Setting |
|---|---|
| Matched | Cumulative oil and water, 37 dates. Gas is left out (it is about 0.3633 x oil) |
| Prior | Uniform on the **train range**, with no padding |
| Likelihood | Gaussian. The noise level is sampled. AR(1) correlation (estimated rho = 0.80). The proxy error is added to the variance |
| Sampler | emcee, 4 ensembles x 16 walkers x 12,000 steps, starting across the whole prior box, with half dropped as burn-in |
| Choice of 10 | Candidates are the 100 simulated cases plus 4,000 posterior sets. Rule out `I_max > 3`, then k-means into 10 clusters and keep one per cluster. Within a cluster, a simulated case wins unless a new set beats it by more than eps |

## 3. Results

### Mismatch of the simulated cases

| | Score |
|---|---|
| Best simulated case (case 6, train) | **0.494 %** |
| Next best | 97 (test) 0.522 %, 76 (validation) 0.528 %, 11 0.533 %, 40 0.537 %, 73 (validation) 0.544 % |
| Median of the 100 cases | 6.58 % |
| Cases not ruled out (`I_max <= 3`, assumed error 1 %) | 8 of 100 |
| Best score the proxy can reach (global search, inside the train range) | 0.467 %, at a corner of the box: lowest fault transmissibility, highest permeability, highest aquifer volume |

The best possible match is only about 0.03 % better than case 6, which is below the proxy error. **The simulated cases already contain matches as good as any parameter set in the design range.**

### Posterior

| Parameter | Prior (train range) | Posterior 5 / 50 / 95 % | sd / prior sd | Reading |
|---|---|---|---|---|
| Porosity multiplier | 0.801 - 1.494 | 1.338 / 1.357 / 1.372 | 0.05 | **Fixed by the history** |
| Permeability multiplier | 0.54 - 9.50 | 1.51 / 3.44 / 6.09 | 0.54 | Partly constrained (positively linked with porosity, see the corner plot) |
| Aquifer pore volume | 54 - 200 | 125 / 167 / 195 | 0.52 | Partly constrained, leans to the upper edge |
| Fault transmissibility | 0.051 - 0.150 | 0.056 / 0.096 / 0.143 | 0.94 | Not constrained |
| Noise, oil / water | | 0.11 % / 0.39 % | | Larger than the proxy error: the mismatch is model error, not proxy error |

- **Convergence.** Acceptance 0.46. Maximum R-hat 1.020. Autocorrelation time 95 steps or less, so the 6,000 kept steps are more than 60 autocorrelation times. Effective sample size about 4,000.
- **Direct sampling.** 400,000 draws, with the noise levels integrated out exactly, give an effective sample size of 1,574. Every mean is within 0.06 posterior sd of the MCMC, and every sd within 2 %. **They agree.** v2's version drew the noise levels at random and reached an effective sample size below 100 even with 3.2 million draws, so that check was changed.
- **Implausibility.** 7.2 % of the train-range box is not ruled out, at an assumed error of 1 %. Only porosity is restricted, to 1.270-1.403. The result depends on the assumed error: 0.5 % rules out every simulated case, and 2 % leaves 21.
- **Hold-out.**
  - The 10 cases chosen on the early history (to 2005-04-01) score 1.07 % on the held-out late history. The best possible 10 score 1.05 %, and the median case 10.2 %. 9 of the 10 are shared with that best set.
  - Spearman correlation between early and late scores: 0.993.
  - So matching the early history predicts the later history well.

### The 10 chosen

| Rank | Source | Case | Score | Oil NRMSE | Water NRMSE | I_max | Fault | Poro | Perm | Aquifer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | new set | - | 0.478 % | 0.21 % | 0.64 % | 1.90 | 0.082 | 1.374 | 3.00 | 198.5 |
| 2 | new set | - | 0.480 % | 0.24 % | 0.64 % | 2.04 | 0.068 | 1.361 | 5.75 | 189.1 |
| 3 | new set | - | 0.484 % | 0.20 % | 0.65 % | 1.82 | 0.065 | 1.376 | 2.31 | 193.2 |
| 4 | new set | - | 0.485 % | 0.21 % | 0.65 % | 1.82 | 0.103 | 1.364 | 3.91 | 184.8 |
| 5 | new set | - | 0.490 % | 0.21 % | 0.66 % | 1.68 | 0.134 | 1.370 | 2.81 | 199.1 |
| 6 | simulated | 6 (train) | 0.494 % | 0.21 % | 0.67 % | 2.16 | 0.058 | 1.358 | 2.47 | 119.8 |
| 7 | simulated | 76 (validation) | 0.528 % | 0.24 % | 0.71 % | 2.60 | 0.108 | 1.383 | 1.64 | 182.2 |
| 8 | simulated | 11 (train) | 0.533 % | 0.24 % | 0.71 % | 2.63 | 0.125 | 1.354 | 7.34 | 188.1 |
| 9 | simulated | 40 (train) | 0.537 % | 0.24 % | 0.72 % | 2.65 | 0.144 | 1.354 | 4.96 | 132.5 |
| 10 | simulated | 73 (validation) | 0.544 % | 0.26 % | 0.72 % | 2.32 | 0.083 | 1.354 | 1.47 | 93.3 |

- **Why 5 new sets.** These 5 clusters, in the high-aquifer part of the posterior where most of its mass lies, contain no simulated case that is not ruled out. In the other 5 clusters a simulated case was within eps of the best new set, so the simulated case was kept.
- **The order inside the 10 is not meaningful.** The scores span 0.066 %, which is less than the proxy error eps (0.149 %). Treat all 10 as equally good matches. The new sets' scores come from the proxy, so they carry that error. The simulated cases' scores are exact.
- **Values at 2008-01-01.**
  - Oil: 38.42-38.72 MMSTB (observed 38.38).
  - Water: 12.48-12.59 MMSTB (observed 12.81).
  - Elton's 3-phase misfit of the 10: 0.0039-0.0045.
- Curves: `hist_match_output/top10_series_real.csv` (simulator curves for the simulated cases, proxy curves for the new sets). Figure: `figs/fit_top10_real.png`.

### Comparison with other ways of choosing 10 (all scored with the same proxy)

| Choice | Simulated cases | Mean score | Worst score | Not ruled out | Outside train range | Spread (mean pairwise distance) |
|---|---|---|---|---|---|---|
| **Hybrid (this pipeline)** | 5 | 0.505 % | 0.544 % | 10 | 0 | 0.61 |
| Simulated only (v1 method A) | 10 | 0.626 % | 0.925 % | 6 | 0 | 0.72 |
| Elton's 10 | 0 | 0.471 % | 0.476 % | 10 | **9** | **0.43** |
| Zi Yang's 10 | 0 | 0.507 % | 0.540 % | 10 | 0 | 0.73 |

(Spread = average distance between two of the 10 in the 0-1 scaled parameter space.)

- **Simulated only.** It has to go down to cases scoring up to 0.93 %, and 4 of its 10 are ruled out by the history. The pool has only about 8 good matches.
- **Elton's 10** score slightly better, but 9 of 10 lie outside the train range (aquifer volume up to 207 against a maximum of 199.8 in the data). There the proxy extrapolates, so their scores are not reliable. They are also the least diverse: his 2028 oil spread was 0.3 %.
- **Zi Yang's 10** are similar to ours in score and spread. Ours replace half of them with simulated cases, whose curves are exact.

## 4. Does the pipeline work? Recovery test

We took 5 validation cases (78, 79, 81, 83, 85) as fake histories, with 0.5 % and 2 % noise, and removed each one from the pool before running the full pipeline.

| Check | Result |
|---|---|
| True parameters inside the posterior 5-95 % interval | **40 of 40** (all 4 parameters, all 10 runs) |
| Porosity error of the posterior mean | 0.5 % on average (absolute), 1.2 % at most |
| Convergence | R-hat 1.07 or less in every run |
| Fit of the chosen 10 to the clean fake history | 0.11-0.32 %, against 0.69-1.45 % for the best 10 simulated cases of the pool |

That last row is optimistic. The new sets are scored with proxy curves, and the fake histories are themselves simulated cases, which the proxy reproduces closely. The parameter-recovery rows have no such bias.

All 8 automated checks in notebook 3 pass: convergence on the real history, direct sampling agreement, porosity recovery, interval coverage, fit of the chosen 10, convergence of the recovery runs, every chosen case not ruled out, every chosen case inside the train range.

## 5. Limitations

1. **Water is under-matched at the end of the history.** Every good match (the 10 chosen and the best simulated cases) ends 1.7-2.6 % low on water and 0.1-0.9 % high on oil at 2008-01-01, and the water residual turns down from 2006. No parameter set in the design range removes it, so it is a limit of the simulator family against the real field. The forecast built on these 10 will likely **underestimate water and slightly overestimate oil** at first.
2. **Edge of the design range.** The aquifer posterior leans to the upper bound, and the proxy's single best point sits on a corner of the box. The real field may have a larger aquifer than any simulated case. The proxy cannot be trusted beyond the train range, so the pipeline does not go there. This should be stated, not hidden.
3. **The 10 are tied.** Score differences are smaller than the proxy error. Present them as 10 equally plausible models, not as a ranking.
4. **New sets have no simulator run.** Their 1998-2008 curves come from the proxy (error about 0.1-0.4 %). If a simulator run were possible, these 5 sets are the ones to run.
5. **Assumed error.** The implausibility cut uses an assumed observation + model error of 1 % (provisional). At 0.5 % no simulated case survives, and at 2 % 21 do. The choice of 10 used it only to rule out poor candidates.
6. **The posterior is not yet used downstream** for forecast bands (P10/P50/P90). The samples are saved in `posterior_samples_real.csv`.

## 6. Next step: forecast to 2028

- **Input.** `top10_cases_real.csv` and `top10_series_real.csv` (the 10 curves to 2008-01-01).
- **Do not reuse Elton's decline-curve extension as it is.** It does not join the 2008 rates. In his forecast the oil rate jumps from 5,247 to 8,905 STB/d in the first quarter, and water falls from 7,782 to about 0 STB/d by 2018. The extension must start from each case's own rates at 2008 and keep them continuous.
- **Account for limitation 1** (the water under-match) when the forecast is presented.

## Files

| Where | What |
|---|---|
| `notebook2_proxy_model.ipynb` | Proxy: tuning, training, evaluation, export |
| `notebook3_hist_match.ipynb` | History matching: MCMC, checks, hybrid choice of 10, recovery test |
| `src/proxy_lib.py` | Elton's model code (paths changed), checkpoint loader, `CumulativeProxy` |
| `proxy_output/` | Models, metrics, figures of notebook 2, `proxy_summary.json` |
| `hist_match_input/` | Hand-over from notebook 2 to notebook 3 (`README.md`) |
| `hist_match_output/` | The 10, all scores, posterior, checks, figures (`README.md`), `hm_run_summary_real.json` |
