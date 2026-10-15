# History matching guide: v1, v2 and Ziyang's pipeline

Working note for the team. It explains three pipelines in two ways: **In plain words** (for anyone) and **Technical** (for the data scientist and the code review).

Sources: the three notebooks (`solution/notebook_hist_matching.ipynb`, `solution/notebook_hist_matching_v2.ipynb`, `history_matching_ziyang.ipynb`), the input README (`solution/histmatch_input/histm_README.md`), and checks run on the data. Where a statement is my inference, it says so.

Diagrams are plain text because the viewer cannot render Mermaid.

---

## 1. The problem in one page

**In plain words**

A reservoir simulator is a very detailed computer model of an oil field. You give it 4 uncertain settings. It returns how much oil and water the field produces over time. It is accurate but slow.

We have 10 years of real production (the *history*). We ask: **which settings make the model reproduce that history?** This is *history matching*. The answer is the basis for the 20-year forecast.

Many different settings can fit the history equally well. So we look for a *set* of good settings, not one.

**Technical**

- Inputs: `fault_trans`, `poro_mult`, `perm_mult`, `aquifer_pv`.
- Outputs: cumulative oil and water (gas is about 0.3633 x oil, within 0.1%, so it is not modelled) on quarterly dates.
- 100 simulated cases (70 train, 15 validation, 15 test) and one observed history.
- A **proxy** (a fast learned copy of the simulator) lets us test thousands of settings in seconds.
- The task has two parts: find settings that fit, and say how sure we are.

```
 4 settings --> [ simulator: slow ] --> curves
 4 settings --> [ proxy: fast, a little less exact ] --> curves

 history matching: which settings give curves that match the real history?
```
*Caption: the proxy lets us try thousands of settings.*

---

## 2. Key terms

| Term | In plain words | Technical |
|---|---|---|
| **Proxy** | A fast copy of the simulator, learned from the known cases. | A function `theta (m,4) -> curves (m,T,Q)`. In our code: PCA on the curves, then one Gaussian process per PCA score. |
| **Prior** | What we believe about the settings before we look at the history. | Uniform over the train range, widened by 5% in our code (Ziyang: no padding). It carries no preference inside the range. |
| **Likelihood** | How well one set of settings explains the history. | A Gaussian on the curve error. Smaller error means higher likelihood. |
| **Posterior** | What we believe after looking at the history. | Prior x likelihood. Represented by a cloud of sampled settings. |
| **MCMC** | A guided random walk that collects the posterior. | Affine-invariant ensemble sampler (stretch move). `emcee` in our code, hand-written in Ziyang's. |
| **Walker** | One explorer in the walk. | One chain in the ensemble. |
| **Burn-in** | The early steps we throw away. | First 50% of steps. Walkers start at arbitrary points. |
| **R-hat** | Do independent groups of explorers agree? | Split R-hat across chains. Below 1.1 is acceptable. |
| **Autocorrelation time** | How many steps until a step is "new". | Integrated autocorrelation time. Run length should be at least 50x this. |
| **Effective sample size (ESS)** | How many independent samples the run is worth. | About (kept samples) / (autocorrelation time). Aim for at least about 1,000. |
| **NRMSE / score** | How far a curve is from the history, as a share of the final value. | Root-mean-square error of cumulative oil and water, each divided by the observed value at 2008-01-01. |
| **Misfit (Ziyang)** | Same idea as the score. | Weighted RMS of gas, oil and water in normalised units. |

---

## 3. How prior and posterior work (the core idea)

**In plain words**

1. Start with an open mind (the prior): any setting in the allowed range is possible.
2. Send out explorers. Each one tries settings.
3. For each try, the proxy predicts the production. We compare it with the real history.
4. Good matches attract explorers. Poor matches lose them.
5. After many steps, the crowd of explorers is the posterior: the settings that are possible given the history.

"Keep the good ones" is soft. A poor guess is kept rarely, not never.

```
 prior (allowed range) ---\
                            >-- explorers search --> posterior (crowd)
 history match ------------/
```
*Caption: the prior and the history match together shape the crowd. The crowd is the posterior.*

**Technical**

- The sampler targets `p(theta | history)` proportional to `prior(theta) x likelihood(history | theta)`.
- The prior only marks the allowed box. Individual train cases do not enter it. The 70 train cases enter in two ways: through the box limits (min and max), and through the proxy training.
- The posterior does not exist before the run. It is the output.
- A simple version of the same idea is **direct sampling**: draw many settings from the prior, then weight each by its likelihood. v2 section 10b does this as a cross-check. It agreed with the MCMC (means within 0.07 posterior standard deviations).

**What the posterior shows here (both pipelines agree)**

| Parameter | Prior range | Posterior (5-95%) | Reading |
|---|---|---|---|
| Porosity multiplier | 0.80 - 1.49 | about 1.30 - 1.38 | The history fixes it. |
| Fault transmissibility | 0.051 - 0.150 | still wide | The history says little. |
| Permeability multiplier | 0.54 - 9.50 | still wide | The history says little. |
| Aquifer pore volume | 54 - 200 | mostly wide | The history says little. |

So only porosity is determined. Many different cases fit equally well.

---

## 4. Pipeline v1: `notebook_hist_matching.ipynb`

**In plain words**

The notebook gets predicted production curves for up to 100 cases and the real history. It scores each case against the history. It then picks the 10 best cases that also differ from each other. It saves their full time series.

In parallel, it runs a Bayesian search (method B) that shows which settings the history allows. This gives the uncertainty picture. By default it does not choose the 10.

**Technical flow**

```
 predictions.csv (cases x dates x oil/water) + observed history
                         |
        [0] align to history window, keep oil + water, scale
                         |
         +---------------+----------------+
         |                                |
  [A] RANK THE CASES                [B] BAYESIAN MCMC
  score each case vs history        sample 4 settings + noise levels
  (default selection)               proxy(theta) -> curves
         |                                |
         +---------------+----------------+
                         |
        [C] choose 10: good AND not near-copies (min_sep)
                         |
        [D] checks, plots, CSV export, template writer
```
*Caption: method A picks the 10 by default. Method B adds the posterior.*

| Step | What happens | Why |
|---|---|---|
| Input | `predictions.csv`, `case_params.csv`, optional `history.csv`. With an empty folder it runs a demo on 25 real train cases. | One input folder, one output folder. No code change for the real run. |
| 0. Align and scale | The window starts at the first date with oil: **1999-01-01**, giving **37 dates** to 2008-01-01. Cumulative oil and water only. Each divided by its observed value at 2008-01-01. | Gas is about 0.3633 x oil in every case, so it would count oil twice. Cumulatives avoid the rate-convention mismatch. Scaling makes oil and water count equally. |
| A. Score | Per case: NRMSE of oil and water, combined `score`, worst error, error at the last date, likelihood weight. | A percentage that is easy to read. |
| Residual correlation `rho` | AR(1) filter, estimated from the best 5 cases' residuals. | Cumulative errors are smooth in time. Treating them as independent makes the answer too confident. `rho = 0` is the paper's version. |
| B. Proxy | The team's proxy, or a stand-in (PCA + Gaussian process) trained on the supplied cases. The stand-in measures its own cross-validated error `sigma_proxy`. | The sampler tries settings that are not among the supplied cases. |
| B. Model | Prior: uniform on the train range +5%. Noise level per quantity: **sampled** (log scale). Variance = sampled noise^2 + `sigma_proxy`^2. | The history has no stated error. A weak proxy gives a wider, more honest answer. |
| B. Sampler | `emcee`: 4 independent ensembles x 16 walkers x 12,000 steps. Walkers start at random points across the whole prior box. First half dropped (384,000 kept steps). | Starting from many places, in separate groups, lets us test that the answer does not depend on the start. |
| B. Checks | R-hat (max 1.047), autocorrelation (about 150-200 steps), ESS (about 1,950). | See section 2. |
| C. Choose 10 | Walk down the ranking. Skip any case closer than `min_sep = 0.05` (0-1 scaled parameter space) to one already chosen. | Avoids near-copies. It never triggers on the real cases (closest pair 0.081), so it is only a safeguard. |
| D. Output | `top10_cases`, `top10_series`, `scores_all`, posterior summary, diagnostics, samples, figures. Gas is derived as 0.3633 x oil when missing. | The 10 series feed template tabs `Case 1` to `Case 10`. |

### How v1 uses the prior and the posterior

| Item | Role in v1 |
|---|---|
| **Prior** | The walkers start from it and may not leave it. It sets the allowed box for the 4 settings and the noise levels. |
| **Posterior** | Output of method B. Used for: parameter ranges, the P10-P90 curve band, the posterior-mean curve score, the nearest-unseen-case check, and diagnostics. |
| **Choosing the 10** | **Default `method="A"`: the posterior is not used.** Cases are ranked by curve mismatch. |
| With `method="B"` | Each case gets a distance to the posterior (Mahalanobis distance in the 0-1 scaled space). Rank by that distance. |
| With `method="hybrid"` | `(rank_A + rank_B) / 2`, ties broken by `rank_A`. |

**Mahalanobis distance, in plain words.** The distance is measured in "posterior widths". Porosity has a narrow posterior, so a small miss in porosity counts a lot. The other three settings have wide posteriors, so a big miss counts little. A case ranks high if its porosity is right, whatever its other settings. This method uses only the case's **parameters**, not its curve.

### What the tests showed (stand-in proxy, 25 cases)

| Question | Answer |
|---|---|
| Does the true case rank first? | Yes with method A (1st with no noise; 1st-3rd with 0.5% and 2% noise). |
| Which ranking is best? | **A.** On unseen fake histories, its top 10 equals the oracle's top 10. B is a little worse (the stand-in is trained on only 25 cases). Hybrid sits between. |
| What is B good for? | The posterior and its uncertainty. Porosity is found to about 1-2%. |
| Does the prior matter? | Little. A triangular prior gives the same top 10. Means move by at most about 6% of the range. |
| Scale | 85 cases in about 24 s. Best case 6 (0.49%). Six cases are below 0.7%. |
| Draft top 10 (25 cases) | 11, 40, 50, 20, 24, 47, 65, 19, 35, 32. |

---

## 5. Pipeline v2: `notebook_hist_matching_v2.ipynb`

**In plain words**

v2 is v1 plus six extra checks. Sections 1-10 are unchanged. The only code difference is the output folder (`histmatch_output_v2/`). Each addition answers one weakness.

| Section | Added | Why it helps (one sentence) |
|---|---|---|
| **10b** Direct sampling | Draw 400,000 settings from the prior and weight each by its fit. | It finds the same posterior by a different route, so we know the MCMC was not fooled by a bad run. |
| **10c** Implausibility | Count how many standard deviations a prediction is from the history. Above 3 means ruled out. | It replaces the hand-picked 1% cutoff with a rule that has a reason, and it maps which settings the history excludes. |
| **10d** Parameter search | Differential evolution on the proxy, then 9 spread-out sets from the equally-good region. | It can propose new settings, so the top 10 are not limited to existing cases. |
| **10e** Oil-water trade-off | Plot oil error against water error, with the Pareto front. | A single score hides whether a case is wrong on oil, on water, or in which direction. |
| **10f** Hold-out check | Match on the first 70% of history, test on the last 30%. | It shows whether a case that fits the early history also predicts the later history, so a lucky fit stands out. |
| **10g** Scorecard | One table per case with four layers: fit, rank, benchmark, beyond-fit. | A case strong in every layer is a safe choice. |

**Technical notes**

- 10b: importance sampling with the same `LogPosterior`. Agreement rule: means within 0.3 posterior sd and sd within 25%. No chains, so no R-hat. The check is ESS.
- 10c: `I = |obs - pred| / sqrt(IMPL_SD^2 + proxy_error^2)`, worst over oil and water at 4 dates. `IMPL_SD = 1%` is **provisional**.
- 10d: `min_sep` also forces the 9 extra sets apart. A warning prints if the best set sits at the edge of the range.
- 10f: the top 10 are chosen on the early period only, then scored on the late period.

**What v2 found (25 cases, stand-in proxy)**

| Addition | Result |
|---|---|
| Direct sampling | ESS 1,655. Means within 0.07 sd, sd within 3%. **Agrees with the MCMC.** |
| Implausibility | At 1% assumed error, only cases 11 and 40 survive. 8.2% of the parameter space survives. Porosity keeps 19% of its range (1.26 - 1.39). At 0.5% / 1% / 2% error, 0 / 2 / 8 of 25 cases survive, so 1% stays provisional. |
| Parameter search | Best predicted score 0.499% against 0.533% for the best pool case. The gain is below the proxy error (0.38%). The best set sits at the edge of the train range. |
| Trade-off | Only case 11 is on the front. Oil and water errors move together. |
| Hold-out | Top 10 chosen on early data score 2.76% on late data. Oracle 2.67%. Pool median 6.08%. 9 of 10 shared with the oracle. |
| Scorecard | Top-10 lists from curve score, posterior distance, hybrid and hold-out share 8-10 cases. The ranking is robust. |

### How v2 uses the prior and the posterior

Same as v1, plus:

- 10b uses the **prior** as the sampling distribution and builds the **posterior** from weights.
- 10d plots the new candidate sets against the posterior cloud. The "equally good region" itself comes from the proxy error, not from the posterior.
- 10g compares top-10 lists, including the one from posterior distance (`rank_B`).

The default selection is still method A.

---

## 6. Ziyang's pipeline: `history_matching_ziyang.ipynb`

**In plain words**

This notebook builds its own proxy. It finds the single best-fit settings. It then explores all good settings with a Bayesian search. At the end it takes 10 settings from that crowd, chosen so they are good **and** different from each other. The 10 are new settings, not existing cases.

**Technical flow**

```
 data_after_engineering/  (same file names as our solution/output/)
              |
   [0] scale: gas, oil, water each / observed maximum
              |
   [1] PROXY: curves -> PCA -> one regressor per score
       candidates: poly2, poly3 (ridge), Gaussian process (GP)
       pick on validation (71-85); refit on train+val; score ONCE on test
              |
   [2] SEARCH: differential evolution x10 -> best-fit settings
              |
   [3] SAMPLE: Bayesian MCMC (hand-written stretch move)
       walkers start AT the best fit; uniform prior on the train range
              |
   [4] CHOOSE 10: better half of samples by misfit -> k-means(10)
       -> lowest-misfit sample per cluster
              |
   hm_selected_cases.csv + hm_posterior.npz  -> forecast stage
```
*Caption: proxy, search, sampling and selection in one notebook.*

| Step | What happens | Why |
|---|---|---|
| Scaling | Each quantity divided by its history maximum. Misfit = weighted RMS (weights all 1). | Equal weight for gas, oil and water. |
| Proxy | PCA (3, 5 or 8 scores) + poly2 / poly3 / GP. **GP with 8 scores wins**: validation RMSE 0.0006. Test RMSE **0.0032** mean, 0.0107 max. | Fair comparison on unseen cases. The test set is scored once. |
| Search | 10 runs of differential evolution from different seeds. All 10 land on one point, misfit 0.0038. Three settings sit on the box edge: fault 0.15 (max), permeability 0.5375 (min), aquifer 199.8 (max). | A global search that cannot get stuck. Identical runs mean one main valley. |
| Likelihood | `-0.5 x N_EFF x misfit^2 / sigma^2`, `N_EFF = 10`. | `N_EFF` replaces the time-correlation model: the 123 values carry far fewer than 123 independent pieces of evidence. It is a heuristic. A larger value gives a narrower posterior. |
| Noise | **Fixed**: `sigma = sqrt(best misfit^2 + proxy test error^2)` = about 0.005. | Simple. The noise is not sampled. |
| MCMC | 32 walkers, 8,000 steps, one ensemble, 50% burn-in, thinned by the largest autocorrelation time. Acceptance 0.43. Autocorrelation 89 - 106 steps. 1,216 posterior samples. | Standard stretch move, written out so no extra package is needed. |
| Selection | Keep samples with misfit at or below the median. K-means into 10 clusters. Take the lowest-misfit sample in each. | Good **and** spread over the posterior. |

**Result**

| Parameter | Posterior 5% / 50% / 95% | Prior range |
|---|---|---|
| fault_trans | 0.056 / 0.103 / 0.145 | 0.051 - 0.150 |
| poro_mult | 1.319 / 1.349 / 1.379 | 0.801 - 1.494 |
| perm_mult | 0.98 / 4.70 / 9.09 | 0.54 - 9.50 |
| aquifer_pv | 67 / 140 / 195 | 54 - 200 |

The 10 selected sets have misfit 0.00381 - 0.00408. The proxy's own uncertainty at those points is about 0.0038. The differences between the 10 are smaller than the proxy error, so the ranking inside the 10 is not meaningful. The notebook says so itself.

### How Ziyang's pipeline uses the prior and the posterior

| Item | Role |
|---|---|
| **Prior** | Uniform on the train box in 0-1 scaled units, no padding. Samples outside are rejected. |
| **Start** | Walkers start at the best-fit point, not across the prior. |
| **Posterior** | **The source of the 10.** Samples are filtered by misfit and clustered. |
| **Also** | 5/50/95% ranges, curve band, corner plot, and saved to `hm_posterior.npz` for the forecast stage. |

### Points to flag (verified unless marked)

1. **Gas is counted with oil.** After scaling, the gas and oil curves of the history differ by at most 0.00002. They are the same curve. Water causes 76% of the squared error across the 100 cases, gas and oil 12% each, so the effect on the total is smaller than the double count suggests.
2. **The window includes 4 empty early dates.** All 41 dates are used. The history has 4 leading zero rows. This lowers the RMS by a factor of about 0.95 (0.0038 would read about 0.0040 on the 37 real dates).
3. **Chains are short.** After burn-in, the length is 38 - 45 times the autocorrelation time. The notebook's own target is above 50. One ensemble, no R-hat.
4. **`N_EFF` is a free choice** and sets the posterior width.
5. **Validation against test.** 0.0006 against 0.0032. 2 of 15 validation and 2 of 15 test cases lie outside the train range, so the proxy extrapolates there. This is a likely cause of part of the gap, not proven.
6. **`sigma` counts the proxy error twice** (my reading of the formula). The best misfit already contains some proxy error. It makes the posterior slightly wider. Not a bug.
7. **The 10 are never simulated.** Their curves come from a proxy with 0.32% test error.
8. **Data folder.** It reads `data_after_engineering/`, which is not in the repo. The file names match our `solution/output/`, so it is probably a copy (inference).
9. **File name.** It was saved without an extension, then renamed to `history_matching_ziyang.ipynb`.

---

## 7. Side by side

| | Our v1 / v2 | Ziyang |
|---|---|---|
| Proxy | Supplied by the proxy team, or a stand-in | Built and tuned in the notebook (GP, 8 scores) |
| Quantities matched | Oil and water | Gas, oil and water |
| Dates in the window | 37 (from 1999-01-01) | 41 (includes 4 empty quarters) |
| Prior | Uniform, train range +5% | Uniform, train range |
| Noise level | **Sampled**, plus proxy error | **Fixed** from best fit and proxy error |
| Time correlation of errors | AR(1) filter, estimated | `N_EFF = 10` shortcut |
| Walker starts | Random over the whole prior box | At the best-fit point |
| Sampler | `emcee`, 4 ensembles x 16 walkers x 12,000 steps | Hand-written, 1 ensemble x 32 walkers x 8,000 steps |
| Convergence checks | R-hat, autocorrelation, ESS; v2 adds direct sampling | Autocorrelation only |
| Posterior used for | Uncertainty, cross-check, optional ranking | Source of the final 10 |
| The 10 | Existing cases (default) | New parameter sets |
| Real simulated curves for the 10? | Yes, up to 2008 | No |
| Selection for variety | `min_sep` on parameters | K-means on the posterior |

**The main difference in one line.** Ours ranks existing cases and uses the posterior to describe the uncertainty. Ziyang's draws the answer from the posterior.

```
 OURS:    pool of cases -> score each vs history -> top 10
                           (posterior informs, via B / hybrid)

 ZIYANG:  posterior -> better half -> k-means -> best sample per cluster
```
*Caption: where the 10 come from.*

---

## 8. What the proxy team must deliver

**In plain words.** One main file, with the proxy's predicted curves for each case. Optionally, the proxy itself.

| File | Rows | Columns | Required |
|---|---|---|---|
| `predictions.csv` | cases x dates | `case_id`, `date`, `oil_cum`, `water_cum` (+ optional `gas_cum`) | **Yes** |
| `case_params.csv` | 1 per case | `case_id`, `fault_trans`, `poro_mult`, `perm_mult`, `aquifer_pv` | **We create it** from `solution/output/cases.csv`. It is not proxy output. |
| `history.csv` | 41 | `date`, `oil_cum`, `water_cum` | No. Already in `solution/output/`. |

**Rules**

- Long format, one row per case and date. Every case has the same dates, no gaps, no NaN.
- `date` as `YYYY-MM-DD`. Cumulative values in real units: oil and water in STB, gas in MSCF.
- Window dates: 1999-01-01 to 2008-01-01 (37 quarters) as a minimum. The full axis is 41 quarters from 1998-01-01.
- To fill the template, dates must run to 2028-01-01 (41 history + 80 forecast = 121 dates; 12,100 rows for 100 cases).
- Drafts and tests use cases 1-85 only. The final run adds test cases 86-100 once.

**Notebook behaviour to remember.** For a real run, the notebook reads `histmatch_input/case_params.csv` only. If the file is missing, the parameters are `None` and **method B and the v2 extras 10b-10d are skipped**. Someone must place the file there.

**Optional proxy function** (needed for method B with the team's proxy):

```python
def proxy(theta):            # theta: (m, 4) float, real units
    ...                      # returns (m, 37, 2): cumulative oil, water [STB]
proxy.sigma_proxy = np.array([oil_err, water_err])   # shape (2,), share of final value
```

Order of columns: `fault_trans, poro_mult, perm_mult, aquifer_pv`. Without `sigma_proxy` the notebook assumes a perfect proxy and the posterior is too confident.

**Note on the real use of proxy predictions.** Real simulated curves already exist up to 2008 for all 100 cases. The proxy output matters for (1) dates after 2008, and (2) method B on new settings.

---

## 9. Leakage and bias

**In plain words.** The observed history never enters a trained model, so there is no leakage of the answer. The risks are smaller.

| Item | Type | Note |
|---|---|---|
| T1 "in-pool" test | Leakage inside our own test | The fake history is a pool case, and the stand-in proxy trained on it. B looks better than it is. Trust the **out-of-pool** tests. |
| `rho = "auto"`, 1% accept level, `IMPL_SD` | Mild | Set using the real history. They are reporting choices, not selection inputs. State this. |
| Default method `A` | Mild | Chosen from tests that use validation cases as fake histories. Those cases are also candidates in the final pool. |
| Proxy accuracy mix | Bias | Train cases are in-sample for the proxy (near exact). Validation and test cases carry real error. Use real simulated curves for the window, proxy only for what must be predicted. |
| Stand-in proxy ranks its own training cases | Bias | Replace with the team's proxy. |
| Range (edge) bias | Bias | Prior and proxy stop at the train range. Ziyang's best fit sits on the edge for 3 of 4 settings. Our v2 search warns about it. |
| Pool-only choice | Limit | The best match cannot beat the best pool case (about 0.49%). |
| Selection optimism | Evaluation bias | We choose the cases that fit one history, then report that fit. Do not present it as accuracy. |
| v2 hold-out | Not leakage | Top 10 are chosen on early data only. Power is limited because cumulative errors are correlated. |

**For grading.** The forecast after 2008 is probably what is judged. Our selection uses only data up to 2008, so nothing about the future leaks in.

**Rules to keep.** Keep test cases 86-100 locked. Freeze settings before scoring them. Score them once.

---

## 10. Is the posterior used downstream?

Not yet. Today it is a result, and no later step reads it.

| Possible use | Status | Value |
|---|---|---|
| Forecast uncertainty (P10 / P50 / P90 to 2028) | Planned (`solution_plan.md`, Step 5). Not built. | Highest. The 20-year forecast has no answer key. |
| Choosing the 10 | Ziyang: yes. Ours: only with `method="B"` or `"hybrid"`. | Only if the 10 are new parameter sets, or the hybrid. |
| Deck and video | Not built | Supports the story: the history fixes porosity only, many cases fit equally well. |
| Dashboard | Not planned | Possible: parameter ranges and curve bands. |
| Calibrate `accept_nrmse` and `IMPL_SD` | Open | Needs the real proxy output. |

Both pipelines save the posterior (`posterior_samples_<tag>.csv`, `hm_posterior.npz`). The Excel template has one series per tab and no place for bands, so the posterior is not required to fill it.

---

## 11. Open decisions

1. **What are the 10?** Existing cases, new parameter sets, or a hybrid (pool cases that lie inside the posterior region and differ from each other)? You said "the top 10 cases of production data for submission". The organisers have not confirmed it.
2. **Diversity of the 10.** Keep `min_sep = 0.05` as a safeguard only, or separate by curve or forecast?
3. **Forecast bands.** Carry the posterior to 2028 as P10 / P50 / P90 in the deck or dashboard?
4. Create `solution/histmatch_input/case_params.csv` (needs a go-ahead).
5. Replace the stand-in proxy with the real one, rerun v1 and v2, then calibrate `accept_nrmse` (1%) and `IMPL_SD` (1%).
6. Fix wording in `solution_plan.md`: window start (1999-01-01, not Oct 1998) and "gas = 0.3633 x oil" (about, within 0.1%).
7. The notebook text refers to a `README.md` in the input folder. The file is `histm_README.md`.

---

## 12. Corrections made during the discussion

| Earlier statement | Corrected to |
|---|---|
| Matching window starts "Oct 1998" | Starts 1999-01-01 (37 dates) |
| Gas / oil is "fixed at 0.3633" | About 0.3633, within 0.1% (up to 0.08% lower later in some cases) |
| `case_params.csv` comes from the proxy team | It comes from `cases.csv`. We create it. |
| Ziyang's notebook gives oil twice the weight of water | Oil and gas are the same curve, but water drives 76% of the squared error, so the effect is smaller |
| `min_sep` gives variety in the top 10 | It never triggers on the real cases. The top 10 already differ in parameters, but not in prediction. |
