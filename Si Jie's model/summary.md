# DWC26 code repository: summary for review

14 October 2026. Repository state: `main` at commit bab8d9b. The code is due on 16 October 2026.

You can review this document without opening any other file. It explains:

- the project;
- what was built in Tasks 0 to 3, and how;
- the results;
- every place where the work departs from the requirements document (the PRD);
- the known limitations;
- what remains, and what needs your decision.

Labels such as D-019 are entries in the project's decision log. They are included only so that each point can be traced, and this document states what each entry says. Technical terms are defined where they first appear and again in Appendix A.

## At a glance

- **Progress.** Tasks 0 to 3 are done and merged. Task 4 (history matching) and Task 5 (forecast) have not started. They are due on 15 and 16 October.
- **Proxy choice.** The proxy model is chosen and frozen:
  - Primary: an LSTM network.
  - Secondary: a GRU recurrent network.
- **Test result.** The 15 test cases were held back and scored once, after the freeze. The primary's error was 44 STB/d for oil and 34 STB/d for water, on field rates of 5,000 to 15,500 STB/d. This is 12% above its validation error, and it is still the best of all models.
- **Tests.** All 420 automated tests pass.
- **Main weakness.** The primary's oil forecast declines only about half as fast after 2010 as the history trend suggests. This has to be disclosed with the forecast. It cannot be fixed by changing the model, because the choice is final once the test cases have been scored.
- **Your decisions** (section 14):
  1. Sign off the departures from the PRD that the researcher made.
  2. Decide whether to refit the proxy on all 100 cases before Task 4. The default is no refit.
  3. Accept the forecast limitations.
  4. Settle two documentation fixes.

## 1. The project

### Goal

The repository builds a fast surrogate model (a *proxy*) for a reservoir simulator. It then uses the proxy for two things:

- finding parameter sets that reproduce the field's observed production (*history matching*);
- continuing the production rates of those sets to 2028.

| Step | Task | Content |
| --- | --- | --- |
| Agents | 0 | Set up the two AI agents that write and review the work |
| Data | 1 | Load, check, align and scale the simulation cases and the field history |
| Proxy | 2 | Train time-dependent proxies on simulated cases 1 to 70 |
| Choice | 3 | Validate the proxies, choose and freeze one, score the test cases once |
| History match | 4 | Sample the parameter posterior with MCMC (Markov chain Monte Carlo) using the proxy, and select ten matched parameter sets |
| Forecast | 5 | Continue the same proxy for the ten sets: quarterly oil, gas and water rates from 1 Apr 2008 to 1 Jan 2028 |

**Out of scope:**

- the grid files 03 to 08;
- converting the rates to volumes and writing file 09 (a later task);
- slides, dashboards and video.

### Data

File 01 holds four uncertain parameters for each of 100 simulation cases. The PRD gives their meaning. The file gives no units and no prior ranges, so the ranges below are the minimum and maximum of the 100 cases (see section 5).

| Parameter | Meaning | Range in the 100 cases |
| --- | --- | --- |
| `fault_trans` | Fault transmissibility | 0.050 to 0.150 |
| `poro_mult` | Porosity multiplier | 0.801 to 1.494 |
| `perm_mult` | Permeability multiplier | 0.521 to 9.785 |
| `aquifer_pv` | Aquifer pore volume | 54.0 to 199.8 |

File 02 holds the oil, gas and water rates and cumulative volumes of each case, on the simulator's own report dates (11,096 date rows), and the observed field history, `RAMP_History`.

**The field history.** Below, the rate at a date is the mean over the quarter that ends on that date (section 5 explains why).

| Date | Oil (STB/d) | Gas (MSCF/d) | Water (STB/d) | Water cut |
| --- | --- | --- | --- | --- |
| 1 Jan 1998 to 1 Oct 1998 | 0 | 0 | 0 | — |
| 1 Jan 1999 (first producing quarter, Q4 1998) | 15,508 | 5,634 | 0 | 0% |
| 1 Jan 2001 | 15,211 | 5,526 | 1,051 | 6% |
| 1 Jan 2003 | 13,901 | 5,050 | 3,504 | 20% |
| 1 Jan 2005 | 8,981 | 3,263 | 5,385 | 37% |
| 1 Jan 2008 (last history date) | 5,247 | 1,906 | 7,782 | 60% |

- Oil stays on a plateau of about 15,200 to 15,500 STB/d until mid-2001, then declines.
- Water appears in mid-1999 and rises steadily.
- Gas is a fixed 0.363 × oil throughout.
- At 1 Jan 2008, the cumulative production is 38.4 MMSTB of oil and 12.8 MMSTB of water.

### Time grid and split

| Item | Definition |
| --- | --- |
| History grid | 41 quarterly dates, 1 Jan 1998 to 1 Jan 2008 |
| Match window | 37 dates, 1 Jan 1999 to 1 Jan 2008 (the producing quarters) |
| Forecast grid | 80 quarterly dates, 1 Apr 2008 to 1 Jan 2028 |
| Training cases | 1 to 70. All model fitting |
| Validation cases | 71 to 85. Tuning and model choice |
| Test cases | 86 to 100. Scored one time only, after the model choice is frozen |

The PRD sets 13 validity rules, V1 to V13, such as "fit only on cases 1 to 70" and "score the test cases one time only". Section 11 lists them in full, with their status.

### Units

- STB/d: stock-tank barrels per day, for oil and water.
- MSCF/d: thousand standard cubic feet per day, for gas.
- GOR: gas-oil ratio, in MSCF/STB.

## 2. Status

| Task | Content | Status | Merge commit on `main` | Reviews |
| --- | --- | --- | --- | --- |
| 0 | Two agents, project instructions, docs scaffold | Done | c2c1050, 61a8f03 | — |
| 1 | Data preparation | Done | aa8432b | Approved at the first review |
| 2 | Proxy training: RNN, LSTM, DMD, benchmarks B0 and B1 | Done | d7015a6 | Returned once (the DMD forecast went negative), then approved |
| 3 | Validation, backtest, selection, freeze, single test scoring | Done | 4e01c57, tagged `freeze-proxy` | Returned once (a bug in a test-set safeguard), then approved |
| 4 | History matching: MCMC, ten matched sets, parameter-recovery test | Not started | — | Due 15 Oct |
| 5 | Rate forecast to 1 Jan 2028, and the one-command run `all` | Not started | — | Due 15 to 16 Oct |

The PRD schedule has been met so far. The proxy choice was due to be frozen on 14 October, and it was frozen on 14 October.

## 3. How the work was organised

The work was split between two Claude Code subagents:

| Agent | Writes | Cannot |
| --- | --- | --- |
| Researcher | One method spec per task: inputs and outputs, equations, defaults, acceptance tests. Review findings. The decision log | Write anything outside `docs/`. Run code (it has no shell) |
| Data scientist | Code, tests, configuration files, run logs | Change a spec without the researcher's approval |

Each task followed the same cycle:

1. The researcher writes the spec. Each equation is marked either "source: (article)" or "assumption".
2. The data scientist implements it on a branch `taskN`, writes the tests, runs them and logs each run.
3. The researcher reviews the code against the spec and rules V1 to V13. Each finding is labelled **blocker**, **required** or **note**.
4. Any blocker sends the work back to step 2. With zero blockers, the main session merges the branch into `main` (as a merge commit) and the researcher logs the decision.

The main session coordinated the agents and carried out the one-time steps itself: tagging `freeze-proxy`, running `freeze`, and running the test scoring. The data scientist was told never to run them.

**Tests at each merge:**

| Task | Passed |
| --- | --- |
| 1 | 104 |
| 2 | 335; 338 with the slow tests |
| 3 | 420; 421 with the slow tests |
| Rerun on `main` today | 420; the 3 slow tests were skipped (188 s) |

**Reviews that returned work.** Each of these two reviews found one blocker, and both blockers were real faults:

1. **Task 2, DMD forecast.** DMD (dynamic mode decomposition) is one of the three proxy types. As the PRD specifies it, the DMD fitted the oil decline as part of a very slow oscillation. Continued past 2008, its oil rate fell below zero in all 15 validation cases. The method was changed (section 6), and the fix was approved at the second review.
2. **Task 3, test-set safeguard.** With the default setting, the safeguard code ran git from a subfolder, so git misread the file paths. As a result, the check that the scored code equals the frozen code could never fire, and `freeze` could never succeed. The tests had missed it because they always set the repository path explicitly. It was fixed, and a new test was added, before the single test scoring.

## 4. Task 0: agents

- Created the two agent definitions and the project instructions (rules V1 to V13, key facts, commands), plus the docs scaffold: the specs folder, the decision log and the reference list.
- Converted the three articles that the PRD cites to text, so that the researcher can quote page and equation numbers. The PDFs and the text copies are not committed. The articles are:
  - **Article 2:** Zhang et al. (2023), LSTM proxy with Bayesian history matching, *SPE Journal* 28(3).
  - **Article 4:** Thenon et al. (2022), kriging for history matching, *Computational Geosciences* 26(5).
  - **Article 6:** Abdollahfard et al. (2025), proxy evaluation in waterflooding optimisation, *Scientific Reports* 15.
- Two set-up choices:
  - The workspace folder `dwc/` is the repository root, because Claude Code loads its agents from the workspace root (D-001).
  - The researcher was given write tools, limited to `docs/`. The PRD asks it to write files but lists no write tool for it (D-002).

## 5. Task 1: data preparation

`python -m dwc26 prepare` does the following, in about 15 s:

1. Reads files 01 and 02.
2. Runs checks C1 to C10.
3. Aligns all 100 cases and the history to the 41 quarterly dates.
4. Splits the cases (train, validation, test).
5. Scales the data.
6. Writes one SQLite database, `data/processed/dwc26.sqlite`, which all tasks use, plus an array file and a check report.

The current data run is `prepare-20261012T201436717380Z`. Every stop check passes.

### Checks

C1 to C7 are in the PRD. C8 to C10 were added (D-006).

| Check | Tests | If it fails | Result |
| --- | --- | --- | --- |
| C1 | Structure: all 6 sheets have cases 1 to 100 and `RAMP_History`; units; dates | Stop | Pass |
| C2 | Each case covers 1 Jan 1998 to 1 Jan 2008 | Stop | Pass |
| C3 | Cumulatives never decrease; rates ≥ 0 | Stop | Pass |
| C4 | The `Field Production` sheet equals `RAMP_History` (tolerance 1e-6) | Stop | Pass, difference 0 for all fluids |
| C5 | Each case reports on all 41 history dates | Interpolate | Pass, so no interpolation was needed |
| C6 | Which quarter the reported history rate averages | Selects the rate definition | It is the mean over the **next** quarter (difference 2.5e-8), not the previous quarter (difference 1.0). See "Rate definition" below |
| C7 | GOR per case and date | Report only | 0.3605 to 0.3654 MSCF/STB in every case and date, median 0.3633 |
| C8 (new) | Cumulative = 0 at the first date | Stop | Pass |
| C9 (new) | How X was sampled | Report only | Uniform and independent. Uniformity test p ≥ 0.89 for all four parameters. Log-uniform rejected for `perm_mult` (p = 7e-11). Largest rank correlation between parameters 0.17, below the 5% critical value 0.197 |
| C10 (new) | Does the history lie inside the range of the training cases on every date? | Report only | Yes. Without a tolerance, 2 oil and 2 gas dates were flagged, but the excess is at most 8.3e-7 of the value, which is float rounding in the file. With the tolerance of 1e-5 (D-009), no dates are flagged |

### Rate definition: the main data finding (D-004, D-005)

The two sources do not report rates the same way:

- The **cases** report the simulator's rate at its last time step before each date.
- The **history** reports, at each date, the mean rate over the *following* quarter.

The PRD's check C6 tests the previous-quarter mean. Here it fails, and the PRD would then pick rule R2 (the last reported rate). R2 gives the cases an instantaneous rate and the history a quarterly mean. These are two different quantities, which breaks rule V4 (the cases and the history must be processed the same way).

What was done instead:

- **One definition for both.** For the cases and for the history, the rate is computed from the cumulatives as Δcumulative ÷ Δtime over the previous quarter.
- **The first date.** The rate there is 0, because the cumulative is 0 (new check C8). The PRD said to use the reported rate at the first date, but that is an instantaneous simulator rate.
- **What a date's value means.** The value at a date is the mean over the quarter that ends on it, and the history value at a date equals the reported history rate one quarter earlier.
- **The match window.** The 37 dates from 1 Jan 1999 to 1 Jan 2008 cover the 37 producing quarters, Q4 1998 to Q4 2007.

This decision needs your sign-off (section 14).

### Scaling

| Quantity | Method | Values |
| --- | --- | --- |
| X (4 parameters) | Min-max to [0, 1] | Design minimum and maximum of the 100 cases (table in section 1) |
| Oil, gas, water rates | Divide by the maximum over training cases 1 to 70, all dates | Oil 15,678 STB/d, gas 5,696 MSCF/d, water 9,167 STB/d |
| History | The same factors as the cases | — |

### Findings that shaped later tasks

- **No prior ranges are given (D-003).** File 01 has no parameter ranges, and you confirmed that the kick-off material gives none. The bounds are therefore the minimum and maximum of the 100 design points. Consequence: the history match cannot leave the range of the simulated cases, so the proxy is never asked to extrapolate in X.
- **The parameters were sampled uniformly (D-008).** This includes `perm_mult`, which might have been sampled log-uniformly (check C9). The Task 4 prior is therefore uniform on the design box, with no log transform.
- **Gas copies oil (D-010).** GOR is 0.3605 to 0.3654 everywhere, so in scaled units gas equals oil within ±0.8%. A loss with equal weights for oil, gas and water therefore weights oil about ⅔ and water about ⅓. Task 2 consequently made the fluid weights a tuned option (section 6).
- **Production starts at date index 4 (Q4 1998) in every case (D-013).** All models fit from there and return exactly 0 before it.

## 6. Task 2: proxy training

`python -m dwc26 train` fits each model on cases 1 to 70 and tunes it on cases 71 to 85.

### Models

The PRD allows only proxies that keep time as a dimension, so that the same model can be continued past 2008 (rule V13).

| Model | Strategy | How it works | Role |
| --- | --- | --- | --- |
| RNN | S3: sequence network | A recurrent network that steps through the quarters and outputs the three rates at each step. The cell type (plain RNN or GRU) is tuned | Proxy |
| LSTM | S3: sequence network | The same, with LSTM cells | Proxy |
| DMD | S4: shared dynamics | A set of decay curves in time, shared by all cases. A Gaussian process maps X to each case's weights on those curves | Proxy |
| B0 | Benchmark | The mean training curve, the same for every case | Benchmark only |
| B1 | Benchmark | k-nearest neighbours in X (k = 3, distance-weighted) | Benchmark only |

**Input options.** The RNN and the LSTM work through the field's life one quarter at a time. At each quarter they update an internal memory (the *state*) and output that quarter's oil, gas and water rates. To do this the network needs two facts: **which reservoir** it is simulating (the four parameters X, like a description card) and **what time it is** (a calendar reading t). The three input options differ only in how these two facts are given:

| Option | The description card (X) | The calendar (t) | In plain words |
| --- | --- | --- | --- |
| **A** | Shown at every quarter | Shown at every quarter | The network is reminded of both facts at every step |
| **B** | Shown once, before the first quarter, to set the starting memory | Shown at every quarter; it is the only input | The network is briefed once and must remember which reservoir it is; after that it sees only the calendar |
| **C** | Shown at every quarter | Never shown | The network knows the reservoir but has no calendar; it tells the time by counting how many quarters it has stepped through |

The calendar reads t = step ÷ 40: 0 on 1 Jan 1998, 1 on 1 Jan 2008 (end of history) and 3 on 1 Jan 2028 (end of forecast). In training the network only sees readings between 0 and 1. In the forecast, options A and B are shown readings from 1.025 to 3 that they have never seen, much like a student asked about years past the end of the textbook. Option C avoids unfamiliar inputs, but it must keep counting for 80 quarters more than it practised, so its memory also moves into states it never reached in training. All three must extrapolate in time; they differ in where the extrapolation happens. The chosen primary (LSTM A) and secondary (RNN A) both use option A.

Each option is tuned and kept as a separate model (D-014). A trained model is named by a *proxy ID*, `<run_id>/<variant>`, for example `train-20261013T000744561060Z/A`.

### Training and tuning

- **Loss.** Mean squared error on the scaled rates, with fluid weights for oil, gas and water.
- **Optimiser.** Adam, with gradient clipping at 1.0.
- **Early stopping.** On the validation score J (defined in section 7), with a patience of 50 epochs and a maximum of 1,000.
- **Search (D-015).** 30 trials per model, chosen by a seeded Latin-hypercube design: 10 trials for each input option.
  - Number of layers: 1 or 2.
  - Units: 16, 32, 48 or 64.
  - Learning rate: 3e-4 to 1e-2, log scale.
  - Batch size: 4, 8, 16 or 32.
  - Fluid weights: 0.25 / 0.25 / 0.5 for oil / gas / water (the default, D-012, which you approved on 13 Oct), or equal (the PRD).
  - For the RNN only: RNN or GRU cell.
- **Seeds.** Each trial trains one seed. The final model of each option is the mean of a 5-seed ensemble (seeds 0 to 4), and the spread across the seeds is its uncertainty band.
- **No refit.** The final model is the one trained on cases 1 to 70.

**Settings chosen:**

| Model | Layers | Units | Learning rate | Batch | Fluid weights | Cell |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM A (**primary**) | 2 | 32 | 0.0040 | 4 | Equal | LSTM |
| LSTM B | 2 | 32 | 0.0099 | 8 | Equal | LSTM |
| LSTM C | 2 | 48 | 0.0076 | 16 | Equal | LSTM |
| RNN A (**secondary**) | 2 | 16 | 0.0096 | 4 | 0.25 / 0.25 / 0.5 | GRU |
| RNN B | 1 | 48 | 0.0020 | 4 | 0.25 / 0.25 / 0.5 | GRU |
| RNN C | 2 | 48 | 0.0010 | 8 | 0.25 / 0.25 / 0.5 | GRU |

The validation score chose equal weights for all three LSTMs, so **the primary uses the PRD's equal weights**. The RNNs all chose the 0.25 / 0.25 / 0.5 default.

### DMD: what went wrong and what changed (D-016, D-019, D-022)

**What the PRD specifies:**

- "exact DMD" with the rank set to keep 99% of the energy;
- a Gaussian process from X to the coefficients;
- the forecast y(x, t) = Re Σ c_k(x) λ_k^t.

**What happened.** At rank 4, exact DMD gave two pairs of complex eigenvalues. The slow pair had a period of 399 quarters, more than ten times the 37 fitted quarters. The method had fitted the decline as the start of a very long oscillation, so when it was continued past 2008, oil went below zero in all 15 validation cases, between 2009 and 2012. In addition, the 99% energy rule gave rank 2 at both energy settings, so 16 of the 24 tuning trials were duplicates.

**The amendment (approved at the Task 2 re-review):**

- **Real decay rates only.** Each time mode is λ^t with λ a real number between 0 and 1. Sums of such modes cover the standard exponential and hyperbolic (Arps) decline curves, and they cannot oscillate.
- **Fitting.** The decay rates are fitted to all the training curves at once, by variable projection, starting from the exact-DMD solution.
- **Rank.** The rank is tuned directly over 2, 3, 4, 5, 6 and 8, instead of being set by the 99% energy rule.
- **Tuning gate.** Tuning also rejects any setting whose 80-quarter forecast goes negative.
- **Other fixed details.**
  - One Gaussian process per fluid and coefficient, with a Matérn 5/2 kernel (after Article 4).
  - A grid of 24 settings: rank × fluid weights × GP target.
  - Eigenvalues above 1 are moved to the unit circle.

**The result.** The amended DMD chose rank 8. One mode has λ = 0.99999924, which is effectively constant. The other seven decay fast (half-lives of at most 2.7 quarters), so after about 2010 the DMD forecast is almost flat: oil changes by at most 9.8% from 2010 to 2028. The researcher accepted this for the merge, on two conditions: DMD may only ever be the secondary proxy, and the flat tail must be stated wherever it is used. In Task 3, DMD then failed the backtest and was not chosen at all (section 7).

### Run times and speed

The laptop has 8 cores and no GPU.

| Model | Training run |
| --- | --- |
| RNN | 460 s |
| LSTM | 136 s |
| DMD | 147 s |
| B0, B1 | Under 1 s each |

The PRD requires that a proxy can make 160,000 predictions (32 MCMC walkers × 5,000 steps) in under 10 minutes on a laptop. Every proxy passes: the primary needs 21 s, and the slowest variant needs 68 s.

## 7. Task 3: validation, backtest, selection, freeze and test

The commands are:

1. `evaluate --split val`: scores all models on the validation cases.
2. `backtest`: retrains the RNN, LSTM and DMD on the data up to 2005.
3. `select`: applies the selection rule.
4. `freeze`: records the choice.
5. `evaluate --split test`: scores the test cases, once.

The PRD places the backtest in Task 5, but it also requires the backtest before the freeze, so it was built here (D-026).

### Measures

| Measure | Definition |
| --- | --- |
| RMSE | Root mean squared error over all cases of a split and all 41 dates, in STB/d or MSCF/d |
| NRMSE | RMSE ÷ (maximum − minimum of the true values of that fluid in the split) |
| **J** (selection score) | Mean of the oil and water NRMSE on the validation cases. Lower is better. Gas is left out because it equals 0.363 × oil |
| R² case | 1 − (squared error) ÷ (squared deviation from the training mean curve at the same date). It shows whether the model separates the cases, beyond the time trend that all cases share. 1 is perfect |
| Mismatch O | The history-matching objective of Task 4, computed with σ_proxy = 0, on oil and water over the 37 match dates (formula in section 13) |
| ρ | Spearman rank correlation, over the cases of a split, between O computed from the proxy's curves and O computed from the simulated curves. It shows whether the proxy ranks cases by closeness to the history correctly |
| Coverage | Share of the true values inside the 95% band of the 5-seed ensemble (mean ± 1.96 × the spread). Target 0.90 to 0.98 |
| **Backtest** | Each model is retrained on the 29 dates up to 1 Jan 2005 with its tuned settings, continued 12 quarters to 1 Jan 2008, and compared with the simulated curves of the validation cases |
| B2 (persistence) | Benchmark for the backtest: each case's last rate before the cutoff, held constant |
| **BT** (tie-break score) | Mean of the oil and water backtest RMSE over horizons 1 to 12, ÷ the range of the true values |
| Event errors | Error in days of the end of the oil plateau, water breakthrough, and the water-oil crossover. The resolution is one quarter |

### Plausibility rules added to the selection (D-020, D-022)

The validation error covers only 1998 to 2008, so it cannot see how a model behaves after 2008. Three rules check the 80-quarter continuation of the validation cases:

| Rule | Test | Effect |
| --- | --- | --- |
| P1 | Any negative rate (clipped to 0) in the continuation | Rejects the model |
| P2 | Oil rises above 1.05 × the model's own 2008 rate in more than half of the validation cases | Rejects the model, because you confirmed on 14 Oct that there are no new wells and no development after 2008 (D-031) |
| P3 | R = median over the cases of (relative oil decline per quarter in forecast quarters 9 to 16, i.e. 2010 to 2011) ÷ (the same in the last 8 history quarters) | Flag only, if R < 0.5. Under Arps decline at the observed 2 to 3.5% per quarter, R should not fall below about 0.64 |
| GOR | The forecast GOR against the training range 0.3606 to 0.3654 (tolerance 0.5%) | Report only |

### Selection rule, as implemented (D-027)

1. **Benchmarks on validation.** A model must beat B0 and B1 on validation RMSE, for oil and for water (PRD step 1).
2. **Backtest.** It must beat B2 at horizons 9 to 12, for oil and for water, using the RMSE pooled over those four horizons (PRD step 2; the pooling is D-026).
3. **Continuation.** It must pass P1 and P2 (added).
4. **Ranking.** The eligible models are ranked by J. Among those within 5% of the best J, the one with the lower BT wins; among those within 5% of the best BT, the higher ρ wins (PRD steps 3 and 4).
5. **Primary.** The primary is the top-ranked model. DMD may not be the primary (added, because of its flat tail).
6. **Secondary.** The secondary is the best eligible model of the other strategy (PRD step 5). If there is none, it is the best eligible model of the other recurrent family (added fallback).

### Results: accuracy

Validation cases decided the choice. The test values were computed once, after the freeze, and are for reporting only. The RMSE is in STB/d.

| Model | J val | J test | Oil RMSE val / test | Water RMSE val / test | R² case oil, val | ρ val | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM A | 0.00291 | 0.00325 | 36 / 44 | 31 / 34 | 0.998 | 0.996 | **Primary** |
| LSTM C | 0.00347 | 0.00383 | 55 / 61 | 31 / 34 | 0.996 | 0.993 | Rejected by P2 |
| RNN A (GRU) | 0.00382 | 0.00406 | 65 / 71 | 31 / 33 | 0.995 | 1.000 | **Secondary** |
| RNN B (GRU) | 0.00445 | 0.00467 | 77 / 83 | 36 / 37 | 0.993 | 0.993 | Eligible, not chosen |
| LSTM B | 0.00464 | 0.00555 | 67 / 87 | 45 / 50 | 0.995 | 0.996 | Eligible, not chosen |
| RNN C (GRU) | 0.00489 | 0.00513 | 82 / 87 | 41 / 43 | 0.992 | 0.989 | Eligible, not chosen |
| DMD | 0.00928 | 0.00963 | 129 / 134 | 93 / 97 | 0.980 | 0.993 | Rejected: fails the backtest |
| B1 (kNN) | 0.0199 | 0.0286 | 239 / 395 | 220 / 290 | 0.932 | 0.839 | Benchmark |
| B0 (mean curve) | 0.0710 | 0.0736 | 918 / 999 | 749 / 755 | 0 | — | Benchmark |

- Every proxy beats both benchmarks by a wide margin, on validation and on test.
- Test J is 4 to 20% above validation J for the proxies. That is normal, because J was the tuning objective. The primary is 12% above, and it is still the best model on test: its test J is 15% below LSTM C's and 20% below RNN A's.
- **Event timing of the primary (validation).** The mean absolute error is 31 days for the plateau end, 12 days for water breakthrough and 18 days for the water-oil crossover. One quarter is about 91 days.

### Results: forecast behaviour (validation cases)

The backtest columns are RMSE pooled over horizons 9 to 12, in STB/d.

| Model | Backtest oil | Backtest water | BT | P1: negative values | P2: rising cases | P3 R | Forecast GOR | 95% coverage oil / water |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM A (primary) | 662 | 501 | 0.036 | 0 | 0 of 15 | **0.50, flagged** | 0.345 to 0.370 | 0.87 / 0.81 |
| LSTM C | 697 | 561 | 0.037 | 0 | **15 of 15**, up to 1.50 × | 0.23, flagged | 0.356 to 0.395 | 0.69 / 0.83 |
| RNN A (secondary) | 386 | 217 | 0.018 | 0 | 0 of 15 | 0.89 | 0.235 to 0.365 | 0.74 / 0.89 |
| RNN B | 784 | 514 | 0.036 | 0 | 0 of 15 | 0.52 | 0.259 to 0.363 | 0.71 / 0.72 |
| LSTM B | 333 | 378 | 0.023 | 0 | 0 of 15 | 0.69 | 0.362 to 0.381 | 0.71 / 0.79 |
| RNN C | 825 | 536 | 0.042 | 0 | 0 of 15 | 0.20, flagged | 0.361 to 0.372 | 0.65 / 0.70 |
| DMD | **11,899** | **15,659** | 0.793 | 0 | 0 of 15 | 0.05, flagged | 0.361 to 0.365 | 1.00 / 0.99 |
| B2 (persistence) | 2,680 | 1,309 | 0.118 | — | — | — | — | — |

- All six recurrent models beat persistence (B2) in the backtest. DMD does not.
  - Retrained on only the 29 dates up to 2005, all of its time modes decay at 0.79 to 0.89 per quarter, so its forecast collapses towards zero (77 values clipped).
  - The method uncertainty in Task 5 therefore comes from two recurrent networks only.
- LSTM C forecasts rising oil in every validation case, up to 1.5 × its 2008 rate. That is impossible in a field without new wells, so it is rejected, although it has the second-best J.
- Coverage is below the 0.90 target for every recurrent model, so the seed-ensemble bands are too narrow. Task 4 does not use these bands (section 13).

### Selection

- **The primary.** The best J is 0.002906 (LSTM A). The 5% band reaches 0.003051, and no other model is inside it. The primary is therefore decided by J alone, and the backtest and ρ tie-breaks did not apply.
- **The secondary.** The PRD asks for a secondary of the other strategy, which here can only be DMD. DMD fails the backtest, so the fallback gives the best eligible model of the other recurrent family: RNN A.
- **Ranking of the eligible models:** LSTM A, RNN A, LSTM B, RNN B, RNN C. LSTM B ranks above RNN B because their J values are within 5% of each other and LSTM B has the lower BT.

**A point to weigh.** The primary has the lowest error over 1998 to 2008, but it does not have the best forecast evidence:

- In the backtest, RNN A and LSTM B forecast 2005 to 2008 better (BT 0.018 and 0.023, against 0.036).
- The primary carries the P3 flag, and RNN A does not.

The PRD rule lets the backtest decide only between models whose J values are within 5% of each other, and none were. Task 5 will report the secondary's forecast beside the primary's.

**The frozen choice (D-034):**

| Role | Proxy ID | Model |
| --- | --- | --- |
| Primary | `train-20261013T000744561060Z/A` | LSTM, input option A, 2 layers × 32 units, equal fluid weights, 5-seed ensemble |
| Secondary | `train-20261012T235952401450Z/A` | GRU, input option A, 2 layers × 16 units, fluid weights 0.25 / 0.25 / 0.5, 5-seed ensemble |

### Inputs fixed for Task 4

Task 4 measures the misfit to the history with σ_obs, the observation error, and σ_proxy, the proxy error (formula in section 13). The values below are in STB/d.

| Quantity | Oil | Water |
| --- | --- | --- |
| σ_obs: 10% of the history rate, with a floor of 2% of its maximum | 525 to 1,551 | 156 to 778 |
| σ_proxy: the primary's validation RMSE at each date | 13 to 59 (median 36) | 9 to 70 (median 26) |

- The observation error dominates. On any match date, the proxy error adds at most 1% to σ² for oil and at most 5% for water. The medians are 0.1% and 0.4%.
- Several simulated cases already match the history well. With the simulated curves, 8 of the 15 validation cases and 7 of the 15 test cases have O below the PRD's acceptance limit of 95.1. The best are case 73 (O = 14.8) and case 90 (O = 16.7).

### Freeze and the single test scoring (D-028)

Rule V3 allows the test cases to be scored only once, and only after the choice is frozen and tagged. Two mechanisms enforce this.

**In the code.** `freeze` and `evaluate --split test` refuse to run unless guards G1 to G7 hold:

| Guard | Condition |
| --- | --- |
| G1 | The tag `freeze-proxy` exists |
| G2 | The current commit contains the tagged commit |
| G3 | The working tree has no uncommitted changes |
| G4 | No file in the frozen paths has changed since the tag. The frozen paths are the model code, the evaluation code, the backtest, the date code, the mismatch function, and the model and evaluation configs |
| G5 | For freeze: the decision log at the tag names both proxy IDs. For test: the freeze record points to the same tagged commit |
| G6 | Once only: there is no earlier freeze, and no earlier test scoring. A failed attempt counts |
| G7 | The test run uses the same metric settings as the validation run |

**In the database.** SQLite triggers allow one freeze and one test scoring, and they refuse test rows in the metric and prediction tables without an open scoring claim. A second attempt needs an override with a written reason of at least 10 characters, and that override is logged. **No override was used.**

**What was run on 14 October:**

1. The researcher approved Task 3 and wrote both proxy IDs into the decision log.
2. The branch was merged to `main` as commit 4e01c57, and that commit was tagged `freeze-proxy`.
3. `freeze`: run `freeze-20261014T110427868031Z` (freeze 1).
4. `evaluate --split test`: run `evaluate-20261014T110434893087Z`, scoring 1, status ok. It wrote 812 metric rows and 5,535 prediction rows.

The test confirmed the choice (D-035):

- The top three models by J are the same as on validation.
- Every recurrent model again beats B0, B1 and B2.
- DMD again fails the backtest.
- The primary's P3 ratio is 0.495 on test, against 0.496 on validation.

Under the PRD, the choice cannot change after this step.

## 8. Departures from the PRD

The last column shows who made each decision.

- **"Researcher":** the researcher made it in a spec, without you. These need your sign-off. The most significant are marked ★.
- **"User":** you made or confirmed it.

| # | PRD says | Done instead | Why | Decided by |
| --- | --- | --- | --- | --- |
| 1 ★ | Rate definition chosen by check C6 (previous-quarter mean, else the last reported rate). Use the reported rate at the first date | The same Δcumulative ÷ Δtime over the previous quarter for the cases and the history. Rate 0 at the first date, with new stop check C8 (D-004, D-005) | The history reports the next-quarter mean. The PRD's rule would give the cases and the history different definitions (rule V4) | Researcher |
| 2 | X bounds are "the bounds of file 01: the prior range" | The minimum and maximum of the 100 design points (D-003) | File 01 has no ranges; you confirmed none are given | Researcher; the fact confirmed by you |
| 3 | Checks C1 to C7 | C1 extended; C8 to C10 added; C10 with a relative tolerance of 1e-5 (D-006, D-009) | Clearer stops. C9 and C10 inform the prior and the match. The tolerance absorbs float rounding in the file | Researcher |
| 4 | "Equal weight for oil, gas and water" in the loss | The default is 0.25 / 0.25 / 0.5, and equal weights remain a tuned option (D-012). The LSTMs, the primary included, chose equal | Gas = 0.363 × oil, so equal weights count oil twice | **User (approved 13 Oct)** |
| 5 | Models use all 41 dates; the DMD matrix has 41 columns | Fit from production start (date index 4); 0 before it (D-013) | The 4 zero quarters carry no information and distort DMD | Researcher |
| 6 | The input option is "a config option"; commands take `<run_id>` | Each option is tuned (10 trials each) and kept as its own model; models are named `<run_id>/<variant>`; RNN cell type tuned (D-014) | Validation cannot see how the options extrapolate in time, so Task 3 must be able to choose among them | Researcher |
| 7 | 30 trials per model (no method given); 5 seeds per model | Latin-hypercube search; 1 seed per trial; final 5-seed ensemble; DMD uses all 24 settings of its grid; early stopping on J (D-015) | No new dependency (Bayesian optimisation samples at random for its first 10 trials anyway); 42 neural fits per model instead of 150; DMD has fewer free settings | Researcher |
| 8 ★ | Exact DMD at 99% energy, complex eigenvalues | Real decay rates in (0, 1], fitted by variable projection, rank tuned 2 to 8; unstable eigenvalues moved to the unit circle (D-016, D-019) | Exact DMD gave negative oil after 2008 in every validation case | Researcher; approved at re-review |
| 9 | Proxy interface as in PRD §3 | Adds a full-run method, clipping at 0 inside `predict`, and benchmarks that refuse to forecast (D-017) | Task 5 needs a 121-step run; V9 needs rates ≥ 0; V13 forbids forecasting with a benchmark | Researcher |
| 10 ★ | Selection steps 1 to 7; rising oil after 2008 is report only (PRD §9) | P1 and P2 reject a model; P3 flags it; DMD may be secondary only; fallback secondary from the other recurrent family (D-020, D-022) | The validation error cannot see the forecast. LSTM C forecast rising oil in 15 of 15 cases | Researcher; P2 rests on **your** "no new wells" answer |
| 11 | Backtest in Task 5; "beat B2 at horizons 9 to 12" | Backtest built in Task 3; RMSE pooled over horizons 9 to 12 for oil and water; BT defined as the tie-break score (D-026) | The PRD requires it before the freeze. Four separate horizon tests on 15 cases are noisy | Researcher |
| 12 | Tie-break "within 5%" | 5% band measured from the best value, applied to J, then BT, then ρ, with a fixed order for ties (D-027) | The PRD names the keys but not the band reference or the tie order. The result must not depend on input order | Researcher |
| 13 | R² and coverage over all dates; plateau end = first date below 0.98 × maximum | R² and coverage from the production start; plateau end searched after the curve's own maximum (D-024) | The pre-production zeros inflate R² and count as covered. Read literally, the PRD's plateau rule gives date 0 | Researcher |
| 14 | ρ and local RMSE "from Article 6" | Defined here and labelled "assumption"; the mismatch function O built in Task 3 and reused by Task 4; σ_proxy = the primary's validation RMSE by date (D-025) | Article 6 does not define them. One O for both tasks | Researcher |
| 15 | `freeze` writes the choice and tags | `freeze` verifies and records only; the main session tags; guards G1 to G7 and database triggers (D-028) | Fail closed: the database enforces "once only" even if the Python checks are bypassed | Researcher |
| 16 | Database tables of PRD §5 | Extra columns (proxy ID, horizon index, run details) and new tables `selection`, `freeze`, `test_scoring` (D-007, D-018, D-029) | Several models per run, per-horizon values, and a record of the freeze for V3 | Researcher |
| 17 | Repository folder `dwc26/`; the researcher has no write tool | Repository root `dwc/`; the researcher writes only under `docs/` (D-001, D-002) | Tooling requirements | Researcher |

The ★ rows change what the PRD specifies, not only how it is done.

## 9. Limitations carried into Tasks 4 and 5

Each of these will be stated with the forecast (D-034, D-035).

1. **The primary's oil forecast flattens after about 2010.**
   - Its decline in 2010 to 2011 is about half its decline in the last 8 history quarters (P3 ratio 0.50, where the flag is < 0.5).
   - Under Arps decline, a ratio of at least about 0.64 would be expected.
   - The primary may therefore overstate oil after 2010.
   - The secondary does not show this (ratio 0.89).
2. **GOR drifts in the forecast.** In the training data, GOR stays between 0.361 and 0.365 at all times. In the forecast:
   - the primary's GOR ranges from 0.345 to 0.371;
   - the secondary's falls as low as 0.22.

   The gas forecast is therefore less reliable than the oil forecast. A forecast GOR of 0.363 × oil would be the physically consistent value.
3. **The uncertainty bands of the proxies are too narrow.** For the primary and the secondary, the 95% seed-ensemble bands cover 74 to 87% of the test values, against a target of 90 to 98%. Task 4 does not use these bands: its proxy error term is the primary's validation RMSE by date.
4. **The method uncertainty covers recurrent networks only.** The primary and the secondary are both recurrent networks, because DMD failed the backtest. The spread between them understates the uncertainty that comes from the choice of method.
5. **The backtest is limited evidence.**
   - It covers 12 quarters, while the forecast covers 80.
   - Its settings were tuned on data that include the years after the cutoff.
   - It cannot see a stall that starts after 2010.
6. **Every forecast is an extrapolation in time.** Option A, which both proxies use, receives time inputs from 1.025 to 3 in the forecast, against 0 to 1 in training. Rule V13 allows no other forecast method, and no simulated data exist after 2008 to check against.
7. **The history match cannot leave the design box** (departure 2). If the true field lies outside the range of the 100 cases, the match will place it on the edge of that range.

## 10. Issues during the work

| Issue | Effect | Handling |
| --- | --- | --- |
| The article PDFs were staged in the first commit | None. They were unstaged before the commit | `*.pdf` added to `.gitignore`. Checked today: no PDF, Excel, database or array file is in the git history, and there is no remote (rule V12) |
| The data-scientist agent hit API usage limits twice (Tasks 2 and 3) | Delays only | The same agent was resumed with its context. No work was lost |
| Two Open items in the decision log were misfiled as Closed (Task 1) | A wrong record, briefly | Corrected in the same task |
| The researcher has no shell | Its reviews read the code but cannot run it | The test counts come from the data scientist's runs, plus a rerun on `main` today |
| An earlier version of this summary said that the fluid weights are 0.25 / 0.25 / 0.5 | It suggested that the primary uses them; it uses equal weights | Corrected here (section 6) |

## 11. Rule compliance (V1 to V13)

| Rule | Text | Status |
| --- | --- | --- |
| V1 | Fit each learned transform (scalers, DMD basis, model weights) on cases 1 to 70 only. Select settings on cases 71 to 85 only | Met. Checked at each review |
| V2 | Split by case, never by row or date | Met |
| V3 | Use cases 86 to 100 for metrics one time only, after the choice is frozen and tagged `freeze-proxy` | Met. One freeze and one scoring, no override, enforced by guards G1 to G7 and by database triggers |
| V4 | Same alignment, rate definition and scaling for the cases and the history | Met, through departure 1 |
| V5 | No hard-coded data values; bounds, dates and counts are read at run time | Met. A search for data values in the code was clean at each review |
| V6 | Each model beats B0 and B1 on validation and B2 in the backtest | Applied in the selection. DMD fails it and is rejected |
| V7 | Fixed seeds, all settings in YAML configs, pinned package versions | Met. Python 3.13, torch 2.14.1+cpu, scikit-learn 1.7.2, numpy 2.3.5, scipy 1.17.1, matplotlib 3.10.8, with a lock file |
| V8 | Log each run: ID, git commit, config, seed, duration, outputs | Met. Every run is in the `runs` table, with the config hash and an uncommitted-changes flag |
| V9 | Assert shapes and values at each module boundary | Met |
| V10 | A unit test for each module; `pytest` passes before a merge | Met: 420 passed. Note: three helper modules (`splits.py`, `runctx.py`, `report.py`) are tested only through other modules' tests |
| V11 | A parameter-recovery test for the history matching | Task 4 |
| V12 | Do not publish the data or the repository; keep raw files out of version control | Met. Checked today |
| V13 | Each proxy keeps the time component; Step 5 continues the Step 2 model; no other forecast method | Met. The benchmarks refuse to forecast |

## 12. PRD acceptance criteria

| Criterion | Status |
| --- | --- |
| `python -m dwc26 all` reproduces everything from a clean clone and the raw files | **Open.** `all`, `match` and `forecast` print "not implemented yet". Due in Task 5 |
| `pytest` passes | Yes (420 passed) |
| C1 to C4 pass; C5 to C7 reported | Yes |
| Each model beats B0, B1 and B2, or is marked rejected | Yes |
| Test metrics computed once, after the tag `freeze-proxy` | Yes |
| MCMC pass criteria met for 3 seeds, or the run marked "not converged" | Open (Task 4) |
| Parameter-recovery test reported | Open (Task 4) |
| Ten matched sets, each with O ≤ O_max and each pair at least d_min apart | Open (Task 4) |
| Forecast backtest reported | Yes (Task 3) |
| Forecast rates: 80 dates × 3 fluids × 10 sets | Open (Task 5) |
| README with setup, commands, run times and outputs | Done for Tasks 1 to 3 |

## 13. What remains

### Task 4: history matching (due 15 October)

The PRD specifies it as follows. Nothing is built yet, and the sampler package (emcee) still has to be installed and pinned.

**Mismatch.** O(x) is the sum, over oil and water and the 37 match dates, of ((proxy rate − history rate) ÷ σ)². Here:

- σ² = σ_obs² + σ_proxy²;
- σ_obs = max(0.10 × the history rate, 0.02 × the maximum history rate);
- σ_proxy is the primary's validation RMSE at each date (values in section 7).

Gas is left out, because it copies oil.

**Posterior.** log p = −O/2 inside the design box, and −∞ outside it. Sampling is done in the scaled space [0, 1]⁴.

**Sampler.**

- emcee's affine-invariant ensemble sampler, with 32 walkers.
- 1,000 to 5,000 steps: stop when the chain is longer than 50 autocorrelation times.
- 3 runs with different seeds.
- Pass criteria:
  - acceptance fraction 0.2 to 0.5;
  - the 3 runs agree within 0.05 (scaled) on the 5%, 50% and 95% quantiles.
- If the criteria are not met, the run is marked "not converged".

**Ten matched sets.**

- Pool the samples. Drop those within 0.02 of a bound, and those with a high proxy spread.
- Sort by O.
- Pick greedily, keeping each pair at least 0.10 apart (relaxed by × 0.8 if needed).
- Each set must have O ≤ 95.1, the 95% quantile of χ² with 74 degrees of freedom.

**Verification.** Compare the ten sets with the ten simulated cases closest to the history, and recompute O with the secondary proxy.

**Parameter-recovery test (rule V11).** Use 3 test cases as pseudo-histories, and check whether their true parameters fall inside the 90% posterior intervals. This is the only use of test cases that Task 4 is allowed.

### Task 5: forecast (due 15 to 16 October)

- Run the frozen primary and secondary for 121 quarters (41 history + 80 forecast) for each of the ten sets, and keep the last 80.
- Deliverable: 80 dates × 3 fluids × 10 sets, from the primary. The secondary measures the method uncertainty.
- Report each set whose oil rises, the spread across the ten sets, and the primary-against-secondary difference.
- State the limitations in section 9.
- Implement `python -m dwc26 all` and complete the README.

## 14. Decisions for the reviewer

1. **Sign off the departures that the researcher made** (section 8). The three ★ items change what the PRD specifies:
   - **1, the rate definition.** Without it, the cases and the history would be different quantities.
   - **8, the DMD method.** Without it, DMD forecasts negative oil. DMD is not used in the end, but the change is part of the merged code.
   - **10, the plausibility rules.** These rejected LSTM C, the model with the second-best J.
2. **Refit before Task 4.** The PRD leaves this open, with the default of no refit. A refit on all 100 cases would give the match more data, but no independent check of the proxy would remain. **Unless you decide otherwise, Task 4 uses the frozen proxy as trained on cases 1 to 70.**
3. **Accept the limitations in section 9 as disclosures.** The proxy choice is final after the test scoring, so the primary's flat tail after 2010 cannot be fixed by switching models. Task 5 will show the secondary's forecast beside it.
4. **Documentation fixes.**
   - The PRD's command lines (§3) still write `freeze --model <run_id>`, while the code and the project instructions use the proxy ID `<run_id>/<variant>`. Whether the PRD is reworded is your call.
   - Two items under "Open" in the decision log are out of date: the scikit-learn pin is done, and the Task 1 follow-ups were approved at the Task 2 review.

## Appendix A: Glossary

| Term | Meaning |
| --- | --- |
| Proxy | A fast model that reproduces the simulator's rates for given parameters X |
| History matching | Finding parameter sets whose rates match the observed field history |
| MCMC | Markov chain Monte Carlo: random sampling of the parameters in proportion to how well they match the history |
| RNN, GRU, LSTM | Recurrent neural networks that step through time. GRU and LSTM are gated variants that hold memory better |
| DMD | Dynamic mode decomposition: describes all curves as weighted sums of shared time modes λ^t |
| Gaussian process (GP) | A smooth regression with an uncertainty estimate, used inside the DMD proxy |
| Input option A, B, C | How X and time enter the recurrent network (section 6) |
| Ensemble | The mean of 5 copies of a network trained with different random seeds. Their spread gives the uncertainty band |
| Latin hypercube | A random design that spreads trials evenly over each setting |
| B0, B1, B2 | Benchmarks: the mean training curve; k-nearest neighbours in X; persistence of the last rate (backtest only) |
| RMSE, NRMSE | Root mean squared error; the same ÷ the range of the true values |
| J | The selection score: the mean validation NRMSE of oil and water |
| BT | The backtest tie-break score (section 7) |
| P1, P2, P3 | Plausibility rules for the forecast: negative rates, rising oil, stalled decline (section 7) |
| O, O_obs | The history-match misfit (section 13). O_obs uses the observation error only |
| σ_obs, σ_proxy | The observation error and the proxy error in O |
| ρ | The rank correlation between the proxy's O and the simulator's O over the cases |
| Coverage | Share of true values inside a model's 95% band |
| Backtest | Retrain on data up to 2005, forecast to 2008, compare with the simulated curves |
| Freeze | Recording the final proxy choice, tagged `freeze-proxy` in git, before the test cases are scored |
| Proxy ID | `<run_id>/<variant>`, the name of one trained model |
| k0 | The first producing date index (4, Q4 1998) |
| R1 | The rate definition used: Δcumulative ÷ Δtime over the previous quarter |
| GOR | Gas-oil ratio, MSCF/STB |
| Spec | The method description that the researcher writes for each task |

## Appendix B: Runs, commits and files

For traceability only; none of this is needed to review the work.

| Item | ID |
| --- | --- |
| Data preparation | `prepare-20261012T201436717380Z` (arrays checksum 451fcdab…) |
| B0, B1 | `train-20261012T235731641838Z`, `train-20261012T235732605558Z` |
| RNN (all options; secondary = option A) | `train-20261012T235952401450Z` |
| LSTM (all options; primary = option A) | `train-20261013T000744561060Z` |
| DMD | `train-20261013T005202581280Z` |
| Validation scoring | `evaluate-20261014T103331432081Z` |
| Backtest | `backtest-20261014T103503306977Z` |
| Selection | `select-20261014T110119013402Z` |
| Freeze | `freeze-20261014T110427868031Z` |
| Test scoring | `evaluate-20261014T110434893087Z` |
| Tag `freeze-proxy` | merge commit 4e01c57 |
| `main` today | bab8d9b (39 commits; no remote) |

**Where the detail lives.** All of these are in the repository root.

- Requirements: `PRD.md`.
- Rules and commands: `CLAUDE.md`.
- Specs: `docs/specs/task0.md` to `task3.md`.
- Decision log with evidence: `docs/decisions.md`, entries D-001 to D-035.
- Article checks: `docs/references.md`.
- Setup and run times: `README.md`.

These are not in git:

- the reports, in `outputs/reports/`;
- the figures, in `outputs/figures/task3/`;
- the database, `data/processed/dwc26.sqlite`.

**Rebuilding Tasks 1 to 3 from the raw files.** Run, in order: `prepare`; `train` for B0, B1, DMD, RNN and LSTM; `evaluate --split val`; `backtest`; `select`. Do **not** rerun `freeze` or `evaluate --split test`: the database allows each only once.
