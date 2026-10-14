# DWC26 Code Repository PRD

As of 13 October 2026

## 1. Overview

The repository trains a proxy of the reservoir simulator on cases 1 to 70, finds ten history-matched parameter sets with MCMC, and forecasts their rates to 1 January 2028. The code is due on 16 October 2026.

| Item | Definition |
| --- | --- |
| Product | Predictive proxy model with a time component, followed by history matching. The same model continues the rates to 2028 |
| Input X | 4 uncertainty parameters for each case (file 01): fault transmissibility, porosity multiplier, permeability multiplier, aquifer pore volume |
| Output Y | Oil, gas and water rates for each case (file 02) |
| Observed target | `RAMP_History` in file 02 |
| Time grid | The 41 quarterly dates of the observed history |
| Match window | 1 January 1999 to 1 January 2008: 37 dates |
| Forecast grid | 80 quarterly dates: 1 April 2008 to 1 January 2028 |
| Split | Train cases 1 to 70, validation 71 to 85, test 86 to 100 |

**In scope:** Tasks 0 to 5, as code in one repository.

**Out of scope:** PowerPoint, Power BI and video. Grid files 03 to 08. The conversion of rates to production volumes and the write to file 09 (a later task).

## 2. Validity Rules

All code obeys rules V1 to V13. The researcher agent checks them before each merge.

| ID | Rule |
| --- | --- |
| V1 | Fit each learned transform (Y scalers, DMD basis, model weights) on cases 1 to 70 only. Select hyperparameters on cases 71 to 85 only |
| V2 | Split by case. Never split by row or by date |
| V3 | Use cases 86 to 100 for metrics one time only, after the model choice is frozen and tagged |
| V4 | Apply the same alignment, rate definition and scaling to the cases and to `RAMP_History` |
| V5 | Do not hard-code data values. Bounds, dates and counts come from the files at run time |
| V6 | Each model must beat three benchmarks: B0 (mean training curve) and B1 (k-nearest neighbours in X) on validation, and B2 (persistence) in the backtest |
| V7 | Fix all random seeds. Keep all settings in YAML configs. Pin package versions |
| V8 | Log each run: run ID, git commit, config, seed, duration, outputs |
| V9 | Assert shapes and values at each module boundary: no NaN, rates ≥ 0, expected array shapes |
| V10 | Write a unit test for each module. `pytest` must pass before a merge |
| V11 | Test the history-matching code with a parameter-recovery test (Task 4) |
| V12 | Do not publish the data or the repository. Keep raw files out of version control |
| V13 | Each proxy keeps the time component. Step 5 continues the Step 2 model. No other forecast method is used |

## 3. Repository Structure and Interfaces

One Python package, `dwc26`, runs all tasks from one command-line entry point. Each task reads from and writes to one SQLite database.

```text
dwc26/
├── CLAUDE.md                 # rules V1-V13 and commands, read by both agents
├── README.md
├── pyproject.toml            # pinned versions
├── .claude/agents/           # researcher.md, data-scientist.md (Task 0)
├── configs/
│   ├── data.yaml
│   ├── models/               # one YAML for each model
│   ├── hm.yaml
│   └── forecast.yaml
├── data/
│   ├── raw/                  # files 01 and 02, not in version control
│   └── processed/            # dwc26.sqlite, arrays.npz
├── docs/
│   ├── specs/                # task0.md to task5.md, written by the researcher
│   ├── decisions.md
│   └── references.md
├── src/dwc26/
│   ├── data/                 # load, check, align, scale, db, split
│   ├── models/               # base, registry, baselines, rnn, lstm, dmd
│   ├── evaluation/           # metrics, report, plots
│   ├── hm/                   # objective, mcmc, select, verify
│   └── forecast/             # rollout, dmd_continue, backtest
├── tests/
└── outputs/                  # models, reports, figures, hm, forecast
```

### Commands

```bash
python -m dwc26 prepare  --config configs/data.yaml
python -m dwc26 train    --model lstm --config configs/models/lstm.yaml
python -m dwc26 evaluate --split val
python -m dwc26 freeze   --model <run_id>
python -m dwc26 evaluate --split test
python -m dwc26 match    --proxy <run_id> --config configs/hm.yaml
python -m dwc26 forecast --matched <run_id> --config configs/forecast.yaml
python -m dwc26 all      # all steps from the raw files
```

### Array contracts

| Array | Shape | Content |
| --- | --- | --- |
| `X` | (n, 4) | Parameters scaled to [0, 1] |
| `Y` | (n, 41, 3) | Rates in the order [oil, gas, water], scaled |
| `hist` | (41, 3) | `RAMP_History` rates, same scaling |
| `dates` | (41,) | History dates, `datetime64[D]` |

### Proxy interface

```python
class Proxy(Protocol):
    name: str
    extrapolates_time: bool                                   # True for all models (rule V13)

    def fit(self, X: np.ndarray, Y: np.ndarray, cfg: dict) -> "Proxy": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...             # (n, 41, 3), scaled
    def predict_std(self, X: np.ndarray) -> np.ndarray | None: ...  # (n, 41, 3) or None
    def forecast(self, X: np.ndarray, n_steps: int) -> np.ndarray: ...  # Step 5 calls it
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Proxy": ...
```

- A registry maps each model name to its class.
- The pipeline trains each registered model with the same data, the same validation cases and the same tuning budget.

## 4. Task 0: Agents

Two agents divide the work: the researcher writes and reviews the method specifications, and the data scientist writes and runs the code.

| Item | Researcher | Data scientist |
| --- | --- | --- |
| Purpose | Make each method correct | Make each method run, tested and reproducible |
| Writes | `docs/specs/taskN.md`, `docs/decisions.md`, `docs/references.md`, review findings | `src/`, `tests/`, `configs/`, run logs |
| Does not | Write code in `src/` | Change a spec without researcher approval |
| Tools | Read, Grep, Glob, WebSearch, WebFetch | Read, Write, Edit, Bash, Grep, Glob |
| Checks | Spec compliance, rules V1 to V13, plausibility of results | `pytest`, data checks, run from a clean clone |

### Handoff

```text
Researcher: writes the spec
        |
        v
Data scientist: codes, tests, logs the run  <-------------------+
        |                                                       |
        v                                                       |
Review passes? --- yes ---> Merge, log the decision             |
        |                                                       |
        no                                                      |
        v                                                       |
Researcher: returns findings ------------- resubmit ------------+
```

A review with a blocker returns the work to the data scientist. Only a passed review merges.

### Specification content

Each `docs/specs/taskN.md` contains:

1. Inputs and outputs, with shapes and units.
2. Equations, each with its source (an article or an official reference) or the label "assumption".
3. Default settings and their tuning ranges.
4. Acceptance tests that the code must pass.

### Definition of done for a merge

- [ ] The code agrees with the spec.
- [ ] `pytest` passes.
- [ ] Rules V1 to V13 are checked.
- [ ] One command reproduces the result from a clean clone.
- [ ] The researcher approves with zero blockers.

### Agent files

```markdown
<!-- .claude/agents/researcher.md -->
---
name: researcher
description: Writes method specs for DWC26 tasks and reviews code against them. Use before coding a task and before each merge.
tools: Read, Grep, Glob, WebSearch, WebFetch
---
You are a reservoir-engineering and statistics researcher.
- For each task, write docs/specs/taskN.md: inputs, outputs, equations with sources, defaults, acceptance tests.
- Label each statement "source: <article>" or "assumption".
- Review each change against its spec and rules V1-V13 in CLAUDE.md.
- Return numbered findings: blocker, required or note. Approve only with zero blockers.
- Do not write code in src/.
```

```markdown
<!-- .claude/agents/data-scientist.md -->
---
name: data-scientist
description: Implements and runs DWC26 code from approved specs. Use for all code, tests and experiments.
tools: Read, Write, Edit, Bash, Grep, Glob
---
You are a data scientist.
- Implement only approved specs in docs/specs/.
- Write the tests for each interface first. Keep pytest passing.
- Obey rules V1-V13 in CLAUDE.md. Log each run.
- Ask the researcher for a spec change. Do not change a spec yourself.
- Report each result with its run ID, metrics and plots.
```

## 5. Task 1: Data Preparation

Task 1 writes one SQLite database and three arrays on the 41 history dates: `X` (100, 4), `Y` (100, 41, 3) and `hist` (41, 3).

### Inputs

| File | Sheet | Use |
| --- | --- | --- |
| 01 Introduction | Uncertainty Values | X: 4 parameters for cases 1 to 100 |
| 02 Field Production | Oil, Gas and Water production rate | Rates of each case and of `RAMP_History` |
| 02 Field Production | Oil, Gas and Water production cumulative | Cumulative volumes, used for the alignment |
| 02 Field Production | Field Production | Observed history, used for check C4 |

- Layout of the six rate and cumulative sheets: column A is the date. Columns `Case_1` to `Case_100` are the cases. The last column is `RAMP_History`. Rows 1 to 3 are headers: row 2 has the names, row 3 has the units.
- Each case reports on its own simulator dates. An empty cell means "no report on this date". It is not a missing value. Do not impute it.

### Steps

1. Load X. Map the columns to `fault_trans`, `poro_mult`, `perm_mult` and `aquifer_pv`. Assign the split from the case number.
2. Load each sheet. For each case, remove its empty cells. Store the native series in `production_native`.
3. Run checks C1 to C7. A failed check C1 to C4 stops the run.
4. Align each case and `RAMP_History` to the 41 history dates:
   1. Convert the dates to days from the first history date, as floats.
   2. Interpolate the cumulative volume linearly in time (`np.interp`). Do not extrapolate outside the dates of the case.
   3. Calculate the rates with the definition that C6 selects (R1 or R2).
5. Scale X and Y.
6. Write the database, `data/processed/arrays.npz` and `outputs/reports/data_checks.md`.

### Data checks

| ID | Check | Action |
| --- | --- | --- |
| C1 | Each of the 6 sheets has `Case_1` to `Case_100` and `RAMP_History` | Fail: stop |
| C2 | Each case covers the first to the last history date | Fail: stop |
| C3 | Cumulative volumes do not decrease. Rates are ≥ 0 | Fail: stop |
| C4 | `Field Production` agrees with `RAMP_History` (relative tolerance 1e-6) | Fail: stop |
| C5 | Each case reports on all 41 history dates | Pass: use the values on the dates. Fail: interpolate |
| C6 | History rate = Δcumulative ÷ Δdays over the previous quarter (maximum relative difference < 1%) | Pass: R1. Fail: R2 |
| C7 | Gas-oil ratio (gas rate ÷ oil rate) for each case and date | Report only: minimum, maximum, trend |

- **R1:** rate at t_k = (cum(t_k) − cum(t_k−1)) ÷ (t_k − t_k−1), the mean over the previous quarter. At t_0, use the reported rate.
- **R2:** rate at t_k = the last reported rate at or before t_k.

### Scaling

| Quantity | Method | Fitted on |
| --- | --- | --- |
| X | Min-max to [0, 1] | Bounds of file 01: the prior range, a design input, not a learned statistic |
| Oil, gas and water rates | Divide by the maximum of that fluid | Cases 1 to 70, all 41 dates |
| `RAMP_History` | The same factors as the cases | Not fitted |

- One factor for each fluid keeps zero at zero and keeps the shape of the curve.
- Store all factors in the table `scalers`. All metrics and the mismatch use physical units after the inverse transform.

### Database

Yes, a database is possible and useful. Use one SQLite file: `data/processed/dwc26.sqlite`.

- **Reasons:** one file, no server, Python standard library (`sqlite3`), SQL queries across tasks, one source for all reports.
- **Alternative:** DuckDB, if the team prefers Parquet files. The data are small, thus speed does not decide.

| Table | Key | Main columns | Written by |
| --- | --- | --- | --- |
| `parameters` | case_id | split, fault_trans, poro_mult, perm_mult, aquifer_pv | Task 1 |
| `production_native` | case_id, quantity, date | value | Task 1 |
| `production_grid` | case_id, date_idx | date, oil/gas/water rate, oil/gas/water cumulative | Task 1 |
| `history` | date_idx | date, oil/gas/water rate, oil/gas/water cumulative, water injection rate | Task 1 |
| `data_checks` | run_id, check_id | status, detail | Task 1 |
| `scalers` | name | fluid, factor, bounds | Task 1 |
| `runs` | run_id | task, model, config, seed, git_commit, start, duration | All tasks |
| `predictions` | run_id, case_id, date_idx | oil, gas, water | Tasks 2 and 3 |
| `metrics` | run_id, split, fluid, metric | value | Task 3 |
| `mcmc_samples` | run_id, chain, step, walker | 4 parameters, log_prob, O | Task 4 |
| `matched_sets` | run_id, rank | 4 parameters, O, O_obs, nearest distance | Task 4 |
| `forecast_rates` | run_id, set_rank, method, date | oil, gas, water | Task 5 |

### Tests

- Interpolation is exact on a synthetic piecewise-linear cumulative.
- Scaling followed by the inverse transform returns the original values.
- The split counts are 70, 15 and 15.
- Each check C1 to C6 fails on a corrupted copy of the input.

## 6. Task 2: Proxy Model Training

One pipeline trains 3 models that keep the time component (RNN, LSTM, DMD) and 2 benchmarks on cases 1 to 70. Step 5 continues the same trained models to 2028. No other forecast method is used.

### The problem: constant X, temporal Y

- X has 4 values for each case. They do not change in time.
- Y has 41 dates × 3 fluids for each case. The dates are ordered and correlated.
- Each model must keep the time component: it gives the rates at any date, inside and after the 41 history dates.

| ID | Strategy | How | Models |
| --- | --- | --- | --- |
| S3 | Conditioned sequence | X enters each step, or X sets the initial hidden state. The network gives 3 rates at each step and continues the steps after 2008 | RNN, LSTM |
| S4 | Shared dynamics | DMD gives a time basis λ_k^t from the training curves. A regressor maps X to the coefficients of each curve. The basis continues to any date | DMD |

Removed: kriging, PCE, SVR, DNN and autoencoder. They map X to the 41 history dates only and cannot continue in time.

### Model specifications

| Model | Role | Library | Settings (tuned) |
| --- | --- | --- | --- |
| RNN | Proxy (S3) | PyTorch `nn.RNN` or `nn.GRU` | 1 to 2 layers, 16 to 64 units, 5 seeds |
| LSTM | Proxy (S3) | PyTorch `nn.LSTM` | 1 to 2 layers, 16 to 64 units, 5 seeds |
| DMD | Proxy (S4) | NumPy SVD, or `pydmd` | Rank r for 99% of the energy, GP from X to the coefficients |
| B0 | Benchmark only | NumPy | Mean training curve |
| B1 | Benchmark only | scikit-learn | k-nearest neighbours in scaled X, k = 3, distance weights |

- B0 and B1 are not proxies. They measure the 1998 to 2008 accuracy (rule V6). They do not go to Steps 4 and 5.

### Model details

- **RNN and LSTM input (config option):**
  - Option A: [X, t] at each step (input size 5).
  - Option B: a dense layer maps X to h₀ and c₀. The input is t only.
  - Option C: X at each step, no t. The step count carries the time.
  - Output: a linear layer with 3 units. Source: article 2.
- **Time scaling:** t = step index ÷ 40. The 41 history steps give t = 0 to 1. The 80 forecast steps give t = 1.025 to 3. Options A and B thus receive t values outside the training range in Step 5. Option C does not. The backtest (Task 5) reports each option.
- **DMD:**
  1. Build the snapshot matrix D: rows = training case × fluid (210), columns = 41 dates.
  2. Fit exact DMD with rank r. Keep the eigenvalues λ_k.
  3. For each training curve, solve the coefficients c by least squares on the basis V(t, k) = λ_k^t.
  4. For each fluid, fit a GP from X to the real and imaginary parts of c. The GP maps X to the coefficients only. Time stays in λ_k^t.
  5. Predict y(x, t) = Re Σ_k c_k(x) λ_k^t. The same formula gives the rates after the 41 dates.
  6. Risk: a plateau held by a rate limit is not linear dynamics. Report the fit error in the plateau period.

### Training rules

- **Tuning:** train on cases 1 to 70. Select the settings by the error on cases 71 to 85. Equal budget: 30 trials for each model. Neural networks use early stopping on cases 71 to 85.
- **Final fit:** the model trained on cases 1 to 70 with the best settings. There is no refit.
- **Model choice:** cases 71 to 85 and the backtest (Task 3). Cases 86 to 100 are used for neither tuning nor choice.
- **Loss (neural models):** mean squared error on the scaled rates, with equal weight for oil, gas and water.
- **Seeds:** 5 seeds for each neural model. The prediction is the mean. The spread is `predict_std`.
- **Negative rates:** clip at 0 after the inverse transform. Log the number of clipped values.
- **Speed:** `predict` evaluates 160,000 parameter sets (32 walkers × 5,000 steps) in less than 10 minutes on a laptop CPU.
- **Continuation:** each model implements `forecast(X, n_steps)` from the same fitted state. Task 5 calls it with n_steps = 80. There is no refit.
- **Save:** model, scalers and config in `outputs/models/<run_id>/`.

### Tests

- Contract test for each registered model: fit on small synthetic data. `predict` returns (n, 41, 3) with no NaN. The same seed gives the same result.
- `forecast(X, 80)` returns (n, 80, 3). The first 41 steps of a 121-step run are equal to `predict`.
- DMD reproduces a synthetic sum of exponentials, inside and after the training dates.

## 7. Task 3: Validation and Testing

The model choice uses the validation error of oil and water, the backtest error and the rank correlation of the mismatch. The test metrics are calculated one time, after the freeze.

### Metrics

$$
\mathrm{RMSE}_f = \sqrt{\frac{1}{nT}\sum_{i=1}^{n}\sum_{k=1}^{T}\left(\hat{y}_{f,i,k} - y_{f,i,k}\right)^2}
$$

$$
R^2_{\mathrm{case},f} = 1 - \frac{\sum_{i,k}\left(\hat{y}_{f,i,k} - y_{f,i,k}\right)^2}{\sum_{i,k}\left(y_{f,i,k} - \bar{y}^{\,\mathrm{train}}_{f,k}\right)^2}
$$

| Metric | Definition | Use | Source |
| --- | --- | --- | --- |
| RMSE | Formula above, in STB/d or MSCF/d, for each fluid | Main error | Standard |
| NRMSE | RMSE ÷ (max − min) of the true values in the split | Comparison of fluids, model choice | Standard |
| R² pooled | 1 − SSE ÷ Σ(y − ȳ)², ȳ = mean of all true values | Report | Article 4, Eq. 16 |
| R² case | Formula above: the denominator uses the training mean curve at each date | Shows if the model separates the cases | Assumption: pooled R² includes the time trend and overstates accuracy |
| Event errors | Error in days of plateau end, water breakthrough and water-oil crossover | Curve shape in time | Kick-off workflow |
| Backtest RMSE | RMSE at horizons 1 to 12 quarters after the cutoff (Task 5) | Forecast skill of the same model | Validity |
| Mismatch rank ρ | Spearman correlation of O(proxy) and O(simulated) over the cases of the split | Accuracy where the search goes | Article 6 |
| Local RMSE | RMSE on the 5 cases of the split with the lowest O(simulated) | Accuracy near the history | Article 6 |
| Interval coverage | Share of true values in the 95% interval. Target 0.90 to 0.98 | Seed ensembles (RNN, LSTM) and the DMD GP | Standard |
| Cost | Fit time. Time to predict 10,000 sets | Feasibility of MCMC | Article 2 |

Event definitions (defaults in the config):

- **Plateau end:** first date with oil rate < 0.98 × maximum oil rate.
- **Water breakthrough:** first date with water rate > 0.01 × maximum water rate.
- **Water-oil crossover:** first date with water rate > oil rate.

### Selection rule

1. Reject each model that does not beat B0 and B1 in the validation RMSE of oil and of water.
2. Reject each model that does not beat B2 in the backtest at horizons 9 to 12.
3. Rank the other models by the mean validation NRMSE of oil and water.
4. If two models are within 5%, prefer the lower backtest RMSE, then the higher mismatch rank ρ.
5. Select a primary proxy, and a secondary proxy with a different strategy (S3 or S4). Tasks 4 and 5 use both.
6. Freeze: write the choice in `docs/decisions.md`. Tag the commit `freeze-proxy`.
7. Calculate all metrics on cases 86 to 100 one time, for all models. Report them. Do not change the choice after this step.

### Outputs

- `outputs/reports/model_comparison.md`: one table with all models and all splits.
- For each validation and test case: predicted and simulated curves of oil, gas and water.
- Parity plots for each fluid.
- All metrics in the table `metrics`.

## 8. Task 4: History Matching

MCMC with the frozen primary proxy samples the posterior of the 4 parameters. Ten samples with low mismatch, far from each other, become the matched sets.

### Mismatch O(x)

$$
O(x) = \sum_{f \in \{\mathrm{oil},\,\mathrm{water}\}} \; \sum_{k \in K} \left( \frac{\hat{y}_f(x, t_k) - d_f(t_k)}{\sigma_f(t_k)} \right)^2
$$

$$
\sigma_f(t_k)^2 = \sigma_{\mathrm{obs},f}(t_k)^2 + \sigma_{\mathrm{proxy},f}(t_k)^2, \qquad \sigma_{\mathrm{obs},f}(t_k) = \max\left(\alpha \, d_f(t_k),\; \beta \, \max_{k} d_f(t_k)\right)
$$

| Symbol | Meaning | Default | Source |
| --- | --- | --- | --- |
| ŷ_f(x, t_k) | Proxy rate in physical units | From Task 2 | Spec |
| d_f(t_k) | `RAMP_History` rate, same rate definition | From Task 1 | Spec |
| K | Dates from 1 January 1999 to 1 January 2008 | 37 dates | Spec |
| α | Relative observation error | 0.10 | Article 4, Eq. 18 |
| β | Floor for rates near zero, for example water before breakthrough | 0.02 | Assumption |
| σ_proxy,f(t_k) | Validation RMSE of the primary proxy at date t_k | From Task 3 | Assumption, after article 6 |

- **Balance of oil and water:** each fluid has 37 terms, each divided by its own σ. Thus the two fluids contribute equally.
- **Sum, not mean:** O is a sum of 74 terms. A mean divides O by 74, which makes exp(−O/2) almost flat. Then the MCMC returns the prior.
- **Gas:** not in O (spec). Task 3 checks gas, and the verification plots it.
- **O_obs:** the same formula with σ_proxy = 0. Use it to compare matched sets with simulated cases.

### Posterior

$$
\log p(x \mid d) = -\tfrac{1}{2}\, O(x) + \log p(x), \qquad \log p(x) = \begin{cases} 0 & x \text{ inside the file 01 bounds} \\ -\infty & \text{otherwise} \end{cases}
$$

- Sample in scaled space, u in [0, 1]⁴. A uniform prior stays uniform after min-max scaling.

### MCMC settings

| Setting | Value | Source |
| --- | --- | --- |
| Sampler | `emcee.EnsembleSampler`, affine-invariant stretch move | Article 2 (AIES) |
| Walkers | 32 | Spec |
| Steps | 1,000 to 5,000. Stop when the steps are more than 50 τ_max | emcee documentation |
| Start | 32 Latin-hypercube points in the prior, fixed seed | Assumption |
| Burn-in | Discard the first ⌈2 τ_max⌉ steps | emcee documentation |
| Thin | Keep every ⌈0.5 τ_min⌉ step | emcee documentation |
| Batch | `vectorize=True`: one proxy call for all 32 walkers | Speed |
| Repeats | 3 runs with different seeds | Validity |

τ is the integrated autocorrelation time of each parameter (`sampler.get_autocorr_time`).

Pass criteria:

- Mean acceptance fraction is 0.2 to 0.5.
- Chain length is more than 50 τ_max. If not at 5,000 steps, mark the run "not converged" in the report.
- The 3 runs agree: the 5%, 50% and 95% quantiles of each parameter differ by less than 0.05 in scaled units.
- If the runs disagree, the posterior can have several modes. Report it, and pool the runs for the selection.

### Selection of the ten sets

1. Pool the thinned samples after burn-in from the 3 runs. Remove duplicate positions.
2. Filters (articles 4 and 6, on by default):
   1. Reject samples within 0.02 of a bound, in scaled units.
   2. Reject samples whose proxy standard deviation is above the 95th percentile of its value at cases 71 to 85.
3. Sort the samples by O, lowest first.
4. Take the first sample. Accept each next sample only if its distance to all accepted samples is at least d_min. Distance: Euclidean, in scaled units. Default d_min = 0.10.
5. Stop at 10 samples. If there are fewer than 10, multiply d_min by 0.8 and repeat. Record the final d_min.
6. Acceptance limit: each selected sample has O ≤ O_max. Default O_max = the 95% quantile of χ² with 74 degrees of freedom (approximately 95.1).

### Verification

1. Calculate O_obs for each of the 100 simulated cases with its simulated curves. The ten lowest give the ten nearest simulated cases.
2. **Parameters:** compare the matched sets with the ten nearest cases: median and range of each parameter, and a scatter-matrix plot. Report each matched set outside the range of the nearest cases.
3. **Mismatch:** compare O_obs of the matched sets with O_obs of the ten nearest cases.
4. **Proxy near the history:** for the ten nearest cases, compare the proxy curves with the simulated curves: RMSE, and O_proxy − O_obs.
5. **Second proxy:** calculate O of the ten matched sets with the secondary proxy. Report each set with a large difference.
6. **Curves:** plot `RAMP_History`, the ten nearest simulated cases and the ten matched proxy curves, for oil, gas and water.

### Parameter-recovery test (rule V11)

1. Use 3 test cases (86 to 100) as pseudo-histories. Select them with a fixed seed. Run this test after the freeze. It does not change the proxy.
2. Run the full Task 4 procedure on each pseudo-history.
3. For each parameter, report whether the true value is in the 90% posterior interval.
4. A parameter that the data do not constrain gives a wide interval. This is a result, not an error.

### Outputs

- Tables `mcmc_samples` and `matched_sets`.
- `outputs/hm/<run_id>/`: trace plots, corner plot, curve plots, `verification.md`.

### Tests

- O = 0 when the prediction equals the history.
- O agrees with a hand calculation on a 3-date example.
- log p(x) = −∞ outside the bounds.
- The selection returns 10 sets, and each pair is at least d_min apart.
- The same seed gives the same chain.

## 9. Task 5: Rate Forecast

Task 5 gives quarterly oil, gas and water rates from 1 April 2008 to 1 January 2028 (80 dates) for each of the ten matched sets. The conversion to production volumes is a later task.

- No training data exist after the last history date. Each forecast is an extrapolation of the Step 2 model. The backtest is the only test of it.

### Method

Step 5 continues the frozen Step 2 proxies. No other forecast method is used.

| ID | Applies to | How |
| --- | --- | --- |
| F1 | RNN, LSTM | Run the network for 121 steps (41 history + 80 forecast) with the same X. Keep steps 42 to 121 |
| F2 | DMD | y(x, t) = Re Σ c_k(x) λ_k^t at the 80 new dates |

- The primary and the secondary proxy from Task 3 both give a forecast for each matched set. The primary forecast is the deliverable. The secondary forecast measures method uncertainty.

### Backtest

1. Cutoff: 1 January 2005 (configurable). This gives 29 dates before the cutoff and 12 dates after it.
2. Train each model (RNN, LSTM, DMD, and each RNN and LSTM input option) on cases 1 to 70 with the dates up to the cutoff only, with the Task 2 settings.
3. Continue each model to 1 January 2008 with its own method (F1 or F2).
4. Compare with the simulated curves: RMSE for each fluid and each horizon (1 to 12 quarters).
5. Benchmark B2 (persistence): the last rate before the cutoff stays constant. Each model must beat B2 at horizons 9 to 12.
6. Run the backtest before the Task 3 freeze. Use cases 71 to 85 for the model choice. Report cases 86 to 100 one time, after the freeze.

The backtest covers 12 quarters, and the forecast covers 80. Thus the backtest is evidence for the method, not proof of the 20-year accuracy.

### Constraints and checks

- Rates ≥ 0. Clip negative values and log them.
- Report each set where the oil rate increases after the last history date.
- Report the spread of the ten sets for each fluid and date.
- Report the difference between the primary and the secondary proxy forecasts, as method uncertainty.

### Outputs

- Table `forecast_rates`: run_id, set_rank, method, date, oil, gas and water rate.
- `outputs/forecast/<run_id>/rates.csv`, and plots of the history with the ten forecasts for each fluid.

### Later task (not in this version)

- Convert the rates to cumulative volumes: volume in a quarter = rate × days. Start from the cumulative on 1 January 2008. Write file 09.
- Add the movable-oil limit from grid files 03 to 08.

## 10. Schedule and Acceptance

All work ends on 16 October, the code date in the kick-off slides.

| Date | Work |
| --- | --- |
| 13 Oct | Task 0, 1: agents, data |
| 13 to 14 Oct | Task 2: proxy models |
| 14 Oct | Task 3: select, freeze |
| 15 Oct | Task 4: history match |
| 15 to 16 Oct | Task 5: forecast, zip |
| 16 Oct | Code repository due |

Tasks 2 and 5 each take two days. A delay in Task 2 moves all later tasks.

### Acceptance criteria

- [ ] `python -m dwc26 all` reproduces the database, models, metrics, matched sets and forecast rates from a clean clone and the raw files.
- [ ] `pytest` passes.
- [ ] Checks C1 to C4 pass, and C5 to C7 are reported.
- [ ] Each model beats B0 and B1 on validation and B2 in the backtest, or the report marks it as rejected.
- [ ] Test metrics are calculated one time, after the tag `freeze-proxy`.
- [ ] MCMC pass criteria are met for 3 seeds, or the report marks the run "not converged".
- [ ] The parameter-recovery test is reported.
- [ ] Ten matched sets: each has O ≤ O_max, and each pair is at least d_min apart.
- [ ] The forecast backtest is reported.
- [ ] Forecast rates: 80 dates × 3 fluids × 10 sets.
- [ ] `README.md` gives the setup, commands, run time and outputs.

### Open decisions

- **Refit before Task 4:** a refit of the frozen proxy on cases 1 to 100 gives more data for the match, but no independent check remains. Default: no refit.
- **Code deadline:** 16 October
