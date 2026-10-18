# Methodology explained in simple terms (notebooks 2 to 4)

This note explains the methods and ideas in notebooks 2, 3 and 4. It uses plain words and few formulas. For the numbers and the evidence, read `summary_hist_mat.md` and `summary_forecast.md`.

> **Confidential.** The data comes from the DWC26 dataset. Do not share it outside the team.

## 0. The big picture

The organiser gave us 100 runs of a reservoir simulator. A *reservoir simulator* is a slow physics program that predicts how much oil, gas and water a field produces. Each run (a *case*) uses a different set of 4 uncertain rock parameters. The organiser also gave us the field's real production from 1998 to 2008 (the *history*).

Our job has three steps:

1. Build a fast copy of the simulator (notebook 2).
2. Find the parameter sets that reproduce the history (notebook 3).
3. Forecast those sets to 2028 (notebook 4).

```
 100 simulator runs                    observed history 1998-2008
 (4 parameters -> curves)                         |
        |                                         |
        v                                         |
 [Notebook 2] PROXY  ----------------------->  [Notebook 3] HISTORY MATCHING
 learn: parameters -> curves                   which parameters fit the history?
 (fast copy of the simulator)                  -> the 10 best cases
                                                          |
                                                          v
                                               [Notebook 4] FORECAST
                                               extend the 10 cases to 2028
                                               -> 10_Final_deliverables.xlsx
```
*Each notebook reads the output of the one before it. Rerun them in the order 2 → 3 → 4.*

**The 4 uncertain parameters**

| Name | Plain meaning |
|---|---|
| `poro_mult` | Porosity multiplier: how much pore space the rock has, so how much oil it holds |
| `perm_mult` | Permeability multiplier: how easily fluid flows through the rock |
| `fault_trans` | Fault transmissibility: how easily fluid crosses the fault |
| `aquifer_pv` | Aquifer pore volume: the size of the water body under the field, which pushes water in |

---

## 1. Notebook 2: the proxy model

### 1.1 Why we need a proxy

History matching must test many thousands of parameter sets. One simulator run is slow, and we cannot run the simulator at all. So we train a machine-learning model on the 100 cases. This model is the **proxy**. It returns a full production curve in milliseconds.

```
 4 parameters  -->  [ simulator ]  -->  production curves     (slow, not available to us)
 4 parameters  -->  [   proxy   ]  -->  production curves     (fast, learned from 100 cases)
```

### 1.2 Data split

We divide the 100 cases into three groups. Each group has one job.

| Group | Cases | Job |
|---|---|---|
| Train | 1-70 | The model learns from these |
| Validation | 71-85 | We tune settings and measure the proxy error on these |
| Test | 86-100 | A final, independent check |

A model can memorise its training data. The validation and test cases show how well it does on cases it has never seen.

### 1.3 Inputs: parameters and grid features

The model uses the 4 parameters plus 8 **grid features**. A grid feature is a summary of the simulator's 3D rock model, for example the mean porosity or the total pore volume.

For a *new* parameter set, no grid file exists. A small **grid mapper** calculates the 8 features from the 4 parameters. It uses a simple linear fit, which is accurate to 0.002 standard deviations.

### 1.4 The network

The proxy is a **multi-layer perceptron (MLP)**. An MLP is a basic neural network: layers of simple units, each layer feeding the next.

- **Residual learning.** The model does not predict the whole curve. It predicts how a case *differs* from the average curve. This is an easier task.
- **Whole curve at once.** The model outputs all dates together, so the curve keeps a smooth shape.
- **Early stopping.** Training stops when the validation error stops falling. This prevents memorising.
- **Ensemble.** We train several networks from different random starts (*seeds*) and average them. The average is more stable than one network.

### 1.5 Tuning: cross-validation

The model has settings, such as the layer width and the learning rate. We choose them with **5-fold cross-validation (CV)**:

1. Split the 70 train cases into 5 parts.
2. Train on 4 parts and test on the fifth.
3. Repeat 5 times, so each part is tested once.
4. Keep the setting with the lowest mean error.

### 1.6 Two proxies: rates and cumulatives

| Proxy | Predicts | Use |
|---|---|---|
| Rate proxy | Production per day | Elton's step-1 result, for comparison |
| **Cumulative proxy** | Total production up to each date | **Used by the history matching** |

The history and the simulator report *rates* in slightly different ways. *Cumulative* production is the same in both, so we match on cumulatives.

### 1.7 The improvement (proxy version 2)

We tested 8 variants of the cumulative proxy in `improve_proxy/`. One won, and we adopted it. It keeps Elton's network and changes three things:

| Change | Reason |
|---|---|
| Predict oil and water only; gas = 0.3633 × oil | Gas follows oil exactly. Predicting it counted oil twice and gave less attention to water, the weak phase |
| Tune the learning rate and weight decay | Elton's search did not tune the learning rate |
| 20 seeds instead of 5 | A larger average is more stable |

The water error fell by about 40 % on the test cases. A paired statistical test showed that the gain is real (p ≈ 1e-9).

### 1.8 How we measure the proxy for history matching

We measure the error as a **share of the final observed cumulative** in 2008. For example, an error of 0.1 % means 0.1 % of the oil the field produced by 2008. We use these units because notebook 3 uses them.

- **sigma_proxy** is the typical proxy error on the validation cases. Notebook 3 adds it to the noise, so it does not trust the proxy more than it should.
- **Ranking check.** We score each case with the proxy curve and with the simulator curve. If both scores put the cases in the same order, the proxy can find good matches.

### 1.9 Hand-over

Notebook 2 writes `hist_match_input/`: the proxy file, its error, the simulated curves and the parameters. A hash code (sha256) in `proxy_meta.json` lets notebook 3 detect an old or wrong model file.

---

## 2. Notebook 3: history matching

### 2.1 The idea

**History matching** means finding the parameter sets whose curves agree with the observed history. Many different sets can fit the history about equally well. So we do not look for one answer. We look for the *range* of answers, and then choose 10 different ones.

### 2.2 The score

1. Use the **matching window**: 1999-01-01 to 2008-01-01 (37 quarterly dates). Oil production starts in 1999.
2. At each date, calculate the residual: (predicted − observed) ÷ final observed value.
3. Take the root-mean-square (RMS) of the residuals for oil, and the same for water.
4. Combine the oil and water values into one **score**. A lower score is a better match.

### 2.3 Two methods, used together

```
                    observed history
                  /                  \
   [A] SCORE THE 100 CASES        [B] BAYESIAN MCMC
   exact simulator curves         the proxy tests many thousands of sets
   "which known case fits?"       "which parameter values are likely?"
                  \                  /
               [C] CHOICE OF THE 10, BEST MATCH FIRST
```
*Method A is exact but limited to 100 points. Method B covers the whole range but depends on the proxy.*

**Method A** scores the 100 simulated cases directly. These curves are exact, but only 100 parameter sets exist.

**Method B** uses **Bayesian inference**. It starts from a *prior* (all parameter values in the train range are equally likely). The data then updates this to a *posterior* (the values that fit the history are more likely).

- **Likelihood.** This measures how well a parameter set fits. It assumes the residuals are random noise. The noise includes the proxy error (sigma_proxy). It also allows each residual to be similar to the one before it (*AR(1) correlation*), because the errors on a smooth curve are not independent.
- **MCMC (Markov chain Monte Carlo).** This method draws samples from the posterior. "Walkers" move through the parameter space. They spend more time where the fit is good. We use the `emcee` package, 4 independent groups of 16 walkers.
- **Convergence checks.** **R-hat** compares the independent groups. If R-hat is below 1.1, they agree. **ESS (effective sample size)** is the number of independent samples. A larger ESS gives a more reliable result.

### 2.4 The choice of the 10: best match first

```
 candidates: 100 simulated cases  +  sets drawn from the posterior
        |
 rule out: implausibility > 3   (clearly does not fit)
        |
 equal-best band: score within the proxy error of the best score
        |
 inside the band: simulated cases first, then new sets, by score
 (skip a set that is a near-copy of one already chosen)
        |
 the 10, ranked by score
```
*The organiser's FAQ asks for history-match quality first, so the score decides.*

- **Implausibility.** This is the worst residual divided by the expected error. If it is above 3, the set is clearly wrong, and we remove it.
- **Equal-best band.** The proxy has an error (eps, 0.11 % in score units). Two scores closer than eps cannot be told apart. So every candidate within eps of the best score is an equally good match.
- **Simulated cases first.** A simulated curve is exact; a proxy curve has an error. Among equal matches, we take the simulated cases first. The new sets fill the remaining places, best score first.
- **Near-copies.** Two almost identical parameter sets add nothing, so we skip the second one.
- **Before the FAQ (version 3).** We used *k-means clustering* (it sorts similar parameter sets into groups) and took one set per group. That put diversity first. FAQ 9 says that match quality must come first, so k-means is now only a comparison.
- **New sets are allowed.** The FAQ asks teams to "optimize" the parameters, and the organisers' reference paper uses posterior samples in the same way.

### 2.5 Checks

| Check | Question it answers |
|---|---|
| Direct sampling (10b) | Does a second method, without chains, give the same posterior as the MCMC? |
| Implausibility map (10c) | Which parameter values does the history rule out? |
| Global search (10d) | What is the best score the proxy can reach? |
| Oil-water trade-off (10e) | Does fitting oil better make water worse? |
| Hold-out (10f) | Does a good fit on early history predict the late history? |
| Scorecard (10g) | All the scores of each simulated case in one table |
| Recovery test (11) | If we hide a known case and use it as a fake history, does the pipeline find its parameters? |
| Physics checks (11) | Are the 10 zero before first production, like the history? Do their cumulatives never fall? |

The **recovery test** is the most important one. In it, we know the right answer, so we can check that the method finds it.

### 2.6 Main result

The history fixes the porosity well (multiplier 1.34-1.38). The other parameters stay uncertain, but they change the 2028 oil by only about 1 %. The 10 cases are 6 simulated cases (6, 97, 76, 11, 40, 73) and 4 new parameter sets. All 10 scores are within the proxy error of the best score.

---

## 3. Notebook 4: the forecast to 2028

### 3.1 The problem

The simulated cases stop at 2008. No simulator run exists after 2008, so we cannot train a proxy for the future. Instead, we extend each of the 10 cases with a **petroleum-engineering forecast method**. We fit the method to the case's late history.

### 3.2 Key terms

| Term | Meaning |
|---|---|
| Rate | Production per day in one quarter: (cumulative change) ÷ days |
| Liquid rate | Oil rate + water rate |
| **WOR** | Water-oil ratio: water rate ÷ oil rate. It rises as the field ages |
| **GOR** | Gas-oil ratio: gas ÷ oil. Here it is constant (0.3633), so gas = GOR × oil |
| Water cut | Water ÷ liquid, from 0 to 1 |

### 3.3 The methods we compared

| Method | Idea in plain words |
|---|---|
| `elton` | Fit an S-shaped curve to the whole cumulative curve (baseline) |
| `arps_wor` | **Arps decline:** the oil rate falls in a standard curved way. Water comes from the WOR trend |
| `liq_wor` | The liquid rate follows a falling trend; the WOR trend splits it into oil and water |
| **`crm_wor`** | **Capacitance-resistance model (CRM):** the field is a tank. Injected water pushes liquid out. The WOR trend splits the liquid into oil and water |

**The chosen method: CRM + WOR.**

```
 water injection I  -->  [ field = tank ]  -->  liquid rate q_L
                         tau: how fast it reacts       |
                         G: share of injection         v
                            that comes out      [ WOR trend ]
                                                 /         \
                                              oil         water
```
*Injection is held constant at its 2008 value (9,327 STB/d). The liquid rate moves smoothly towards G × I.*

- **tau** (the time constant) and **G** (the gain) are fitted on the last 12 quarters.
- **WOR trend.** The log of WOR rises in a straight line with cumulative oil. We keep the fitted slope and pass the line through the latest data ("WOR last").

### 3.4 Backtest: how we chose the method

A **backtest** tests a method on the past, where we know the answer.

```
 1998 ------------- cut-off (2005, 2006 or 2007) ------- 2008
     fit the method on this part   |  forecast this part, compare with the truth
```
*We use 3 cut-offs, all 100 cases and the real field.*

CRM + WOR had the lowest error at all 3 cut-offs (about 2.9 % on average). Elton's method failed this test.

### 3.5 The start-point problem and the hindcast

The 10 cases fit the observed *cumulatives* well. But in 2008, their *rates* are wrong: oil is about 11-15 % too high and water about 10 % too low. A forecast that starts from the case's own rates carries this gap for 20 years.

We tested four start points:

| Start | Cumulative from | Rates from |
|---|---|---|
| `main` (own) | The case | The case |
| `A1` | The field | The case |
| **`A2`** | **The field** | **The field**; the case gives only the shape (tau, G, WOR slope) |
| `fade_T` | The field | The field at first, then back to the case over T days |

We tested them with a **blind hindcast**. At an earlier date, we repeated the whole chain without seeing the future, and then compared with what happened. We set the rule before the test: switch only if the new start is clearly better.

- **Pseudo-fields.** Each of the 100 simulated cases served once as a "fake field". This gives many tests, so we can calculate a significance.
- **Result.** A2 halved the error (3.5 % against 7.5 %, p ≈ 5e-37). It was also better on the real field (3.8 % against 6.5 %). So we adopted A2 (forecast version 3).

### 3.6 Uncertainty

| Range | What it includes |
|---|---|
| P10 / P50 / P90 of the 10 cases | Only the spread between the 10 matched cases. Very narrow (about 1 %) |
| Posterior band | 200 sets from the MCMC posterior, each forecast the same way |
| **Wider range** | The 10 cases, their own-start forecasts and a data-only field forecast. **Use this one in slides and Power BI** |

The narrow range understates the real uncertainty. In the hindcast, the 10 cases' own range never contained the field's water. The wider range adds the uncertainty of the start point and the model choice.

**P10 / P50 / P90** are percentiles. Here, P10 is the low value, P50 the middle and P90 the high value.

### 3.7 The deliverable

Notebook 4 copies the organiser's template `09 Template Deliverable.xlsx`. It fills the sheets `Case 1` to `Case 10` with cumulative gas, oil and water for the 80 quarters from 2008-04-01 to 2028-01-01. Sheet `Case k` holds the case with history-matching rank k. The notebook then reads the file back and checks it.

### 3.8 Is it physically realistic?

The organiser's FAQ 5 asks that a model "remain physically realistic". We check this in three layers.

```
 Parameters  ──>  History curves  ──>  Forecast
 (plausible?)     (physics obeyed?)    (physics obeyed after 2008?)
```
*A case is realistic only if it passes all three layers.*

| Layer | Checks | Result |
|---|---|---|
| Parameters | Inside the sampled range; not ruled out by the history | Pass |
| History curves | Zero before first production; cumulatives never fall | Pass (a 50 STB dip in a proxy water curve is removed) |
| Forecast | Cumulatives never fall; oil rate never rises; water cut below 1; no jump at 2008; G close to the field's | Pass |

One gap remains. In the field, the **WOR slope** (how fast the water-oil ratio rises) is about 1.7 times the slope of the matched cases. So the forecast probably has too much oil and too little water in later years. The wider range covers this.

### 3.9 Main result

- Oil in 2028: 58.7 MMSTB (it was 38.4 MMSTB in 2008).
- Wider range: 54.2-61.7 MMSTB.

---

## 4. Limits to remember

1. **The simulator family has a gap.** No parameter set in the design reproduces the field's late water rise. A2 fixes the 2008 starting level, but not the shape of the decline: the cases' WOR slope is only about 60 % of the field's.
2. **The backtest is short.** It covers at most 12 quarters, but the forecast covers 80. It supports the method, but it does not prove 20-year accuracy.
3. **Fixed operations.** The forecast assumes constant water injection, no new wells and no change in how the field is operated after 2008.
4. **The proxy works only inside the train range.** The search stays inside it.

## 5. Glossary

| Term | Meaning |
|---|---|
| Case | One simulator run with one set of 4 parameters |
| Proxy | A fast machine-learning copy of the simulator |
| Cumulative | Total production from the start up to a date |
| Ensemble | The average of several models |
| Cross-validation | Repeated train-and-test on parts of the training data, to choose settings |
| Prior / posterior | What we believe about the parameters before / after we see the history |
| Likelihood | How well a parameter set fits the history |
| MCMC | A sampler that draws parameter sets in proportion to the posterior |
| R-hat, ESS | Checks that the MCMC has run long enough |
| Implausibility | Residual ÷ expected error. Above 3 = ruled out |
| k-means | A method that sorts points into groups of similar points |
| Backtest / hindcast | A test on past data where the answer is known |
| p-value | The chance of a result this strong if there were no real effect. Small = the effect is real |
| MMSTB | Million stock-tank barrels (oil or water volume) |
