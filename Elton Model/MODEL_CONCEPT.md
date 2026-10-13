# Model Concept — DWC26 Proxy Model (video script notes)

This document explains, in plain language, what our model is, how it works, and why we chose this method.
It is written to be read aloud or paraphrased for a short video.

---

## 1. The big picture

The challenge asks three things: reproduce 10 years of observed field production (history matching),
then forecast the next 20 years, and submit the 10 best-performing cases.
The problem: the full physics simulator is slow, and history matching means testing **thousands** of
parameter combinations. Running the real simulator for every candidate is not practical.

**Our solution: a "proxy" model.**
A proxy (also called a surrogate) is a machine-learning stand-in for the simulator.
It takes the same inputs and produces the same outputs, but answers in milliseconds instead of hours.
Think of it as a calculator trained from solved examples: it never re-solves the physics, it has *learned* the physics.

---

## 2. What the model does

**Input (12 numbers per case):**
- 4 uncertainty parameters: fault transmissibility, porosity multiplier, permeability multiplier, aquifer pore volume.
- 8 grid summary features (see section 4) — a "fingerprint" of how those knobs changed the geological model.

**Output (123 numbers per case):**
- Oil, gas, and water production rates at 41 quarterly dates (1998 → 2008) — the **whole curve at once**,
  not one point at a time. This makes the model fast for screening and preserves the curve's shape.

Why quarterly? Each simulator case has its own timestamps, so we aligned all cases to the 41 quarterly
dates shared by every case and by the observed history.

---

## 3. How the model works (step by step, no jargon)

1. **Learn the shape first, then the details.** We compute the average production curve of the training
   cases, and the network only learns the *difference* between each case and that average (residual learning).
   Small task, easier training.
2. **A small neural network.** Two hidden layers of 128 units (~34,000 weights — tiny by deep-learning
   standards). With only 70 training cases, small is a feature, not a limitation: big networks would memorize.
3. **Careful data protocol.** Train on cases 1–70 only. Hyperparameters chosen by 5-fold cross-validation
   *inside* those 70. Cases 71–85 kept for validation, cases 86–100 opened only once, at the end, as a final test.
4. **Five models, one answer (ensemble).** We train five copies from different random seeds and average their
   predictions. This removes the luck of a single random start and makes results stable.
5. **Early stopping + scaled inputs/outputs.** Training stops when validation error stops improving, and all
   numbers are normalized so the network sees comparable ranges.

Result: given four uncertainty values (and the case's grid fingerprint), the proxy predicts the full
10-year production history of that case in milliseconds.

---

## 4. The clever part: the 8 grid features

The four uncertainty knobs don't act in the abstract — they physically modify the reservoir grid:
- the porosity multiplier scales porosity in **every** active cell,
- the permeability multiplier scales permeability in **one zone**,
- the aquifer volume multiplier scales pore volume only at the **aquifer edge**.

Instead of feeding the network 42,512 grid cells per case, we summarize each case with 8 numbers:
total pore volume, aquifer and reservoir pore volumes, mean porosity, mean permeability (global and in-zone),
and mean transmissibility in both directions.

Why this works: the grid of every case is a scaled copy of the base case (verified in the data analysis),
so a handful of summaries captures the whole geological difference. These features make the "global scale
effects" easy for the network to read — and in tuning they clearly helped: the grid-feature version scored a
5-fold CV RMSE of **36.3** versus **~55** for the four parameters alone.

---

## 5. Why this method (four reasons)

1. **Speed.** Milliseconds per prediction makes tens of thousands of history-matching candidates feasible.
2. **Accuracy.** On the untouched test cases (86–100), test RMSE (rate units) was:

   | Phase | Our proxy | Ridge baseline | Mean-curve baseline |
   |-------|-----------|----------------|---------------------|
   | Oil   | 41.2      | 190.8          | 1,015.9             |
   | Gas   | 13.9      | 68.4           | 369.0               |
   | Water | 35.7      | 119.5          | 762.1               |
   | Pooled| **32.5**  | 135.8          | 763.5               |

   About **4x better than the classic linear baseline** and far better than "just predict the average".
   Beating strong, simple baselines is what proves the deep-learning step actually adds value.
3. **Data-frugal design.** With only ~85 labeled cases available, residual learning + a small network +
   regularization + ensembling is the right toolbox — not a giant model.
4. **Honest evaluation.** Validation and test cases were never used for fitting, and the hardest cases are
   the ones slightly outside the training parameter range (cases 82, 87, 98) — we report that openly.

---

## 6. Where the proxy fits in the full workflow

| Step | Task | Status |
|------|------|--------|
| 1 | Build the proxy (simulator stand-in) | **Done** — trained, validated, tested |
| 2 | History matching: find the parameter sets whose simulated history best matches the **observed** 1998–2008 field production | **Done** — 45,000 sets scored in ~2 s; best misfit 0.00385, better than the best of the 100 provided cases (0.0040) |
| 3 | Keep the best-matching sets → these are the "history-matched models" | **Done** — top-20 saved as `hm_matched_models.csv`; ~4,900 sets within 25% of the best misfit |
| 4 | Forecast each matched case 20 years ahead (80 quarters, 2008 → 2028), rank by performance, choose the best and most **diverse** 10 cases, and fill the deliverable | **Done** — `10_Final_deliverables.xlsx` written and readback-validated (10 sheets × 80 quarters) |

Key insight from the exploratory analysis (useful for the video):
the observed history pins down mostly the **porosity** multiplier; the other three parameters are only
loosely identifiable, so *many* parameter sets match the history almost equally well.
This is exactly why the proxy is valuable (it can scan that whole space), and why the final 10 cases
should be **diverse** matches, not 10 near-identical ones — they represent the remaining uncertainty
about the field rather than one "single truth".

---

## 7. Steps 2–4 in plain language (history matching + 20-year forecast)

**Why we match cumulative, not rates.** The observed history and the simulated cases report rates in
slightly different ways; cumulative production (total produced so far) is comparable everywhere and
is what the deliverable asks for. So we match, forecast and submit cumulative curves.

**History matching (step 2).** A parameter set is scored by how far its predicted cumulative curves
sit from the observed history: the error at every quarter, expressed as a fraction of the final
observed value, combined into one number (root-mean-square over the three phases). Using the
cumulative proxy (milliseconds per case), we test **45,000 parameter sets** — a space-filling Sobol
design plus a local polish around the best 20 — in **about 2 seconds**. The best set reaches a
misfit of **0.00385**: slightly better than the best of all 100 provided cases (case 6, 0.0040) and
~14× closer to the history than a typical case (0.054).

**The history does not pin one answer.** ~4,900 parameter sets sit within 25% of the best misfit —
mostly agreeing that the porosity multiplier is ≈ 1.37. The "history-matched models" (step 3: the
top 20) are therefore not one truth, but a set of equally plausible explanations of the field.

**Forecast (step 4).** The simulated cases stop at 2008-01-01, so no forecast data exists to learn
from. Each matched curve is extended with a physical decline model instead: three candidate families
(Arps hyperbolic, Gompertz, rational) are fitted on the first 25 quarters of every one of the 100
known cases and scored on the remaining 16. Winners: **Arps for oil and gas** (median relative error
5%), **Gompertz for water** (water breakthrough is the hardest part). The winner per phase is refit
on a matched case's full curve, kept continuous at 2008-01-01 and monotone, and evaluated at the 80
template quarters (2008 → 2028).

**Choosing the best 10.** "Best performing" = highest oil cumulative in 2028. To avoid submitting 10
near-identical cases, the 100 best-matching sets are clustered into 10 groups (k-means) and each
group contributes its best 2028-oil member. Result: 10 diverse cases that all reproduce the history;
oil rises from 38.4 MMSTB (2008, observed) to ~91 MMSTB (2028) across the selection.

**Deliverable.** `10_Final_deliverables.xlsx` = the provided 09 template with the Date column
untouched and the Gas/Oil/Water cumulative columns filled for the 10 cases — validated on readback
(10 sheets × 80 quarterly rows, dates identical to the template, values finite, monotone, and
continuous with the observed history).

---

## 8. One-line summary for the video

> "The simulator is the truth but too slow; our neural-network proxy has learned the simulator's behavior
> from 70 solved cases. It predicts a full 10-year production curve in milliseconds with about 4x better
> accuracy than a classical baseline — fast enough to search thousands of parameter combinations for a
> history match. From those matches we keep the diverse best ones and extend them to 2028 with decline
> models that were validated on all 100 known cases."

---

## 9. Key files (for reference)

- `cod/proxy_model.py` — the step-1 pipeline: data loading, features, network, training, tuning, evaluation.
- `cod/proxy_outputs/final_model.pt` — the trained 5-model ensemble + everything needed to predict.
- `cod/proxy_outputs/metrics.csv` — all step-1 accuracy numbers vs. baselines (train / validation / test).
- `cod/forecast_pipeline.py` — cumulative twin proxy, parameters → grid-features mapping, curve families,
  forecast, 10-case diversity selection, deliverable writer.
- `cod/history_match.py` — the 45,000-set search and its artifacts (`hm_*.csv`, `hm_summary.json`, `hm_*.png`).
- `cod/proxy_outputs/forecast_cum.csv`, `hm_top10_parameters.csv`, `forecast_fan.png` — the selected 10 cases,
  their 80-quarter forecast, and the history+forecast fan chart.
- `10_Final_deliverables.xlsx` — the final deliverable (workspace root; the 09 template filled for the 10 cases).
- `cod/check_checkpoint.py` — quick health check of the saved checkpoints.
- `cod/notebook1_data_eng.ipynb` — data preparation and the grid analysis.
- `cod/notebook2_sample_model1.ipynb` — the step-1 modeling story with plots (best source for slides).
- `cod/notebook3_history_match_forecast.ipynb` — steps 2–4: search, matches, family backtest, forecast, deliverable.
