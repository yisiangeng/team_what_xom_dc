# Summary — Best Model and Results

**Short version:** the best model is a tuned **5-seed MLP ensemble** (feature set `p4g8` = 4 uncertainty
parameters + 8 grid features, hidden layers 128×128). On the unseen test cases (86–100) it reproduces the
simulator's oil/gas/water curves with a pooled RMSE of **32.5 rate units — ~4.2× better than the ridge
baseline** and ~23× better than predicting the average curve. The same architecture, retrained on
cumulative targets ("cumulative twin"), drives the history matching and the 20-year forecast that fill
the deliverable.

---

## 1. Which model is the best

The winner of the model comparison and of the cross-validated hyperparameter search
(`cod/proxy_outputs/best_config.json`):

| Setting | Value |
|---|---|
| Feature set | `p4g8` — 4 parameters + 8 grid summary features |
| Architecture | MLP, 2 × 128 hidden units, SiLU, dropout 0.05 (~34k weights) |
| Target | mean-curve residuals, standardized (no log transform) |
| Training | Adam, lr 1e-3, cosine decay, early stopping (patience 300) |
| Ensemble | 5 seeds averaged → `final_model.pt` |
| Search result | Winner of the two-stage search; full 5-fold CV RMSE **36.3** |

Why it won:

- **Best CV score** of all 36 tuned configurations (two-stage search: single-fold screen, then full
  5-fold CV on the top 3).
- **Grid features help a lot:** on the fold-0 screen, the 8 grid summaries cut the error from 55.0
  (`p4`) to 30.9 (`p4g8`) at equal settings; the winner's full 5-fold CV score is 36.3 — they make the
  global scaling effects of the four knobs easy to read.
- **Ensembling stabilizes:** the 5-seed average removes single-run luck; validation metrics end up close
  to training metrics (no overfitting gap).
- **It beats both baselines on every split and every phase** — see below.

## 2. Results — rate proxy, unseen test cases 86–100

Pooled RMSE (rate units; lower is better):

| Model | Train | Validation 71–85 | **Test 86–100** |
|---|---|---|---|
| **DNN ensemble (`p4g8`)** | 28.3 | 32.7 | **32.5** |
| Ridge (same 12 inputs) | 110.9 | 127.6 | 135.8 |
| Mean curve (no case info) | 685.5 | 719.9 | 763.5 |

Per-phase test RMSE:

| Phase | DNN ensemble | Ridge | Mean curve |
|---|---|---|---|
| Oil (STB/d) | 41.2 | 190.8 | 1,015.9 |
| Gas (MSCF/d) | 13.9 | 68.4 | 369.0 |
| Water (STB/d) | 35.7 | 119.5 | 762.1 |
| **Pooled** | **32.5** | 135.8 | 763.5 |

- Test pooled `nRMSE` 0.0073 (ours) vs 0.0306 (ridge) vs 0.1722 (mean curve); pooled R² ≈ 0.99995.
- Test ≈ validation ≈ train: the model generalizes — no sign of overfitting.
- Hardest cases are among the few whose parameters lie slightly outside the training range (75, 82, 87, 98;
  the three largest held-out errors are 82, 87, 98); this is reported openly, not hidden.

## 3. Results — cumulative twin proxy

Same architecture and protocol, retrained on cumulative targets (the basis for matching and forecasting).
Pooled test RMSE (physical units):

| Model | Train | Validation | **Test** |
|---|---|---|---|
| **DNN ensemble** | 21,898 | 21,723 | **23,097** |
| Ridge | 143,096 | 166,057 | 181,002 |
| Mean curve | 1,101,121 | 1,158,406 | 1,217,280 |

≈ **7.8× better than ridge**, `nRMSE` 0.0022 vs 0.0173 (test pooled).

## 4. Results — history matching (step 2)

- **45,000 parameter sets** (40,000 space-filling Sobol + 5,000 local polish) scored against the observed
  1998–2008 history in **~2.2 s** on CPU (the proxy answers in milliseconds).
- **Best misfit 0.00385** (normalized RMS error over 3 phases × 41 quarters) — better than the best of
  the 100 provided cases (case 6, misfit 0.0040) and ~14× better than a typical provided case (median 0.054).
- The history does **not** pin one answer: ~4,900 sets sit within 25% of the best misfit; basically all
  agree that the porosity multiplier ≈ 1.36, while the other three parameters stay loosely constrained.
- Top-20 matches saved as `hm_matched_models.csv` (step 3 "history-matched models").

## 5. Results — 20-year forecast and the best-10 selection (step 4)

- **Decline-family backtest** (fit on quarters 1–25 of all 100 known cases, scored on 26–41), median
  relative RMSE:

  | Family | Oil | Gas | Water |
  |---|---|---|---|
  | **Arps** (chosen for oil & gas) | **5.0%** | **5.0%** | 148% |
  | Gompertz (chosen for water) | 13.3% | 13.3% | **107%** |
  | Rational | 10.9% | 10.8% | 128% |

- Each matched case is extended with the winner for its phase: continuous at 2008-01-01, monotone,
  evaluated at the 80 template quarters to 2028.
- **Diversity-aware selection:** the best-matching sets are clustered (k-means, one representative per
  cluster) so the 10 submitted cases are different plausible matches, not 10 near-copies.
- Selected 10 cases reproduce the history and forecast **oil from ~38.6–38.7 MMSTB (2008, matched-model
  prediction; observed 38.38 MMSTB) to ~90.9–91.2 MMSTB (2028)**, gas ~14.0 → ~33.1 million MSCF,
  water ~12.6 → ~16.5 MMSTB.
- **Deliverable** `10_Final_deliverables.xlsx`: all 10 sheets filled with 80 quarterly rows, validated
  on readback (correct row counts, dates untouched, values finite, monotone and continuous with history).

## 6. Caveats (what the results do not say)

- **No ground truth after 2008** exists in the data; the forecast is a validated parametric
  extrapolation, not a simulator answer key.
- **Water is the hardest phase** — every family's backtest error on water is large (breakthrough
  behavior); treat water forecasts with more uncertainty.
- The 10 submitted cases represent the **remaining uncertainty spread** of the history match, not one
  "true" field model.
- A few validation/test cases lie slightly outside the training parameter range; accuracy there is
  lower but is included in the reported test metrics.

## 7. Where to read more

- `WORKFLOW.md` — how the model was built (data cleaning → features → model → comparison).
- `MODEL_CONCEPT.md` — the plain-language "why" (video-script style).
- `cod/README.md` — commands to reproduce everything.
- `cod/proxy_outputs/metrics.csv`, `cumulative_metrics.csv`, `hm_summary.json`, `family_backtest.csv` —
  the raw numbers behind this summary.
