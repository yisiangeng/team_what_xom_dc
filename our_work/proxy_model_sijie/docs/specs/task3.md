# Task 3: Validation, Backtest, Selection, Freeze and Test

Status: draft for implementation, 13 Oct 2026, on branch `task3` (from `main` at d7015a6, Task 2 merged, D-023). Review 1 (14 Oct 2026): returned with 1 blocker (D-032). Its rulings are in §0.5. Review 2 (14 Oct 2026): approved (D-033). The frozen choice is D-034.

- **Base:** PRD §7 (metrics, selection rule, outputs), PRD §9 (backtest; it runs before the freeze), PRD §8 (the mismatch O(x) that ρ and local RMSE need), the rulings D-020, D-021 and D-022 (P1, P2, P3, fallback secondary, coverage, GOR, max-epoch flags), and the Task 2 outputs (`docs/specs/task2.md` §5).
- **Departures from the PRD:** D-024 to D-030 (§0.2), logged in `docs/decisions.md`.

**Label key.** As in `docs/specs/task2.md`:

- "source: PRD §n" means the PRD states it.
- "source: Article N, p. X, Eq. Y" cites the journal page. The PDF pages of the text copies in `refs/` map as follows:
  - Article 4: journal page = PDF page + 1100.
  - Article 6: article page = PDF page − 1.
- "source: D-0xx" cites `docs/decisions.md`.
- Other sources are in `docs/references.md`.
- "assumption" means that this spec decides it and gives a reason.

## 0. Summary

### 0.1 What Task 3 does

Five commands. Each one writes one `runs` row (V8), a log file, and its rows in the database.

| Order | Command | What it does | Splits it reads |
| --- | --- | --- | --- |
| 1 | `python -m dwc26 evaluate --split val` | Loads every Task 2 candidate (7 proxies, B0, B1). Computes the error metrics on the train cases and every §2 metric on the validation cases, including the 80-step validation continuation (P1, P2, P3, GOR), the coverage and the cost. Writes `metrics`, the report and the validation figures | train, val |
| 2 | `python -m dwc26 backtest` | Retrains rnn, lstm and dmd at the cutoff (`train --t-cut 29 --settings-from <candidate run>`). Continues each variant over the 12 quarters after the cutoff, and compares it with the simulated curves and with B2. Validation cases only | train, val |
| 3 | `python -m dwc26 select` | Applies the selection algorithm (§2.9). Proposes the primary and the secondary. Writes `selection`, `outputs/reports/selection.md`, and refreshes `model_comparison.md` | metrics of 1 and 2 |
| 4 | `python -m dwc26 freeze --model <primary> [--secondary <id>]` | Verifies the tag `freeze-proxy`, the decision text and the lineage. Records the frozen choice in the table `freeze`. It does not create the tag | none |
| 5 | `python -m dwc26 evaluate --split test` | Runs once, after the freeze. Computes all metrics, including the backtest, on cases 86 to 100 for all models. Refuses to run unless the guards G1 to G7 pass (§1.10) | train, test |

`--config` defaults to `configs/evaluate.yaml` for all five commands, so the commands in CLAUDE.md work as written.

**Sequence for V3.** The spec is built around this sequence. Rules for each actor:

1. **Data scientist.** Implements everything, including the test path (step 5). Tests the test path on synthetic data only, in a temporary git repository (E24).
2. **Data scientist.** Runs commands 1, 2 and 3 on the real data, then reports the run IDs, `selection.md` and `model_comparison.md`. Does **not** run `freeze` or `evaluate --split test` on the real database. The guards refuse it anyway, because `main` has no tag `freeze-proxy`.
3. **Researcher.** Reviews the code (including the test path) and the proposed choice. Any bug in the test path is fixed before it runs, because the test metrics can be computed only once.
4. **Main session, after approval.**
   1. Merges `task3` into `main`.
   2. The researcher writes the choice (both proxy IDs) in `docs/decisions.md`, and the main session commits it.
   3. The main session runs `git tag -a freeze-proxy -m "D-0xx"` on that commit.
   4. Runs `python -m dwc26 freeze --model <primary> --secondary <secondary>`.
   5. Runs `python -m dwc26 evaluate --split test` once.
5. **Researcher.** Reports the test metrics. The choice does not change after this (PRD §7 step 7).

### 0.2 Departures from the PRD

| ID | PRD | This spec | Reason |
| --- | --- | --- | --- |
| D-024 | Metric formulas (§7), event definitions as written | RMSE, NRMSE and J use all T dates (PRD formula). R² pooled, R² case and coverage use the producing window k ≥ k0. Mismatch metrics use the match window K. Plateau end is searched from the curve's maximum. An event in one curve only is counted, not scored | §2.2, §2.3. The pre-onset zeros are exact by construction (D-013). They inflate the pooled R² denominator. The literal plateau rule gives k = 0, where the rate is 0 |
| D-025 | ρ and local RMSE, "source: Article 6". σ_proxy "from Task 3" | `hm/objective.py` is built now and Task 4 reuses it. ρ, R²_O and local RMSE use O_obs (σ_proxy = 0) for every model. Task 3 stores each proxy's per-date validation RMSE as σ_proxy for Task 4. ρ and local RMSE are relabelled "assumption, motivated by Article 6" | §0.4, §2.4 |
| D-026 | Backtest in Task 5 (§9). "Beat B2 at horizons 9 to 12" | The backtest is a Task 3 command. t_cut and H are read from the dates. The V6 rule pools horizons 9 to 12 for oil and for water. The tie-break uses one backtest score BT | §2.8 |
| D-027 | Selection steps 1 to 5 (§7) | An exact algorithm: eligibility (steps 1 and 2, P1, P2), banded tie-breaks relative to the best, a deterministic last key, the secondary rule with the D-020 fallback, and DMD as secondary only (D-022) | §2.9 |
| D-028 | "Freeze: write the choice … Tag the commit" (§7 step 6). `freeze --model <run_id>` | `freeze` verifies and records; it does not tag. `--secondary` is added. The test scoring is guarded by git (G1 to G4), the database (triggers), and the config (G7). A second scoring needs a logged override reason | §1.10, §2.10 |
| D-029 | `metrics`: run_id, split, fluid, metric, value (§5) | Adds `proxy_id` and `idx`, the split `none` and the fluid `all`. New tables `selection`, `freeze` and `test_scoring`, with triggers | §1.6 |
| D-030 | No plotting library is pinned | matplotlib 3.10.8 in a new extra `eval`. Ten figures | §1.8, §3.2 |

### 0.3 Facts from Tasks 1 and 2 used here

- **Real sizes** are read at run time, never hard-coded (V5): T = 41, k0 = 4, n_train = 70, n_val = 15, n_test = 15, p = 4.
  - y_factor = (15678.18 STB/d, 5695.88 MSCF/d, 9166.93 STB/d).
  - `hist_phys[0:4] = 0`. The match window K (1999-01-01 to 2008-01-01) is k = 4..40, 37 dates (task1 §5).
- **d = `hist_phys`** (R1, D-004). Never the `*_rate_reported` columns, which are one quarter earlier.
- **Candidates (D-023).** All have status ok, git_dirty 0, `arrays_sha256` 451fcdab…, k0 = 4 and T_fit = 41.

| Model | Run | Variants | Validation J (Task 2) | Notes |
| --- | --- | --- | --- | --- |
| b0 | `train-20261012T235731641838Z` | main | 0.0710 | Val RMSE oil 917.8, water 748.8 |
| b1 | `train-20261012T235732605558Z` | main | 0.0199 | Val RMSE oil 239.1, water 220.0 |
| rnn | `train-20261012T235952401450Z` | A, B, C | A 0.0038, B 0.0044, C 0.0049 | 9 trials reached `max_epochs` (D-021 b) |
| lstm | `train-20261013T000744561060Z` | A, B, C | A 0.0029, B 0.0046, C 0.0035 | C: rising oil in 15 of 15 validation cases, ratio up to 1.50 (fails P2) |
| dmd | `train-20261013T005202581280Z` | main | 0.0093 | A2 rerun. Rank 8, flat tail (P3, D-022). The earlier run `train-20261012T235740838203Z` is superseded and is not a candidate |

- **Continuation (Task 2 reports).** 0 clipped values for all 7 proxies (P1 passes). Coverage preview: 0.65 to 0.92 for the seed ensembles, 0.995 to 0.998 for DMD.
- **Backtest run time.** rnn 460 s and lstm 136 s for a full tuning run (README). With `--settings-from` there are 15 member fits for each model and no tuning.

### 0.4 PRD claims checked (Articles 4 and 6)

| PRD claim | Finding | Consequence |
| --- | --- | --- |
| "R² pooled — Article 4, Eq. 16" (§7) | **Partly supported.** Eq. 16 (p. 1105) is R² = 1 − Σ_j (y(x̃_j) − ŷ(x̃_j))² ÷ Σ_j (y(x̃_j) − ȳ)² over an independent test sample, with ȳ the mean of the test outputs. Article 4 applies it to a **scalar** output: the objective function, on independent samples of 200 (Punq) and 101 (Brugge) points (p. 1114). It does not pool dates of a time series | The form is cited. Pooling over cases and dates is our extension (D-024). Article 4's own use, R² of the objective function, is added as the report-only metric R²_O (§2.4) |
| "Mismatch rank ρ — Article 6" (§7) | **Not supported as a definition.** Article 6 evaluates proxies with RMSE (Eq. 21, p. 13), MAE, MAPE, R² (Eq. 24, p. 14) and relative error. It has no rank correlation and no mismatch. It supports the motivation: proxies with test R² 0.90 to 0.97 (Table 8, p. 21) gave optimal points far from the simulator optimum (Euclidean distance, p. 16) and NPV errors (Table 9, p. 30). It concludes that proxies must also be judged on "optimization-relevant dynamics" (p. 1; p. 17–18) | Relabelled "assumption, motivated by Article 6" (D-025) |
| "Local RMSE — Article 6" (§7) | **Not supported as a definition**, for the same reason. The closest item is Table 9 (p. 30): the proxy NPV ("Cal.") against the simulator NPV ("Actual") at each proxy's optimum | Relabelled as for ρ. The report adds the median O_prx − O_sim of the local cases, after Table 9 (§2.4) |
| "σ_proxy … Assumption, after article 6" (§8) | Article 6 has no σ_proxy. The PRD already labels it an assumption | Task 3 stores it (§2.2), Task 4 uses it. No change |
| "α = 0.10 — Article 4, Eq. 18" (§8; checked now because `hm/objective.py` is built in Task 3) | **Partly supported.** Eq. 18 (p. 1109) is OF(x) = ½ Σ_p ω_p Σ_k ((y_p(x, t_k) − y_ref,p(t_k)) ÷ σ_p(t_k))², with a generic σ_p(t). The 10% comes from the case studies: Punq, σ_p(t) = max(0.1 y_ref,p(t), 0.005) (p. 1111). Brugge, "10% of the data, with a lower limit to 50" (p. 1113–1114). Both floors are absolute, not β × max d | α = 0.10 is cited to p. 1111 and p. 1113–1114. β stays an assumption (PRD). The PRD's O equals 2 × OF with ω_p = 1, and the ½ returns in log p = −O ÷ 2. No change to O |

### 0.5 Review 1 rulings (14 Oct 2026, D-032)

These rulings are part of the spec. Each one is an assumption of this spec unless a source is given.

| # | Item | Ruling |
| --- | --- | --- |
| R1 | `GitOps(None)` (§1.3, §1.10) | With `freeze.repo_dir: null`, `GitOps` finds the top level from the package directory, then runs every git command with the top level as cwd. Reason: `git diff -- <paths>` and `git ls-tree <rev> -- <path>` read paths relative to the cwd (git-ls-tree documentation, DESCRIPTION). Run from `src/dwc26/evaluation`, G4 then never fires and `show` never finds `docs/decisions.md`. New test **E18b**: build the `repo_dir: null` path from a subdirectory of `tmp_git_repo` (for example, a helper that `GitOps(None)` calls with the package directory). `show(c1, "docs/decisions.md")` must not be None, and `changed_paths` must list a changed frozen file |
| R2 | Failure after `finish_run(ok)` (§1.4, §1.5) | After `ctx.finish`, an exception in the figures or the reports does not change the run status and does not remove any `outputs` key. It is logged at ERROR, stored as `outputs.post_finish_error`, and the command exits 1. This protects the lineage keys of the single test run (§1.5.5 step 6). New test **E24b**: run the E24 test scoring with `report.plots: true`, or with a plot function patched to raise. The test-run outputs keep `freeze_id`, `scoring_id` and `tag_commit`, and `write_model_comparison` still runs |
| R3 | `continuation.end_date` at load time (§3.1) | `load_eval_config` checks that `end_date` is a whole number of quarters after `mismatch.window[1]`. This is only a pre-check. The binding check is the run-time `quarters_between(dates[-1], end_date)` (§1.5.2 step 1) |
| R4 | `p3_D_hist_true` (§2.7 P3 step 4) | The median D_hist of the simulated oil curves over every case of the split with positive simulated oil at steps T − 1 − L_h and T − 1. `p3_min_decline` is not applied. The result is a reference that does not depend on the model, so it is the same for every proxy. Report only |
| R5 | E27 timing | One untimed warm-up figure is drawn first (the matplotlib import and font cache). The 2 s limit applies to each figure after it |
| R6 | E24 `select` outcome | E24 fixes the `select` outcome, because no tiny model is eligible. E24 tests the V3 path. E11 and E23 test the algorithm |
| R7 | Test run rows (§1.5.5 step 4) | The test run writes rows of split `test` only. The train-split rows are in E, and they would be identical (same models, same train arrays) |
| R8 | Max-epoch count (§2.7) | The rule is n_epochs ≥ max_epochs. On the real runs it gives 8 rnn trials (trial 14 stopped at 988 by patience) and 8 lstm trials. None is selected, and no final member reached `max_epochs`. This count replaces the "9" in D-021 (b). The ruling does not change |
| R9 | No S4 secondary (§2.9 step 7) | DMD fails step 2 (V6 against B2), so it is not eligible as primary or as secondary. The D-020 fallback applies as written: the secondary is the best eligible S3 proxy of the other family, and the method uncertainty covers S3 only. No amendment is needed |
| R10 | Report text on new wells (§1.7.1 item 5, §1.7.2 item 6) | The user answered on 14 Oct 2026: there are no new wells and no development after 2008 (D-031). The reports must say so, and must not call the question open |

## 1. Inputs and outputs

### 1.1 Inputs

| Input | Content | Read by |
| --- | --- | --- |
| `data/processed/dwc26.sqlite` | Task 1 tables via `read_task1_arrays` (F2 guard); Task 2 `runs` and `predictions` | `load_split`, `resolve_candidates` |
| `outputs/models/<run_id>/<variant>/` | Task 2 models, loaded with `load_model(..., con=con)` (sha256 and scaler checks, V4) | `evaluate`, `backtest` |
| `outputs/models/<run_id>/trials.csv`, `config.yaml` | Trial epochs and `max_epochs` (max-epoch flags); DMD trial `fit_s` | `cost_and_flags` |
| `configs/evaluate.yaml` | All Task 3 settings (V7), §1.2 | `load_eval_config` |
| `configs/models/{rnn,lstm,dmd}.yaml` | Model configs for the backtest retraining | `run_backtest` |
| git repository | Tag, HEAD, status, diffs, `docs/decisions.md` at the tag | `GitOps` (freeze and test only) |

### 1.2 Config `configs/evaluate.yaml`

`load_eval_config(path)` validates every key against §3. A missing key, an unknown key or a value outside the allowed range raises `ConfigError`. These are the deliverable defaults:

```yaml
seed: 0                                    # X draw of the cost benchmark
paths:
  db: data/processed/dwc26.sqlite
  models: outputs/models
  reports: outputs/reports
  figures: outputs/figures/task3
  logs: outputs/logs
candidates:                                # train run of each model: 'latest' or an explicit run ID (§1.5.1)
  runs: {b0: latest, b1: latest, rnn: latest, lstm: latest, dmd: latest}
  require_clean: true                      # candidate runs must have git_dirty = 0
strategy: {rnn: S3, lstm: S3, dmd: S4}     # PRD §6
runtime:
  torch_threads: 1
consistency_rel_tol: 1.0e-7                # recomputed vs stored Task 2 predictions, relative to y_factor
metrics:
  objective_fluids: [oil, water]           # J, steps 1 and 2, BT (PRD §7)
  coverage_z: 1.959963984540054            # 95% two-sided normal quantile
  coverage_target: [0.90, 0.98]            # PRD §7
events:
  plateau_frac: 0.98                       # PRD §7
  breakthrough_frac: 0.01                  # PRD §7
mismatch:
  window: ["1999-01-01", "2008-01-01"]     # K, inclusive (PRD §8)
  fluids: [oil, water]                     # PRD §8
  alpha: 0.10                              # Article 4, p. 1111 and p. 1113-1114
  beta: 0.02                               # PRD §8 (assumption)
  local_n: 5                               # PRD §7
continuation:
  end_date: "2028-01-01"                   # n_fc = quarters from the last history date (80 on the real data)
plausibility:
  p2_mode: reject                          # reject | report (D-020; the user confirmed no new wells, D-031)
  p2_ratio: 1.05                           # D-020
  p2_case_share: 0.5                       # D-020: fail if more than this share of the cases rise
  p3_hist_quarters: 8                      # D-022
  p3_fc_quarters: [9, 16]                  # D-022
  p3_ratio_min: 0.5                        # D-022
  p3_min_decline: 0.002                    # per quarter; cases with a smaller history decline are excluded from R
  gor_rel_tol: 0.005                       # D-021 (c)
  gor_oil_frac: 0.01                       # GOR only where oil > this share of its maximum (Task 2 report)
backtest:
  cutoff_date: "2005-01-01"                # PRD §9
  models: [rnn, lstm, dmd]                 # PRD §9 step 2
  model_configs: {rnn: configs/models/rnn.yaml, lstm: configs/models/lstm.yaml, dmd: configs/models/dmd.yaml}
  v6_horizons: [9, 12]                     # PRD §9 step 5, inclusive
  reuse: {}                                # optional: model -> an existing backtest train run (§1.5.2)
selection:
  band: 0.05                               # PRD §7 step 4
  secondary_only: [dmd]                    # D-022
cost:
  n_rows: 10000                            # PRD §7
  repeats: 3
freeze:
  tag: freeze-proxy                        # PRD §7 step 6
  repo_dir: null                           # null = the repository that holds the package
  decisions_path: docs/decisions.md
  frozen_paths: [src/dwc26/evaluation, src/dwc26/models, src/dwc26/forecast/backtest.py,
                 src/dwc26/forecast/dates.py, src/dwc26/hm/objective.py, configs/evaluate.yaml, configs/models]
  min_override_chars: 10
report:
  plots: true
  dpi: 120
```

### 1.3 Modules and signatures

All arrays are `np.float64` unless stated. V9 failures raise `ContractError`. V3 refusals raise `GuardError(ContractError)`, defined in `dwc26.evaluation.guards`. `FLUIDS` comes from `dwc26.data.load`.

| Module | Content |
| --- | --- |
| `src/dwc26/evaluation/metrics.py` | Existing `rmse`, `nrmse`, `objective_j` (unchanged). New: `r2_pooled`, `r2_case`, `rmse_by_date`, `rmse_pooled`, `coverage`, `spearman`, `r2_scalar` |
| `src/dwc26/evaluation/events.py` | `event_indices`, `event_errors` |
| `src/dwc26/evaluation/plausibility.py` | `continuation_checks`, `gor_range`, `gor_check`, `max_epoch_flags` |
| `src/dwc26/evaluation/splits.py` | `SplitArrays`, `TrainRef`, `load_split`, `train_reference` |
| `src/dwc26/evaluation/store.py` | `Candidate`, `MetricRow`, `METRICS`, `resolve_candidates`, `write_metrics`, `read_metrics`, `metric_value`, `read_rmse_by_date`, `latest_ok_run` |
| `src/dwc26/evaluation/guards.py` | `GuardError`, `GitOps`, `TestTicket`, `check_freeze_tag`, `claim_test_scoring`, `finish_test_scoring`, `verify_ticket` |
| `src/dwc26/evaluation/evaluate.py` | `EvalResult`, `evaluate_models`, `cost_and_flags`, `run_evaluate` |
| `src/dwc26/evaluation/selection.py` | `CandidateScore`, `Selection`, `rank_candidates`, `select`, `run_select` |
| `src/dwc26/evaluation/freeze.py` | `run_freeze` |
| `src/dwc26/evaluation/report.py` | `write_model_comparison`, `write_selection_report` |
| `src/dwc26/evaluation/plots.py` | `plot_curves`, `plot_parity`, `plot_backtest` (matplotlib, Agg backend, imported lazily) |
| `src/dwc26/hm/__init__.py`, `src/dwc26/hm/objective.py` | `window_indices`, `sigma_obs`, `MismatchSpec`, `build_mismatch_spec`, `mismatch` (Task 4 reuses them) |
| `src/dwc26/forecast/__init__.py`, `src/dwc26/forecast/dates.py` | `quarters_between`, `forecast_dates` (Task 5 reuses them) |
| `src/dwc26/forecast/backtest.py` | `cutoff_index`, `persistence`, `backtest_errors`, `run_backtest` |
| `src/dwc26/config.py` | Adds `load_eval_config(path) -> dict` |
| `src/dwc26/data/db.py` | Adds the §1.6 DDL to `SCHEMA` (tables and triggers) and `matplotlib` to `PACKAGES` |
| `src/dwc26/cli.py` | Subcommands `evaluate`, `backtest`, `select`, `freeze` |

```python
# metrics.py (new functions). y arrays (n, T, F); `window` is a boolean (T,) mask or None (all dates)
def r2_pooled(y_true, y_pred, window=None) -> np.ndarray: ...       # (F,); NaN where the denominator is 0
def r2_case(y_true, y_pred, mean_curve, window=None) -> np.ndarray: ...  # mean_curve (T, F); NaN where denominator 0
def rmse_by_date(y_true, y_pred) -> np.ndarray: ...                  # (T, F)
def rmse_pooled(y_true, y_pred) -> np.ndarray: ...                   # (F,) over every axis but the last (= rmse)
def coverage(y_true, y_pred, std, z: float, window=None) -> tuple[np.ndarray, np.ndarray]: ...
    # (share (F,), mean interval width 2 z s (F,))
def spearman(a: np.ndarray, b: np.ndarray) -> float | None: ...     # (n,), (n,); None if either is constant or n < 3
def r2_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float | None: ...  # (n,); None if y_true is constant
```

```python
# events.py
EVENTS = ("plateau_end", "breakthrough", "crossover")
def event_indices(Y_phys: np.ndarray, k0: int, cfg_events: dict) -> dict[str, np.ndarray]: ...
    # (n, T, 3) -> {event: (n,) int64, -1 = no event}, §2.3
def event_errors(Y_true, Y_pred, t_days, k0, cfg_events) -> dict[str, dict]: ...
    # {event: {mae_days, bias_days, n_both, n_mismatch, n_none}}; mae and bias None if n_both = 0
```

```python
# hm/objective.py (PRD §8; D-025)
def window_indices(dates: np.ndarray, start, end) -> np.ndarray: ...   # (|K|,) int64, start <= dates[k] <= end
def sigma_obs(d: np.ndarray, alpha: float, beta: float) -> np.ndarray: ...  # (|K|, F_O) -> (|K|, F_O)

@dataclass(frozen=True)
class MismatchSpec:
    k_idx: np.ndarray          # (|K|,) int64
    fluid_idx: np.ndarray      # (F_O,) int64, indices into FLUIDS
    d: np.ndarray              # (|K|, F_O) physical history rates, hist_phys[k_idx][:, fluid_idx]
    sigma_obs: np.ndarray      # (|K|, F_O) > 0
    sigma_proxy: np.ndarray    # (|K|, F_O) >= 0; zeros for O_obs
    sigma: np.ndarray          # (|K|, F_O) = sqrt(sigma_obs² + sigma_proxy²) > 0

def build_mismatch_spec(hist_phys: np.ndarray, dates: np.ndarray, cfg_mismatch: dict,
                        sigma_proxy: np.ndarray | None = None) -> MismatchSpec: ...
def mismatch(Y_phys: np.ndarray, spec: MismatchSpec) -> np.ndarray: ...   # (n, T, 3) physical -> (n,) O >= 0
```

```python
# splits.py
@dataclass(frozen=True)
class SplitArrays:
    split: str                 # 'train' | 'val' | 'test'
    case_id: np.ndarray        # (n,) int64, ascending
    X: np.ndarray              # (n, p), scaled, [0, 1]
    Y_phys: np.ndarray         # (n, T, 3), physical, >= 0
    dates: np.ndarray          # (T,) datetime64[D]
    t_days: np.ndarray         # (T,)
    hist_phys: np.ndarray      # (T, 3), R1 history rates (D-004)
    y_factor: np.ndarray       # (3,)
    prepare_run_id: str
    arrays_sha256: str

@dataclass(frozen=True)
class TrainRef:                # statistics of the train cases only (V1)
    k0: int
    mean_curve: np.ndarray     # (T, 3) physical, the B0 curve
    gor_range: tuple[float, float]

def load_split(con, split: str, *, ticket: "TestTicket | None" = None) -> SplitArrays: ...
def train_reference(train: SplitArrays) -> TrainRef: ...
```

`load_split`:

1. Calls `read_task1_arrays(con)` and checks `arrays_sha256` against the prepare run, as `load_dataset` does (reuse its helper).
2. Selects rows by the stored `split` column (V2, V5).
3. For `split = 'test'`, it first calls `verify_ticket(con, ticket)`. Without a valid ticket it raises `GuardError` before any test row is selected.
4. For `train` and `val`, `ticket` must be None. The returned object holds no row of another split.

```python
# store.py
@dataclass(frozen=True)
class Candidate:
    proxy_id: str              # '<run_id>/<variant>'
    model: str                 # 'b0' | 'b1' | 'rnn' | 'lstm' | 'dmd'
    variant: str
    run_id: str
    is_proxy: bool
    strategy: str | None       # cfg['strategy'][model] for proxies; None for benchmarks
    label: str                 # '<model>/<variant>', e.g. 'lstm/A'

@dataclass(frozen=True)
class MetricRow:
    proxy_id: str              # candidate proxy ID, or 'b2'
    split: str                 # 'train' | 'val' | 'test' | 'none'
    fluid: str                 # 'oil' | 'gas' | 'water' | 'all'
    metric: str                # a key of METRICS (§1.6)
    idx: int                   # horizon h, date_idx, or -1
    value: float | None

def resolve_candidates(con, cfg: dict, arrays_sha256: str) -> list[Candidate]: ...   # §1.5.1
def write_metrics(con, run_id: str, rows: list[MetricRow]) -> int: ...   # no commit; V9 (§1.9)
def read_metrics(con, run_id: str, *, split: str | None = None) -> list[MetricRow]: ...
def metric_value(rows, proxy_id, split, fluid, metric, idx=-1) -> float | None: ...  # KeyError if absent
def read_rmse_by_date(con, run_id: str, proxy_id: str) -> np.ndarray: ...  # (T, 3) val split; Task 4 sigma_proxy
def latest_ok_run(con, command: str, *, where: dict | None = None) -> str | None: ...
    # latest `start` among runs with this command, status 'ok' and runs.outputs[key] == value for each key in where
```

```python
# guards.py
class GuardError(ContractError): ...

class GitOps:
    def __init__(self, repo_dir: Path | None): ...   # None = the repository that holds the package
    def head(self) -> str: ...
    def resolve_tag(self, tag: str) -> str | None: ...       # git rev-parse -q --verify refs/tags/<tag>^{commit}
    def is_ancestor(self, a: str, b: str) -> bool: ...       # git merge-base --is-ancestor a b: 0 True, 1 False
    def is_dirty(self) -> bool: ...                          # git status --porcelain --untracked-files=no non-empty
    def changed_paths(self, a: str, b: str, paths: list[str]) -> list[str]: ...  # git diff --name-only a b -- paths
    def show(self, rev: str, path: str) -> str | None: ...   # git show rev:path
    # Any other git failure (git missing, not a repository, exit code not 0 or 1) raises GuardError: fail closed.

@dataclass(frozen=True)
class TestTicket:
    scoring_id: int
    run_id: str                # the evaluate-test run
    freeze_id: int
    tag_commit: str

def check_freeze_tag(git: GitOps, cfg: dict) -> tuple[str, str]: ...   # G1 to G3 -> (tag_commit, head)
def claim_test_scoring(con, run_id: str, cfg: dict, git: GitOps,
                       override_reason: str | None) -> TestTicket: ...  # G1 to G7, then the claim row (§1.10)
def verify_ticket(con, ticket: TestTicket) -> None: ...   # the claim row exists, status 'running', same run_id
def finish_test_scoring(con, ticket: TestTicket, status: str) -> None: ...   # 'ok' | 'failed'; commits
```

```python
# evaluate.py
@dataclass
class EvalResult:
    rows: list[MetricRow]
    preds: dict[str, np.ndarray]         # proxy_id -> (n, T, 3) physical, clipped
    stds: dict[str, np.ndarray | None]   # proxy_id -> (n, T, 3) physical, or None
    conts: dict[str, np.ndarray]         # proxies: proxy_id -> (n, T + n_fc, 3) physical rollout
    O_sim: np.ndarray | None             # (n,)
    O_prx: dict[str, np.ndarray]         # proxy_id -> (n,)
    summary: dict                        # per proxy_id: j, P1, P2, P3, coverage flags (for runs.outputs)

def evaluate_models(models: dict[str, object], cands: list[Candidate], arr: SplitArrays, ref: TrainRef,
                    spec: MismatchSpec | None, n_fc: int, cfg: dict) -> EvalResult: ...
    # Pure computation (no DB, no files). spec is None for the train split.
def cost_and_flags(cand: Candidate, model, models_dir: Path, con, cfg: dict) -> list[MetricRow]: ...  # §2.6, §2.7.5
def run_evaluate(config_path: Path, split: str, *, override_reason: str | None = None) -> str: ...  # run_id
```

```python
# selection.py
@dataclass(frozen=True)
class CandidateScore:
    proxy_id: str; label: str; model: str; strategy: str
    step1: dict[str, bool]          # fluid -> RMSE_val < min(B0, B1)
    step2: dict[str, bool]          # fluid -> RMSE_bt(9..12) < B2
    p1: bool; p2: bool; p3_flag: bool | None
    eligible: bool                  # all of step1, step2, p1, and p2 when p2_mode is 'reject'
    j: float; bt: float; rho: float | None

@dataclass(frozen=True)
class Selection:
    primary: str | None
    secondary: str | None
    fallback_secondary: bool
    status: str                     # 'ok' | 'no_secondary' | 'no_eligible_primary'
    ranking: list[str]              # eligible primary candidates in rank order
    trace: dict                     # j*, T1, b*, T2, and the reasons (for selection.md)

def rank_candidates(scores: list[CandidateScore], band: float) -> list[str]: ...   # §2.9 step 4
def select(scores: list[CandidateScore], cfg: dict) -> Selection: ...               # §2.9
def run_select(config_path: Path, *, override_reason: str | None = None) -> str: ...
```

```python
# freeze.py
def run_freeze(config_path: Path, primary: str, secondary: str | None, *,
               override_reason: str | None = None) -> str: ...   # §1.5.4

# forecast/dates.py
def quarters_between(d0: np.datetime64, d1: np.datetime64) -> int: ...
    # whole quarters from d0 to d1 (same day of month, month difference divisible by 3), else ContractError
def forecast_dates(last: np.datetime64, n: int) -> np.ndarray: ...   # (n,) datetime64[D], last + 3, 6, ... months

# forecast/backtest.py
def cutoff_index(dates: np.ndarray, cutoff) -> int: ...      # t_cut = number of dates <= cutoff
def persistence(Y_phys: np.ndarray, t_cut: int, H: int) -> np.ndarray: ...   # (n, T, 3) -> (n, H, 3), §2.8
def backtest_errors(Y_true_h: np.ndarray, Y_pred_h: np.ndarray, cfg: dict, span: np.ndarray) -> list[MetricRow]: ...
    # (n, H, 3) each; span (3,) = split range of each fluid over all T dates; rows without proxy_id and split
    # (the caller fills them)
def run_backtest(config_path: Path) -> str: ...
```

```python
# plots.py: all values in physical units; one PNG for each call; figures are closed after saving
def plot_curves(path: Path, dates: np.ndarray, fc_dates: np.ndarray, case_id: np.ndarray, Y_true: np.ndarray,
                hist: np.ndarray, curves: dict[str, np.ndarray], fluid: int, title: str, dpi: int) -> None: ...
    # curves: label -> (n, T + n_fc, 3); one panel for each case
def plot_parity(path: Path, Y_true: np.ndarray, preds: dict[str, np.ndarray], k0: int, title: str, dpi: int) -> None: ...
def plot_backtest(path: Path, rmse_h: dict[str, np.ndarray], b2: np.ndarray, title: str, dpi: int) -> None: ...
    # rmse_h: label -> (H, 3); b2 (H, 3)
```

### 1.4 CLI

```text
evaluate --split {val,test} [--config PATH] [--override-reason TEXT]
backtest [--config PATH]
select   [--config PATH] [--override-reason TEXT]
freeze   --model PROXY_ID [--secondary PROXY_ID] [--config PATH] [--override-reason TEXT]
```

- `--config` defaults to `configs/evaluate.yaml`.
- `--override-reason` is accepted only with `--split test`, `select` and `freeze`. It must have at least `freeze.min_override_chars` characters after stripping; otherwise `ConfigError`.
- Exit codes:
  - 0: success. The command prints the run ID.
  - 1: `ConfigError` or `ContractError`, printed to stderr.
  - 3: `GuardError` (V3 refusal), printed to stderr with the guard ID (G1 to G7).
  - 2: argparse usage errors, and the commands that are still not built (`match`, `forecast`, `all`).
- In every failure after `start_run` and before `finish_run(status='ok')`, the `runs` row ends `failed` (task2 §1.3 pattern). A failure after it follows §0.5 R2.
- `freeze --model` takes a proxy ID (D-014). `--secondary` defaults to the secondary of the selection run.

### 1.5 Processing order

Every command:

1. Loads the config.
2. Runs `connect` and `init_schema`.
3. Calls `start_run(con, task='task3', command=<cmd>, cfg, config_path, seed=cfg['seed'], model=None)`.
4. Opens `outputs/logs/<run_id>.log`.

On any exception after `start_run`, the command rolls back the open transaction, calls `finish_run(status='failed', outputs={'error': ...})`, and re-raises. For `evaluate --split test`, it also calls `finish_test_scoring(..., 'failed')` if a ticket exists.

#### 1.5.1 Candidate resolution (`resolve_candidates`)

For each model in `candidates.runs` (config order: b0, b1, rnn, lstm, dmd):

1. **Find the run.**
   - `latest`: the latest `start` among `runs` rows with `task = 'task2'`, `command = 'train'`, this model, `status = 'ok'`, `outputs.t_cut` null, `outputs.settings_from` null, and `outputs.arrays_sha256` equal to the current Task 1 hash.
   - An explicit ID must satisfy the same conditions. Otherwise `ContractError`.
   - None found: `ContractError`.
2. **Clean code.** With `require_clean`, `git_dirty` must be 0. Otherwise `ContractError`.
3. **Variants.** The variants are the keys of `outputs.variants`, sorted. Proxy ID = `<run_id>/<variant>`.
4. **Benchmarks.** `b0` and `b1` must each resolve to exactly one variant, `main`.

The resolved IDs are logged and stored in `runs.outputs.candidates`. On the real database they must equal the D-023 runs (§4.4). If they do not, the data scientist sets the explicit IDs in `configs/evaluate.yaml` and reports it.

#### 1.5.2 `evaluate --split val` (`run_evaluate`)

1. **Setup.**
   - If a neural candidate exists: `torch.set_num_threads(runtime.torch_threads)` and `torch.use_deterministic_algorithms(True)`.
   - `train = load_split(con, 'train')`, `val = load_split(con, 'val')`, `ref = train_reference(train)`.
   - `n_fc = quarters_between(dates[-1], continuation.end_date)`.
   - `spec = build_mismatch_spec(val.hist_phys, val.dates, cfg['mismatch'])`. σ_proxy is None (D-025).
   - Check that n_fc ≥ `p3_fc_quarters[1]`. Otherwise `ConfigError`.
2. **Models.** For each candidate: `load_model(proxy_id, models_dir, con=con, proxies_only=False)`.
3. **Consistency (V8).** For train and val, recompute `predict(X) · y_factor` and compare it with the `predictions` rows of that run and variant. The maximum of |difference| ÷ y_factor_f must be ≤ `consistency_rel_tol`. Otherwise `ContractError`. The maximum is stored in `runs.outputs.consistency`.
4. **Metrics.**
   - `evaluate_models(..., train, ref, None, ...)` for the train split: RMSE, NRMSE, J, the two R², `n_clipped_pred`.
   - `evaluate_models(..., val, ref, spec, n_fc, cfg)` for the validation split: every metric of §2.2 to §2.7.
   - Then `cost_and_flags` for each candidate.
5. **Identity with Task 2.** For each candidate, J_val must equal `runs.outputs.variants[v].j_val` of its train run within 1e-12, and the RMSE within relative 1e-9. Otherwise `ContractError`. Reason: the ranking metric equals the tuning objective (task2 §5).
6. **Database.** In one transaction: `write_metrics`, then `finish_run(status='ok', outputs=...)`.
7. **Files.** If `report.plots` is true: the validation figures (§1.8). Then `write_model_comparison(con, cfg)`.

#### 1.5.3 `backtest` (`run_backtest`)

1. **Window.**
   - `val = load_split(con, 'val')`.
   - `t_cut = cutoff_index(val.dates, backtest.cutoff_date)` and `H = T − t_cut`.
   - The check k0 + 3 ≤ t_cut and `v6_horizons[1]` ≤ H must hold, else `ConfigError`. Real data: t_cut = 29 and H = 12.
2. **Source runs.** `resolve_candidates` restricted to `backtest.models`.
3. **Training.** For each model m:
   - **Reuse.** If `reuse[m]` is set, that run must be a `task2` `train` run with status ok, `outputs.t_cut` = t_cut, `outputs.settings_from` = the source run, and the same `arrays_sha256`.
   - **Otherwise, config check.** Load `model_configs[m]` with `load_model_config`. Compare it with `json.loads(runs.config)` of the source run after setting `data.t_cut` to None in both. A difference raises `ConfigError`, naming the keys (V7: the backtest must use the Task 2 settings).
   - **Otherwise, train.** `bt_run = run_train(model_configs[m], m, t_cut=t_cut, settings_from=source_run)`. This makes a normal Task 2 train run, with its own `runs` row and its own validation-only early stopping on the first t_cut dates.
4. **Errors.** For each variant v:
   - `bt = load_model(f"{bt_run}/{v}", models_dir, con=con)` (proxies only).
   - Check that `bt.T_fit == t_cut`.
   - `F = bt.forecast(val.X, H) · y_factor`, and record `last_n_clipped`.
   - Truth `Y_h = val.Y_phys[:, t_cut:T]`.
   - `backtest_errors(Y_h, F, cfg, span)`, where span is the per-fluid range of the validation truth over all T dates.
   - The rows are stored under the candidate proxy ID `<source_run>/<v>`, split `val`.
5. **B2.** `persistence(val.Y_phys, t_cut, H)` gives the same rows under `proxy_id = 'b2'`.
6. **V6 (B2).** `bt_v6_pass` for each candidate and each objective fluid (§2.8).
7. **Database and files.**
   - `write_metrics` and `finish_run`.
   - `outputs`: `cutoff_date`, `t_cut`, `H`, `v6_horizons`, `source_runs`, `bt_runs`, `mapping` {candidate proxy ID: backtest proxy ID}, `v6` {proxy ID: {fluid: bool}}, `files`.
   - Figure `backtest_val.png`, then `write_model_comparison`.

#### 1.5.4 `select` and `freeze`

**`select` (`run_select`).**

1. **Frozen choice.** If any `freeze` row exists, raise `GuardError` (G6), unless an override reason is given (logged).
2. **Inputs.**
   - E = `latest_ok_run(con, 'evaluate', where={'split': 'val'})`.
   - B = `latest_ok_run(con, 'backtest')`.
   - Both must exist and have `git_dirty` = 0.
   - Both must name the same candidate runs for rnn, lstm and dmd, and the same `arrays_sha256`. The G7 config blocks of E, B and the select run must be equal. Otherwise `ContractError`.
3. **Scores.** Build one `CandidateScore` for each proxy from the `metrics` rows of E and B, split `val` only. A missing metric raises `ContractError`.
4. **Choice.** `select(scores, cfg)` (§2.9).
5. **Database and files.**
   - `selection` rows (one for each proxy, with the role).
   - `finish_run`, with `outputs`: `evaluate_run`, `backtest_run`, `primary`, `secondary`, `fallback_secondary`, `status`, `p2_mode`, `band`, `ranking`, `files`.
   - `outputs/reports/selection.md` (§1.7.2), then `write_model_comparison`.

**`freeze` (`run_freeze`).**

1. **Selection run.** S = `latest_ok_run(con, 'select')`. Its `status` must not be `no_eligible_primary`.
2. **Proposal.** `primary` must equal S's primary. `secondary` (given, or S's) must equal S's secondary. A difference needs an override reason (logged).
3. **Git** (§1.10): G1 to G3 via `check_freeze_tag`. The tag commit must contain S, E and B: each of their `git_commit` values is an ancestor of the tag commit, and each has `git_dirty` = 0.
4. **Decision text.** `git.show(tag_commit, freeze.decisions_path)` must contain the primary and the secondary proxy IDs as exact substrings. Otherwise `GuardError` (G5). This shows that the choice was written before the tag (PRD §7 step 6).
5. **Loadable.** Both proxies load with `load_model(..., con=con, proxies_only=True)`.
6. **Record.** Insert the `freeze` row. The trigger refuses a second freeze unless `override_reason` is set. Then `finish_run`.

#### 1.5.5 `evaluate --split test`

1. **Claim.** `ticket = claim_test_scoring(con, run_id, cfg, GitOps(freeze.repo_dir), override_reason)` (G1 to G7, §1.10). No test row is read before this call.
2. **Lineage.** From the active freeze row (the highest `freeze_id`), take the selection run S. From S, take E and B. The candidates are E's candidates, exactly. The backtest models are B's `mapping`.
3. **Arrays.** `test = load_split(con, 'test', ticket=ticket)`. Train arrays and `ref` as in validation. `spec` is built from `test.hist_phys`, which is the same history.
4. **Metrics.**
   - Every validation metric (§1.5.2 step 4) on the test cases, except `rmse_by_date` and the split-free `cost_and_flags` rows, which are not repeated.
   - Backtest errors (§1.5.3 steps 4 to 6) on the test cases, with the backtest models of B and B2.
   - All rows have split `test`.
   - No consistency check against `predictions`, because Task 2 wrote no test rows.
5. **Test predictions.** For each candidate, `write_predictions(con, run_id=<train run>, variant, case_id, split='test', ..., allow_test=True)`.
   - If test rows of that run and variant exist (an override scoring), compare them instead of writing (`consistency_rel_tol`).
6. **Finish.** The predictions of step 5, `write_metrics` and `finish_test_scoring(ticket, 'ok')` form one transaction, which `finish_test_scoring` commits. Then `finish_run`.
   - `outputs` adds `freeze_id`, `scoring_id`, `tag_commit`, `override_reason`, `selection_run`, `evaluate_run`, `backtest_run`.
7. **Files.** The test figures (§1.8), then `write_model_comparison`.

### 1.6 Database

**Schema.** The new DDL is appended to `SCHEMA` in `db.py`, in this order, so that each trigger's tables exist when it is created. `init_schema` stays idempotent (`IF NOT EXISTS`). SQLite returns `SQLITE_CONSTRAINT` for `RAISE(ABORT, …)` (SQLite documentation, CREATE TRIGGER), which Python raises as `sqlite3.IntegrityError`. Test E15 checks this.

```sql
CREATE TABLE IF NOT EXISTS freeze (
  freeze_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id             TEXT NOT NULL REFERENCES runs(run_id),       -- the freeze run
  selection_run      TEXT NOT NULL REFERENCES runs(run_id),
  primary_id         TEXT NOT NULL,
  primary_model      TEXT NOT NULL,
  primary_variant    TEXT NOT NULL,
  secondary_id       TEXT,                                        -- NULL only if the selection found none
  secondary_model    TEXT,
  secondary_variant  TEXT,
  fallback_secondary INTEGER NOT NULL CHECK (fallback_secondary IN (0, 1)),
  tag                TEXT NOT NULL,
  tag_commit         TEXT NOT NULL,
  head_commit        TEXT NOT NULL,
  created            TEXT NOT NULL,                               -- UTC ISO 8601
  override_reason    TEXT CHECK (override_reason IS NULL OR length(trim(override_reason)) >= 10)
);
CREATE TRIGGER IF NOT EXISTS freeze_once BEFORE INSERT ON freeze
  WHEN NEW.override_reason IS NULL AND EXISTS (SELECT 1 FROM freeze)
  BEGIN SELECT RAISE(ABORT, 'V3: the proxy choice is already frozen; a new freeze needs an override reason'); END;
CREATE TRIGGER IF NOT EXISTS freeze_no_update BEFORE UPDATE ON freeze
  BEGIN SELECT RAISE(ABORT, 'V3: freeze rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS freeze_no_delete BEFORE DELETE ON freeze
  BEGIN SELECT RAISE(ABORT, 'V3: freeze rows are append-only'); END;

CREATE TABLE IF NOT EXISTS test_scoring (
  scoring_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          TEXT NOT NULL UNIQUE REFERENCES runs(run_id),    -- the evaluate-test run
  freeze_id       INTEGER NOT NULL REFERENCES freeze(freeze_id),
  tag_commit      TEXT NOT NULL,
  head_commit     TEXT NOT NULL,
  started         TEXT NOT NULL,
  finished        TEXT,
  status          TEXT NOT NULL CHECK (status IN ('running','ok','failed')),
  override_reason TEXT CHECK (override_reason IS NULL OR length(trim(override_reason)) >= 10)
);
CREATE TRIGGER IF NOT EXISTS test_scoring_once BEFORE INSERT ON test_scoring
  WHEN NEW.override_reason IS NULL AND EXISTS (SELECT 1 FROM test_scoring)
  BEGIN SELECT RAISE(ABORT, 'V3: the test split was already scored; a second scoring needs an override reason'); END;
CREATE TRIGGER IF NOT EXISTS test_scoring_final BEFORE UPDATE ON test_scoring
  WHEN OLD.status <> 'running' OR NEW.run_id <> OLD.run_id OR NEW.freeze_id <> OLD.freeze_id
  BEGIN SELECT RAISE(ABORT, 'V3: a finished test scoring is immutable'); END;
CREATE TRIGGER IF NOT EXISTS test_scoring_no_delete BEFORE DELETE ON test_scoring
  BEGIN SELECT RAISE(ABORT, 'V3: test_scoring rows are append-only'); END;

CREATE TABLE IF NOT EXISTS metrics (
  run_id   TEXT NOT NULL REFERENCES runs(run_id),     -- the evaluate or backtest run that computed the value
  proxy_id TEXT NOT NULL,                             -- '<train_run_id>/<variant>' of the candidate, or 'b2'
  split    TEXT NOT NULL CHECK (split IN ('train','val','test','none')),   -- 'none': cost and flags
  fluid    TEXT NOT NULL CHECK (fluid IN ('oil','gas','water','all')),
  metric   TEXT NOT NULL,
  idx      INTEGER NOT NULL DEFAULT -1,               -- horizon h (bt_rmse), date_idx (rmse_by_date), else -1
  value    REAL,                                      -- NULL = undefined (for example no event in both curves)
  PRIMARY KEY (run_id, proxy_id, split, fluid, metric, idx)
);
CREATE TRIGGER IF NOT EXISTS metrics_test_guard BEFORE INSERT ON metrics
  WHEN NEW.split = 'test' AND NOT EXISTS
       (SELECT 1 FROM test_scoring WHERE run_id = NEW.run_id AND status = 'running')
  BEGIN SELECT RAISE(ABORT, 'V3: test metrics need a running test_scoring claim of the same run'); END;
CREATE TRIGGER IF NOT EXISTS predictions_test_guard BEFORE INSERT ON predictions
  WHEN NEW.split = 'test' AND NOT EXISTS (SELECT 1 FROM test_scoring WHERE status = 'running')
  BEGIN SELECT RAISE(ABORT, 'V3: test predictions need a running test_scoring claim'); END;

CREATE TABLE IF NOT EXISTS selection (
  run_id     TEXT NOT NULL REFERENCES runs(run_id),   -- the select run
  proxy_id   TEXT NOT NULL,
  model      TEXT NOT NULL,
  variant    TEXT NOT NULL,
  strategy   TEXT NOT NULL,
  step1_oil INTEGER NOT NULL, step1_water INTEGER NOT NULL,     -- one column for each objective fluid
  step2_oil INTEGER NOT NULL, step2_water INTEGER NOT NULL,
  p1_pass    INTEGER NOT NULL,
  p2_pass    INTEGER NOT NULL,
  p3_flag    INTEGER,                                   -- NULL if R is undefined
  eligible   INTEGER NOT NULL,
  j_val      REAL NOT NULL,
  bt_score   REAL NOT NULL,
  rho        REAL,
  rank       INTEGER,                                   -- 1 = best eligible primary candidate; NULL otherwise
  role       TEXT CHECK (role IN ('primary','secondary') OR role IS NULL),
  PRIMARY KEY (run_id, proxy_id)
);
```

- The `selection` columns `step1_*` and `step2_*` follow `metrics.objective_fluids`. `load_eval_config` accepts only [oil, water] for this key. Changing it needs a schema change and a decision.
- The predictions trigger also applies to Task 2's `write_predictions(..., allow_test=True)`, as it should.

**Metric catalogue (`METRICS`).** `write_metrics` refuses a name, fluid, split or idx outside this table.

| Metric | Fluid | idx | Splits | Models | Section |
| --- | --- | --- | --- | --- | --- |
| `rmse`, `nrmse`, `r2_pooled`, `r2_case` | oil, gas, water | −1 | train, val, test | all | §2.2 |
| `j` | all | −1 | train, val, test | all | §2.2 |
| `n_clipped_pred` | all | −1 | train, val, test | all | §2.7 |
| `rmse_by_date` | oil, gas, water | date_idx | val | all | §2.2 (σ_proxy) |
| `ev_<event>_mae_days`, `ev_<event>_bias_days`, `ev_<event>_n_both`, `ev_<event>_n_mismatch`, `ev_<event>_n_none` | all | −1 | val, test | all | §2.3 |
| `rho_O`, `r2_O`, `local_dO_median` | all | −1 | val, test | all | §2.4 |
| `local_rmse` | oil, gas, water | −1 | val, test | all | §2.4 |
| `coverage`, `interval_width` | oil, gas, water | −1 | val, test | proxies | §2.5 |
| `p1_n_clipped` | all | −1 | val, test | proxies | §2.7 |
| `p2_n_rising`, `p2_max_ratio`, `p3_R`, `p3_n_used`, `p3_D_hist_true` | oil | −1 | val, test | proxies | §2.7 |
| `gor_pred_min`, `gor_pred_max`, `gor_flag` | gas | −1 | val, test | all | §2.7 |
| `gor_cont_min`, `gor_cont_max`, `gor_cont_flag` | gas | −1 | val, test | proxies | §2.7 |
| `cost_fit_s`, `cost_run_s`, `cost_predict_s` | all | −1 | none | all | §2.6 |
| `n_trials_max_epochs`, `n_members_max_epochs` | all | −1 | none | rnn, lstm | §2.7 |
| `bt_rmse` | oil, gas, water | h = 1..H | val, test | proxies, b2 | §2.8 |
| `bt_rmse_v6`, `bt_rmse_all` | oil, gas, water | −1 | val, test | proxies, b2 | §2.8 |
| `bt_score` | all | −1 | val, test | proxies, b2 | §2.8 |
| `bt_n_clipped` | all | −1 | val, test | proxies | §2.8 |
| `bt_v6_pass` | oil, water | −1 | val, test | proxies | §2.8 (1 or 0) |

Flags are stored as 1.0 and 0.0. Counts are stored as floats.

**Row counts on the real data (order of magnitude).**

- `evaluate --split val`: about 9 × 120 + 9 × 41 × 3 ≈ 2,200 rows.
- `backtest`: about 8 × (36 + 10) ≈ 400 rows.
- The test run has the same order as the two together, without `rmse_by_date`.

### 1.7 Reports

Both reports are rebuilt from the database (the single source, PRD §5). Numbers have 4 significant digits. RMSE is in STB/d (oil, water) or MSCF/d (gas). Labels are `<model>/<variant>`. A legend maps each label to its full proxy ID.

**Run chain.** The reports use one chain of runs:

- If a freeze row exists: the selection run of the active freeze row, its E and B runs, and the latest `test_scoring` row with status ok for that freeze.
- Otherwise: the latest ok `select` run (if any) and its E and B. With no select run, the latest ok E and B.

#### 1.7.1 `outputs/reports/model_comparison.md`

1. **Header.**
   - The run IDs of the chain, with their git commits.
   - Prepare run and `arrays_sha256`.
   - Config path and sha256.
   - k0, T, |K| (with the first and last dates), n_fc, t_cut, H, the case counts.
   - The freeze status: "not frozen", or freeze ID, tag and tag commit.
   - The test status: "test split not scored (V3)", or "scored once by run X at commit Y". With an override: every scoring with its reason.
2. **Table 1: all models and all splits** (PRD §7 outputs).
   - Rows: each of the 9 candidates × {train, val, test}.
   - Columns: J; NRMSE oil, water; RMSE oil, gas, water; R² pooled oil, water; R² case oil, water; ρ; local RMSE oil, water; coverage oil, water; BT RMSE h 9–12 oil, water; status.
   - Status: `primary`, `secondary`, `eligible`, or `rejected: <steps>` (PRD §10: a model that fails V6 is marked rejected).
   - Cells that do not apply are "—". Test cells are "not scored" until the test run exists.
3. **V6.** For each proxy and each objective fluid: RMSE_val against B0 and B1 (pass or fail), and the backtest RMSE (h 9–12) against B2 (pass or fail). Validation decides; test rows are report only.
4. **Backtest.** RMSE per horizon h = 1..H for each fluid, for each proxy and B2. Plus `bt_rmse_all`, `bt_score` and `bt_n_clipped`. Validation; test after the freeze.
   - States that the backtest settings saw the full-window validation data (task2 §2.13), and that 12 quarters do not test the 80-quarter forecast (PRD §9).
5. **Plausibility.** For each proxy:
   - P1: clipped values.
   - P2: rising cases out of n, and the maximum ratio.
   - P3: R, the cases used, the flag, and the median true D_hist.
   - GOR: the history and continuation ranges against the train range, with flags.
   - Then the D-020 statement (no new wells or development after 2008; the user confirmed this on 14 Oct 2026, D-031), and the D-022 statement on the DMD tail.
6. **Event errors.** For each model and event: n_both, n_mismatch, n_none, MAE (days), bias (days).
7. **Mismatch.**
   - |K|, α and β, and σ_obs (min and max for each fluid).
   - The O_sim of each case, sorted, with the local cases marked.
   - For each model: ρ, R²_O, local RMSE for each fluid, and the median O_prx − O_sim of the local cases.
8. **Coverage.** For each proxy and fluid: coverage and mean width. The flag `under` (< 0.90) or `over` (> 0.98). The D-021 note: no recalibration, because Task 4's σ_proxy and std filter do not depend on a global variance scale.
9. **Cost.** For each model: `cost_fit_s`, `cost_run_s`, `cost_predict_s` for `cost.n_rows` sets, and the Task 2 MCMC extrapolation (`runs.outputs.variants[v].speed`).
10. **Training flags (D-021 b).** For rnn and lstm: the trials that reached `max_epochs` (ID, variant, lr, epochs per seed), whether the selected trial is among them, and the final members that reached it.
11. **σ_proxy for Task 4.** For each proxy, the minimum, median and maximum over K of `rmse_by_date` (oil, water). The full values are in `metrics`.
12. **Clipping.** `n_clipped_pred` for each split.
13. **Figures.** Their relative paths.

#### 1.7.2 `outputs/reports/selection.md`

1. **Header.** Select run ID, git commit, E and B run IDs, config sha256, `p2_mode`, `band`, `secondary_only`.
2. **Step table.** One row for each proxy:
   - Step 1 (oil, water), step 2 (oil, water), P1, P2, P3 flag.
   - Eligible, J, BT, ρ, rank, role.
   - The reason for each failure, with the numbers (for example "water: 251.3 ≥ B1 220.0").
3. **Ranking trace.** j*, the band limit and T1. b*, its limit and T2. The ρ values. The chosen primary and the rule that decided it.
4. **Secondary.**
   - The strategy rule and the candidates with the other strategy.
   - The choice, or the D-020 fallback. With the fallback, the report states: "method uncertainty covers S3 only".
   - With DMD as secondary, the D-022 statement: Task 5 must state the DMD tail limitation (P3).
5. **Proposal.** Two lines, `PRIMARY: <proxy_id> (<label>, <strategy>)` and `SECONDARY: <proxy_id> (...)` or `SECONDARY: none`. Then the freeze commands of §0.1 step 4, filled in.
6. **Caveats.**
   - Validation J is optimistically biased (task2 §2.5).
   - The backtest settings were tuned on the full window (task2 §2.13).
   - ρ rests on n_val cases.
   - There are no new wells or development after 2008: the user confirmed it on 14 Oct 2026 (D-031). P2 and P3 rest on this.

### 1.8 Figures (`outputs/figures/task3/`, PNG, `report.dpi`)

| File | Content | Command |
| --- | --- | --- |
| `curves_val_oil.png`, `curves_val_gas.png`, `curves_val_water.png` | One panel for each validation case, 5 columns. Simulated curve (black), `RAMP_History` (grey dotted), and each proxy (one colour each). History solid, the n_fc continuation dashed after a vertical line at the last history date | `evaluate --split val` |
| `parity_val.png` | Rows = the 9 models, columns = 3 fluids. Predicted against simulated for k ≥ k0, the 1:1 line, and the RMSE and R² pooled in each panel | `evaluate --split val` |
| `backtest_val.png` | 3 panels (fluids). RMSE against horizon for each proxy, with B2 dashed black | `backtest` |
| `curves_test_{oil,gas,water}.png`, `parity_test.png`, `backtest_test.png` | The same for the test cases. The curve figures show the primary and the secondary only; parity and backtest show all models | `evaluate --split test` |

That gives 10 figures. Colours follow the candidate order in matplotlib's `tab10` cycle. `matplotlib.use("Agg")` is called before `pyplot` is imported. Each figure is closed after saving. Plotting never changes a metric.

### 1.9 V9 assertions at module boundaries

| Boundary | Assertion |
| --- | --- |
| `load_split` → | X (n, p): finite, in [0, 1]. Y_phys (n, T, 3): finite, ≥ 0. hist_phys (T, 3): finite, ≥ 0. Every case has the requested split. Case IDs strictly ascending. n ≥ 1 |
| metrics inputs | Equal shapes, finite (existing `_pair`). `std` finite and ≥ 0. Window mask of length T with at least one True |
| `mismatch` | Y_phys (n, T, 3) with T > max(k_idx), finite, ≥ 0. Output (n,): finite, ≥ 0. `sigma` > 0 everywhere, else `ContractError` at `build_mismatch_spec` |
| `continuation_checks` input | Rollout (n, T + n_fc, 3), finite, ≥ 0 |
| `backtest_errors` input | (n, H, 3) each, finite, ≥ 0. H ≥ `v6_horizons[1]` |
| `write_metrics` | Each row is in `METRICS`. Value finite or None. idx in range: 1..H for `bt_rmse`, 0..T−1 for `rmse_by_date`, else −1 |
| `select` input | One score for each proxy. j and bt finite and > 0 |

### 1.10 V3 guards (`evaluate --split test`, `freeze`, `select`)

| ID | Guard | Applies to | Override |
| --- | --- | --- | --- |
| G1 | The tag `freeze.tag` exists (`resolve_tag` is not None) | freeze, test | No |
| G2 | HEAD contains the tag commit (`is_ancestor(tag_commit, HEAD)`; a commit is its own ancestor) | freeze, test | No |
| G3 | The working tree is clean (`is_dirty()` is False) | freeze, test | No |
| G4 | No file under `freeze.frozen_paths` differs between the tag commit and HEAD | test | No |
| G5 | Freeze: the decision text at the tag holds both proxy IDs, and the lineage of §1.5.4 step 3 holds. Test: an active freeze row exists, and its `tag_commit` equals the current tag commit (the tag was not moved) | freeze, test | No |
| G6 | Once only. Freeze: no earlier `freeze` row. Test: no earlier `test_scoring` row, whatever its status (a failed attempt counts). Select: no `freeze` row. Enforced in Python (a clear message) and in SQLite (the triggers of §1.6) | freeze, test, select | `--override-reason` |
| G7 | The test run's resolved config blocks `metrics`, `events`, `mismatch`, `continuation`, `plausibility`, `backtest` and `selection` equal those of the runs E, B and S of the frozen chain (`json` equality of each block) | test | No |

- The claim row (`test_scoring`, status `running`) is committed before `load_split(con, 'test', ...)` is called. The triggers then admit test rows in `metrics` and `predictions` for that run only.
- **Override.**
  - Logged at WARNING in the run log.
  - Stored in `runs.outputs.override_reason` and in the `freeze` or `test_scoring` row.
  - Printed in the header of both reports.
  - Each override needs a researcher entry in `docs/decisions.md`. An override never bypasses G1 to G5 or G7.
- **Tests.** `GitOps` takes `repo_dir`, so the tests use a temporary repository. No test may create, move or delete a tag in the project repository (§4.5 checks `git tag -l freeze-proxy` before the merge).

## 2. Equations and methods

### 2.1 Notation

- i = case of the split (n cases), f = fluid, k = date index 0..T − 1.
- y_{f,i,k} is the simulated rate (physical), and ŷ_{f,i,k} the prediction (physical, clipped at 0; task2 §2.12).
- k0 = the onset index of the **train** cases (task2 §2.2), via `TrainRef`. W = {k : k0 ≤ k ≤ T − 1} is the producing window.
- K = the match window: the dates in [`mismatch.window[0]`, `mismatch.window[1]`], inclusive. On the real data K = W, 37 dates (k = 4..40).
- F_J = `metrics.objective_fluids` = {oil, water}. F_O = `mismatch.fluids` = {oil, water}.
- t_days = days from the first history date (Task 1).

### 2.2 Error metrics

$$
\mathrm{RMSE}_f = \sqrt{\frac{1}{nT}\sum_{i=1}^{n}\sum_{k=0}^{T-1}\big(\hat y_{f,i,k}-y_{f,i,k}\big)^2},\qquad
\mathrm{NRMSE}_f = \frac{\mathrm{RMSE}_f}{\max_{i,k} y_{f,i,k}-\min_{i,k} y_{f,i,k}},\qquad
J = \frac{1}{|F_J|}\sum_{f\in F_J}\mathrm{NRMSE}_f
$$

- Source: PRD §7. RMSE is also Article 6, Eq. 21 (p. 13). J is task2 §2.5 (D-015).
- All T dates, as in Task 2, so that J equals the tuning objective (§1.5.2 step 5). The pre-onset values are 0 in both curves (D-013), so they add no error to any model.

$$
R^2_{\mathrm{pooled},f} = 1-\frac{\sum_{i}\sum_{k\in W}(\hat y_{f,i,k}-y_{f,i,k})^2}{\sum_{i}\sum_{k\in W}(y_{f,i,k}-\bar y_f)^2},\qquad \bar y_f = \frac{1}{n|W|}\sum_i\sum_{k\in W} y_{f,i,k}
$$

- Source: PRD §7 and the form of Article 4, Eq. 16 (p. 1105).
- The window W is an assumption (D-024). The pre-onset zeros are fixed by construction, but they lie far from ȳ, so they inflate the denominator and the score.

$$
R^2_{\mathrm{case},f} = 1-\frac{\sum_{i}\sum_{k\in W}(\hat y_{f,i,k}-y_{f,i,k})^2}{\sum_{i}\sum_{k\in W}(y_{f,i,k}-m_{f,k})^2},\qquad m_{f,k}=\frac{1}{n_{\mathrm{train}}}\sum_{i\in\mathrm{train}} y_{f,i,k}
$$

- Source: PRD §7 (assumption in the PRD: pooled R² includes the time trend).
- m is the B0 curve, so R²_case = 1 − SSE ÷ SSE(B0). B0 therefore scores exactly 0 (test E02, and the real-data check of §4.4).
- A NaN (denominator 0) is stored as NULL.

**Per-date validation RMSE (σ_proxy for Task 4).**

$$
\hat\sigma_{f}(k) = \sqrt{\frac{1}{n_{\mathrm{val}}}\sum_{i\in\mathrm{val}}\big(\hat y_{f,i,k}-y_{f,i,k}\big)^2},\qquad k=0..T-1
$$

- Source: PRD §8 ("validation RMSE of the primary proxy at date t_k").
- Stored for every candidate as `rmse_by_date`, validation split only. Task 4 reads the frozen primary's values with `read_rmse_by_date` and slices K.

### 2.3 Event errors (PRD §7; D-024)

For each curve (true or predicted) of case i:

| Event | Index | Source |
| --- | --- | --- |
| Plateau end | Let k_max = the first argmax of q_oil over k ≥ k0. Then k_pe = the first k > k_max with q_oil(k) < `plateau_frac` × q_oil(k_max). None if no such k, or if max q_oil ≤ 0 | PRD §7 (0.98). The search from k_max is an assumption: the PRD rule read literally gives k = 0 (rate 0 before onset), and from k0 a ramp-up quarter below 0.98 × max would end the plateau before it starts |
| Water breakthrough | k_bt = the first k ≥ k0 with q_w(k) > `breakthrough_frac` × max_k q_w(k). None if max q_w ≤ 0 | PRD §7 (0.01). Same rule as task2 §2.10 |
| Water-oil crossover | k_x = the first k ≥ k0 with q_w(k) > q_oil(k). None if no such k | PRD §7 |

- Each curve uses its own maximum (PRD wording).
- Error of case i: e_i = t_days[k_pred,i] − t_days[k_true,i], in days. A positive error means the prediction is late.

| Statistic | Definition |
| --- | --- |
| `n_both` | Cases with the event in both curves |
| `mae_days`, `bias_days` | Mean \|e_i\| and mean e_i over `n_both`. NULL if `n_both` = 0 |
| `n_mismatch` | Cases with the event in exactly one curve. Counted, not scored, because the true or the predicted date lies beyond the window |
| `n_none` | Cases with the event in neither curve |

The resolution is one quarter (90 to 92 days), because the rates are quarter means (D-004).

### 2.4 Mismatch, rank correlation and local accuracy (`hm/objective.py`; D-025)

**Mismatch O(x).** Source: PRD §8. The ½ and the series weights of Article 4, Eq. 18 (p. 1109) are absorbed: O = 2 × OF with ω = 1.

$$
O(\hat y) = \sum_{f\in F_O}\sum_{k\in K}\left(\frac{\hat y_f(t_k)-d_f(t_k)}{\sigma_f(t_k)}\right)^2,\qquad
\sigma_f^2=\sigma_{\mathrm{obs},f}^2+\sigma_{\mathrm{proxy},f}^2,\qquad
\sigma_{\mathrm{obs},f}(t_k)=\max\!\big(\alpha\, d_f(t_k),\ \beta \max_{k'\in K} d_f(t_{k'})\big)
$$

- **d.** d_f(t_k) = `hist_phys[k, f]`: the R1 history, with the same alignment and rate definition as the cases (V4, D-004). Never the `*_rate_reported` columns.
- **α.** α = 0.10: source Article 4, p. 1111 (Punq, max(0.1 y_ref, 0.005)) and p. 1113–1114 (Brugge, 10% with a floor of 50).
- **β.** β = 0.02, relative to the maximum over K: assumption (PRD §8). Article 4's floors are absolute.
- **Sum, not mean** (PRD §8).
- **Errors.** `build_mismatch_spec` raises `ContractError` if K is empty, if a fluid has max_K d_f = 0 (σ would be 0), or if `sigma_proxy` is not (|K|, F_O), finite and ≥ 0.
- **O_obs.** O with σ_proxy = 0 (PRD §8).

**Task 3 uses O_obs for every model.** Assumption (D-025), with three reasons:

1. All models are compared on one fixed scale.
2. σ_proxy is an output of Task 3, so it cannot enter the choice that it depends on.
3. A σ_proxy for each model would weight dates and fluids differently for each model and change the ranks.

**Per split.** O_sim,i = O_obs(y_i), the simulated curve of case i. O_prx,i = O_obs(ŷ(x_i)), the model's curve for the same X.

**Mismatch rank ρ.**

$$
\rho = \operatorname{corr}\big(\operatorname{rank}(O_{\mathrm{prx}}),\ \operatorname{rank}(O_{\mathrm{sim}})\big)
$$

- The Pearson correlation of the average ranks: Spearman's ρ. Ties take the mean rank (`scipy.stats.rankdata`, method `average`; Spearman 1904).
- None if either vector is constant (B0 gives one curve for all cases, so its ρ is NULL) or if n < 3.
- Source: assumption, motivated by Article 6 (p. 1, p. 16–18, Table 9 p. 30): global accuracy does not show accuracy where the search goes.
- Use: a tie-break only (§2.9). With 15 cases, the standard error of ρ near 0 is about 1 ÷ √14 ≈ 0.27, so it cannot decide alone.

**R²_O** (report only). The R² of Article 4's Eq. 16 applied to the objective function, as Article 4 does (p. 1114): R²_O = 1 − Σ_i (O_prx,i − O_sim,i)² ÷ Σ_i (O_sim,i − mean O_sim)². It is dominated by the cases with large O, so it is reported but not used.

**Local RMSE.** L = the `local_n` = 5 cases of the split with the lowest O_sim. Ties go to the lower case ID.

$$
\mathrm{RMSE}^{\mathrm{local}}_f = \sqrt{\frac{1}{|L|\,|K|}\sum_{i\in L}\sum_{k\in K}(\hat y_{f,i,k}-y_{f,i,k})^2}
$$

- Over K, the dates that enter O. All three fluids are reported.
- `local_dO_median` = the median over L of O_prx,i − O_sim,i: the proxy's mismatch bias near the history. It is the analogue of Article 6's "Cal." against "Actual" NPV at the proxy optimum (Table 9, p. 30).
- Source: PRD §7 (n = 5). The definition is an assumption, motivated by Article 6.
- Report only. It is not in the selection rule (PRD §7).

### 2.5 Interval coverage (PRD §7; D-021)

$$
\mathrm{cov}_f = \frac{1}{n|W|}\sum_i\sum_{k\in W}\mathbf 1\big\{|y_{f,i,k}-\hat y_{f,i,k}|\le z\, s_{f,i,k}\big\},\qquad \mathrm{width}_f = \frac{2z}{n|W|}\sum_i\sum_{k\in W}s_{f,i,k}
$$

- s = `predict_std` × y_factor (physical). z = `coverage_z` = 1.959964 (two-sided 95%, normal). Proxies only; benchmarks are "—".
- **Window W.** Before k0, s = 0 and the error is 0, so those values would count as covered and inflate the share. Assumption, as in the Task 2 preview.
- **Flags.** `under` if cov < 0.90, `over` if cov > 0.98 (PRD target 0.90 to 0.98).
- **No recalibration (D-021).** The seed spread captures only the optimisation spread, not the model error (Lakshminarayanan et al. 2017).

### 2.6 Cost (PRD §7; Article 2, p. 993)

| Metric | Definition |
| --- | --- |
| `cost_fit_s` | Time to fit the final model. rnn and lstm: the sum of `fit_s` over the members in `meta.json` (`members`). dmd: `fit_s` of the selected trial in `trials.csv` (`best_trial` from `runs.outputs.variants.main`). b0, b1: `runs.duration_s` |
| `cost_run_s` | `runs.duration_s` of the train run: tuning, final fit and diagnostics. Shared by the variants of one run |
| `cost_predict_s` | The median wall time of `cost.repeats` calls of `predict` on `cost.n_rows` rows, after one untimed warm-up call on the same rows. X = `np.random.default_rng(seed).random((n_rows, p))` |

The report adds the Task 2 MCMC extrapolation (`speed.mcmc_extrapolated_s`, against 600 s). Timings vary between runs, so the determinism test (E25) excludes them.

### 2.7 Plausibility checks (D-020, D-021, D-022)

**Inputs.**

- R = `model.rollout(X, T + n_fc)` × y_factor: the history and the continuation, on the cases of the split. Proxies only.
- `p1_n_clipped` = the clipped count of the steps T..T + n_fc − 1: the model's `last_n_clipped` right after `forecast(X, n_fc)` (task2 §2.16).
- `n_clipped_pred` = `last_n_clipped` of `predict(X)` (history window, all models).

**P1 (D-020).**

- The proxy fails if `p1_n_clipped` > 0, over all fluids and continuation steps.
- If it fails, the proxy is not eligible as primary or as secondary.
- Reason: a rate below 0 is impossible, and validation J cannot see the continuation.

**P2 (D-020).**

- Let q̂_i = R[i, T − 1, oil], the proxy's own rate at the last history date.
- Case i rises if max over k = T..T + n_fc − 1 of R[i, k, oil] > `p2_ratio` × q̂_i. If q̂_i = 0, the case rises if any continuation oil is > 0.
- `p2_n_rising` = the number of rising cases. `p2_max_ratio` = the maximum of R[i, k, oil] ÷ q̂_i over the cases with q̂_i > 0.
- The proxy fails P2 if `p2_n_rising` > `p2_case_share` × n (strictly more than half).
- `p2_mode: reject`: a fail makes the proxy ineligible as primary and as secondary.
- `p2_mode: report`: report only. It would apply if there were new wells or development after 2008. The user confirmed on 14 Oct 2026 that there are none (D-031), so the deliverable keeps `reject`.

**P3 (D-022; report only).**

1. **Declines.** With L_h = `p3_hist_quarters` = 8 and [a, b] = `p3_fc_quarters` = [9, 16]:

$$
D_{\mathrm{hist},i} = \frac{\ln q_i(T-1-L_h) - \ln q_i(T-1)}{L_h},\qquad
D_{\mathrm{fc},i} = \frac{\ln q_i(T-2+a) - \ln q_i(T-1+b)}{b-a+1}
$$

   - q_i = R[i, ·, oil], the proxy's own curve.
   - Forecast quarter j is the rollout step T − 1 + j (quarter 1 = step T).
   - D_hist covers the L_h quarter-to-quarter declines into the last L_h history quarters: steps T − 1 − L_h → T − 1.
   - D_fc covers the b − a + 1 declines into forecast quarters a to b: steps T − 2 + a → T − 1 + b. With [9, 16] these are the steps T + 7 → T + 15, 8 declines. On the real data that is 2010-Q1 to 2011-Q4 (D-004: the value dated 2010-04-01 is the 2010-Q1 mean).
   - D is the mean log decline per quarter, the "mean relative decline" of D-022.
2. **Excluded cases.** A case is excluded if any of the four rates is ≤ 0, or if D_hist,i < `p3_min_decline` (0.002 per quarter; a curve that does not decline in history has no decline to continue).
3. **Ratio.** R_P3 = the median over the used cases of D_fc,i ÷ D_hist,i. `p3_n_used` = the number of used cases.
   - R_P3 is None (NULL) if no case is used.
   - Flag if R_P3 < `p3_ratio_min` = 0.5.
4. **Reference.** `p3_D_hist_true` = the median D_hist of the simulated curves of the same cases.
5. **Basis (D-022).** Under Arps decline with b ≤ 1, D falls to no less than about 1 ÷ (1 + b D_i t) ≈ 0.64 of its value over 16 quarters at the observed 2 to 3.5% per quarter (Arps 1945).
6. **Use.** The flag never changes eligibility. With DMD as secondary, it carries the Task 5 statement.

**GOR (D-010, D-021 c; report only).**

- **Training range.** [g_min, g_max] of q_gas ÷ q_oil over the train cases and dates with q_oil > 0 (`TrainRef.gor_range`).
- **Predicted range.** The minimum and maximum of q̂_gas ÷ q̂_oil where q̂_oil > `gor_oil_frac` × max q̂_oil. History predictions for all models; R's continuation steps for proxies.
- **Flag.** pred_min < g_min (1 − tol), or pred_max > g_max (1 + tol), with tol = `gor_rel_tol` = 0.005.
- Task 2 found rnn/A at 0.3661 against g_max 0.3654 (0.19%), which is within the tolerance.

**Max-epoch flags (D-021 b; report only).**

- From `trials.csv` (`n_epochs`, `;`-joined per seed) and `config.yaml` (`training.max_epochs`) of each neural run.
- A trial is flagged if any member has n_epochs ≥ max_epochs. A final member is flagged if its `meta.json` n_epochs ≥ max_epochs.
- Stored as `n_trials_max_epochs` (per run, written under each variant) and `n_members_max_epochs` (per variant). The report lists the flagged trials and whether the selected trial is among them.

### 2.8 Backtest (PRD §9; D-026)

**Window.**

- t_cut = #{k : dates[k] ≤ cutoff_date}. The dates k < t_cut are fitted.
- H = T − t_cut, and the horizon of date k ≥ t_cut is h = k − t_cut + 1.
- Real data: the cutoff 2005-01-01 is k = 28, so t_cut = 29 and H = 12. Y[:, 28] is the 2004-Q4 mean (D-004), so the fit sees production through 31 December 2004.

**Models.**

- Each candidate proxy is retrained with `train --t-cut t_cut --settings-from <candidate run>`: the Task 2 settings, no tuning (PRD §9 step 2; task2 §2.13).
- Each variant (A, B, C; `main`) is continued with its own `forecast` (F1 or F2, PRD §9 step 3).
- No refit and no other method (V13).

**B2 persistence** (PRD §9 step 5). Source: PRD §9.

$$
\hat y^{\,B2}_{f,i,t_{\mathrm{cut}}-1+h} = y_{f,i,t_{\mathrm{cut}}-1},\qquad h=1..H
$$

B2 knows each case's own simulated rate at the cutoff, while the proxies know only X. This is the usual information advantage of persistence.

**Errors.** For the n cases of the split:

$$
\mathrm{RMSE}^{\mathrm{bt}}_f(h)=\sqrt{\tfrac{1}{n}\textstyle\sum_i(\hat y_{f,i,t_{\mathrm{cut}}-1+h}-y_{f,i,t_{\mathrm{cut}}-1+h})^2},\qquad
\mathrm{RMSE}^{\mathrm{bt}}_f(\mathcal H)=\sqrt{\tfrac{1}{n|\mathcal H|}\textstyle\sum_i\sum_{h\in\mathcal H}(\cdot)^2}
$$

| Metric | Horizon set |
| --- | --- |
| `bt_rmse` | Each h = 1..H |
| `bt_rmse_v6` | 𝓗 = [`v6_horizons[0]`, `v6_horizons[1]`] = {9, 10, 11, 12} |
| `bt_rmse_all` | 𝓗 = {1..H} |

**V6 rule for B2** (assumption, D-026). For each f ∈ F_J:

`bt_v6_pass[f]` = RMSE^bt_f(9..12) < RMSE^B2_f(9..12), strictly.

- Step 2 of the selection passes if both fluids pass.
- Reasons for pooling: PRD §9 gives no pooling rule. With 15 cases, four separate horizon tests per fluid are noisy. Oil and water follow PRD §7 step 1, and gas follows oil (D-010). Per-horizon wins are reported.

**Backtest score** (tie-break, D-026):

$$
BT = \frac{1}{|F_J|}\sum_{f\in F_J}\frac{\mathrm{RMSE}^{\mathrm{bt}}_f(1..H)}{\max_{i,k} y_{f,i,k}-\min_{i,k} y_{f,i,k}}
$$

The denominator is the range of the split's true values over all T dates, the same as in NRMSE.

**Splits.**

- Validation cases for the choice (`backtest`).
- Test cases once, after the freeze (`evaluate --split test`), with the same backtest models (PRD §9 step 6).

### 2.9 Selection algorithm (PRD §7 steps 1 to 5; D-020, D-022, D-027)

Inputs: the validation metrics of E and B for the proxies, B0, B1 and B2.

1. **Step 1 (V6, B0 and B1).** For each f ∈ F_J: RMSE_val,f(m) < min(RMSE_val,f(B0), RMSE_val,f(B1)). Both fluids must pass.
2. **Step 2 (V6, B2).** `bt_v6_pass` for both fluids (§2.8).
3. **P1 and P2** (§2.7). P2 counts only under `p2_mode: reject`. P3 is computed and shown, never used.
4. **Eligible** = steps 1 and 2, P1, and P2 (reject mode).
5. **Ranking** (`rank_candidates(S, band)`) for a set S of eligible proxies:
   1. j* = min J over S. T1 = {m ∈ S : J(m) ≤ (1 + band) j*}.
   2. b* = min BT over T1. T2 = {m ∈ T1 : BT(m) ≤ (1 + band) b*}.
   3. Order T2 by ρ descending (None after every number), then J ascending, then BT ascending, then proxy ID ascending (string order). The first is the best.
   4. Remove the best, and repeat on the rest for the full order.
   - Band = `selection.band` = 0.05, relative to the best of the current set (PRD §7 step 4: "within 5%").
6. **Primary** = the first of `rank_candidates({eligible m : model ∉ secondary_only})`. With `secondary_only: [dmd]`, DMD cannot be primary (D-022).
   - If the set is empty, `status = no_eligible_primary`: no primary and no secondary. `freeze` refuses, and the researcher decides.
7. **Secondary** (PRD §7 step 5). Let s = strategy(primary).
   - The secondary = the first of `rank_candidates({eligible m : strategy(m) ≠ s})`. `secondary_only` models are allowed here.
   - **D-020 fallback.** If that set is empty and s = S3, the secondary = the first of `rank_candidates({eligible m : strategy(m) = S3, model(m) ≠ model(primary)})`. Set `fallback_secondary = true`. The reports state that the method uncertainty covers S3 only.
   - If that set is empty too, there is no secondary and `status = no_secondary`.
8. **Determinism.** The result does not depend on the input order (test E11 j).

### 2.10 Freeze and the test scoring (PRD §7 steps 6 and 7; D-028)

- **Freeze.** The freeze is the commit tagged `freeze-proxy`. That commit holds the decision text with both proxy IDs. The `freeze` row records the tag commit and the choice.
- **Test scoring.**
  - `evaluate --split test` scores cases 86 to 100 once, for all candidates, with the metric code and config of the tag (G4, G7).
  - It writes the test metrics, the test predictions and the test figures.
  - After this the choice does not change (G6 blocks a new `select` and a new `freeze`).
- **What the test split may not do.** It never enters a fit, a selection, σ_proxy or a threshold (V1, V3). Task 4's parameter-recovery test (PRD §8) uses test cases as pseudo-histories after the freeze. That access is specified in the Task 4 spec, through the same guard module.

## 3. Defaults and tuning ranges

Nothing in Task 3 is tuned. `load_eval_config` enforces the "Allowed" column. A value outside it needs an entry in `docs/decisions.md`.

### 3.1 Settings

| Key | Default | Allowed | Source |
| --- | --- | --- | --- |
| `seed` | 0 | Integer in [0, 2³² − 1] | V7 |
| `candidates.runs.<model>` | `latest` for all five models | `latest` or a run ID. b0 and b1 are required, plus at least one proxy model. The deliverable lists all five | §1.5.1 |
| `candidates.require_clean` | true | true, false (false only in tests; otherwise it needs a decision) | V8 |
| `strategy` | rnn S3, lstm S3, dmd S4 | Keys = exactly the proxy models in `candidates.runs`; values S3 or S4 | PRD §6 |
| `runtime.torch_threads` | 1 | 1 to 16 | task2 §3.2 |
| `consistency_rel_tol` | 1e-7 | 0 to 1e-4 | Assumption: float32 networks on one machine, deterministic algorithms |
| `metrics.objective_fluids` | [oil, water] | Exactly [oil, water] (schema columns) | PRD §7 |
| `metrics.coverage_z` | 1.959963984540054 | > 0 | Standard normal quantile |
| `metrics.coverage_target` | [0.90, 0.98] | 0 < lo < hi ≤ 1 | PRD §7 |
| `events.plateau_frac` | 0.98 | 0.5 to 1 | PRD §7 |
| `events.breakthrough_frac` | 0.01 | 0 to 0.5 | PRD §7 |
| `mismatch.window` | [1999-01-01, 2008-01-01] | ISO dates, start ≤ end; at least one history date inside | PRD §8 |
| `mismatch.fluids` | [oil, water] | Non-empty subset of the fluids | PRD §8 (gas out, D-010) |
| `mismatch.alpha` | 0.10 | 0.01 to 0.5 | Article 4, p. 1111, p. 1113–1114 |
| `mismatch.beta` | 0.02 | 0.001 to 0.2 | PRD §8 (assumption) |
| `mismatch.local_n` | 5 | 1 to n of the split (checked at run time) | PRD §7 |
| `continuation.end_date` | 2028-01-01 | ISO date, a whole number of quarters after the last history date | PRD §1 |
| `plausibility.p2_mode` | reject | reject, report | D-020 |
| `plausibility.p2_ratio` | 1.05 | 1.0 to 1.5 | D-020 |
| `plausibility.p2_case_share` | 0.5 | 0 to 1 | D-020 |
| `plausibility.p3_hist_quarters` | 8 | 1 to T − k0 − 1 (checked at run time) | D-022 |
| `plausibility.p3_fc_quarters` | [9, 16] | 1 ≤ a < b ≤ n_fc | D-022 |
| `plausibility.p3_ratio_min` | 0.5 | 0 to 1 | D-022 |
| `plausibility.p3_min_decline` | 0.002 | 0 to 0.05 | Assumption: 0.2% per quarter is about a tenth of the observed decline |
| `plausibility.gor_rel_tol` | 0.005 | 0 to 0.05 | D-021 (c) |
| `plausibility.gor_oil_frac` | 0.01 | 0 to 0.5 | Task 2 report (§1.7 item 9) |
| `backtest.cutoff_date` | 2005-01-01 | ISO date; k0 + 3 ≤ t_cut and H ≥ `v6_horizons[1]` | PRD §9 |
| `backtest.models` | [rnn, lstm, dmd] | Exactly the proxy models in `candidates.runs` (each one must be backtested, V6) | PRD §9 |
| `backtest.model_configs` | configs/models/<m>.yaml | One path for each model in `backtest.models` | §1.5.3 |
| `backtest.v6_horizons` | [9, 12] | 1 ≤ lo ≤ hi ≤ H | PRD §9 |
| `backtest.reuse` | {} | model → run ID | §1.5.3 |
| `selection.band` | 0.05 | 0 to 0.2 | PRD §7 |
| `selection.secondary_only` | [dmd] | Subset of PROXIES | D-022 |
| `cost.n_rows` | 10000 | ≥ 1 | PRD §7 |
| `cost.repeats` | 3 | 1 to 20 | Assumption: the median of 3 resists one slow call |
| `freeze.tag` | freeze-proxy | Non-empty | PRD §7 |
| `freeze.repo_dir` | null | null or a path | §1.10 |
| `freeze.decisions_path` | docs/decisions.md | Relative path | PRD §7 |
| `freeze.frozen_paths` | See §1.2 | Non-empty list of relative paths | D-028 |
| `freeze.min_override_chars` | 10 | 10 to 200 | D-028 |
| `report.plots` | true | true, false (false only in tests) | PRD §7 |
| `report.dpi` | 120 | 50 to 300 | Assumption |

### 3.2 Packages (V7)

| Package | Version | Group | Status |
| --- | --- | --- | --- |
| matplotlib | 3.10.8 | New extra `eval` in `pyproject.toml`; regenerate `requirements-lock.txt` (with contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing) | Not in `.venv`. Released 10 Dec 2025; a cp313 win_amd64 wheel exists (PyPI). Add `"matplotlib": "matplotlib"` to `PACKAGES` |
| scipy | 1.17.1 | main | Already pinned. `scipy.stats.rankdata` (method `average`) gives the ranks of ρ |
| numpy, torch, scikit-learn, pandas, pyyaml | As pinned | Unchanged | — |

The README install line becomes `pip install -e ".[dev,models,eval]"`.

## 4. Acceptance tests

`pytest` must pass (V10). The new fast tests must finish in under 90 s on the laptop; the data scientist reports the time. There is one test file for each new module.

No test is marked `slow` by default. The pipeline tests train only the tiny Task 2 configs. If E22 or E24 takes more than 20 s, mark it `slow` and report it. E24 must still run before the merge (`pytest -m slow`).

### 4.0 Fixtures (`tests/synth_eval.py`, `tests/conftest.py`)

- **`tiny_task2_db(tmp_path_factory)`** (module scope):
  - `synthetic_task1_db` (task2 §4.0): n = 30, T = 13 quarterly dates from 2000-01-01, k0 = 2, train 1 to 18, validation 19 to 24, test 25 to 30.
  - Then `run_train` for b0, b1, dmd and lstm with `tiny_cfg`. Returns the database path, `models_dir` and the run IDs.
- **`tiny_eval_cfg(tmp_path, ...)`:**
  - Explicit candidate IDs for b0, b1, lstm and dmd (rnn is left out to save time; E11 covers the rnn fallback), `strategy` {lstm: S3, dmd: S4}, `require_clean: false`.
  - `mismatch.window` [2000-07-01, 2003-01-01] (k = 2..12), `local_n` 2.
  - `continuation.end_date` 2005-01-01 (n_fc = 8).
  - `p3_hist_quarters` 4, `p3_fc_quarters` [3, 6].
  - `backtest.cutoff_date` 2002-01-01 (t_cut = 9, H = 4), `v6_horizons` [3, 4], `models` [lstm, dmd].
  - `cost.n_rows` 50, `repeats` 1, `report.plots` false, `freeze.repo_dir` a temporary repository.
- **`tmp_git_repo(tmp_path)`:**
  - `git init`, with user name and email set by `-c` options.
  - `docs/decisions.md`, a placeholder file under each `frozen_paths` entry, and one commit.
  - Skipped (`pytest.skip`) if `git` is not on PATH.
- **`FakeGit`:** an in-memory `GitOps` with settable tag, head, ancestry, dirty flag, changed paths and file contents.

### 4.1 Unit tests

| ID | File | Test | Pass criterion |
| --- | --- | --- | --- |
| E01 | `test_metrics.py` | J identity | `evaluate_models` J equals `objective_j` on the same arrays in scaled units (1e-12). The task2 M09 example still passes |
| E02 | `test_metrics.py` | R² by hand. 2 cases, 3 dates, k0 = 1, oil true [[0, 10, 8], [0, 12, 6]], predicted [[0, 11, 8], [0, 12, 5]], train mean curve [0, 10, 7] | R²_pooled oil = 1 − 2 ÷ 20 = 0.9 (over W = k 1..2, ȳ = 9). R²_case oil = 1 − 2 ÷ 6 = 2/3. Predicting the train mean curve gives R²_case = 0 exactly. A perfect prediction gives 1. A zero denominator gives NaN, stored as NULL |
| E03 | `test_metrics.py` | rmse_by_date and rmse_pooled | Hand values on a (2, 3, 3) example (1e-12). The mean over dates of rmse_by_date² equals rmse² |
| E04 | `test_metrics.py` | Coverage | Errors [1, 3] with s [1, 1] and z 2 give 0.5. Values with k < k0 are excluded. The width equals 2 z mean s |
| E05 | `test_metrics.py` | Spearman | Equals `scipy.stats.spearmanr(a, b).statistic` on 50 seeded draws with ties (1e-12). Constant input → None. n = 2 → None |
| E06 | `test_events.py` | Events | Hand curves: (a) a ramp at k0 (0.95 × max), then a plateau, then a decline: plateau end = the first date after the maximum below 0.98 × max, not k0. (b) Breakthrough at a known k. (c) A crossover at a known k, and a curve with none. (d) Error in days from `t_days`, with the right sign. (e) The event in one curve only counts in `n_mismatch`, and n_both = 0 gives NULL MAE |
| E07 | `test_objective.py` | O(x) (PRD §8 tests) | O = 0 when ŷ = d on K. A 3-date hand example with α = 0.1 and β = 0.02, including a water date where the floor applies, matches within 1e-12. Doubling K with identical residuals doubles O (sum, not mean). σ_proxy adds in quadrature. `window_indices` is inclusive. max_K d_f = 0 raises `ContractError` |
| E08 | `test_objective.py` | d is R1 | In a fixture where `history.*_rate_reported` differs from the R1 rates, the spec's d equals `hist_phys[K]` (R1) and not the reported values |
| E09 | `test_metrics.py` | Local set | O_sim = [5, 1, 3, 1, 9] with case IDs [3, 4, 5, 6, 7] and local_n 2 selects cases 4 and 6. Local RMSE by hand over K. `local_dO_median` by hand |
| E10 | `test_plausibility.py` | P1, P2, P3 on crafted rollouts (a fake model) | One negative raw value in the continuation → P1 fails, and a negative in the history only does not. 8 of 15 cases rise → P2 fails, 7 of 15 pass. A ratio of 1.04 does not count, and q̂ = 0 with later oil > 0 counts. P3: oil = exp(−0.03 k) up to T − 1, then q(T − 1) exp(−0.01 (k − T + 1)), gives D_hist = 0.03, D_fc = 0.01, R = 1/3 (1e-12) and the flag. A flat history case (D_hist < 0.002) is excluded, and all excluded gives R NULL |
| E11 | `test_selection.py` | Algorithm, table-driven with fake `CandidateScore`s | (a) Step 1 and step 2 failures are not eligible. (b) P1 fail is not eligible. (c) P2 fail is not eligible under `reject` and eligible under `report`. (d) J = [1.00, 1.04, 1.06] with BT = [2.0, 1.0, 0.5]: the second wins (BT inside the band; the third is outside the J band). (e) J tie, BT within 5%: the higher ρ wins, and ρ None loses to any number. (f) The secondary has the other strategy, and DMD is never primary (`secondary_only`). (g) No eligible S4 and an lstm primary: the secondary is the best eligible rnn, and `fallback_secondary` is true. (h) No candidate for the secondary: `no_secondary`. (i) Nothing eligible: `no_eligible_primary`. (j) Ten random permutations of the input give the same `Selection` |
| E12 | `test_backtest.py` | Window and persistence | `cutoff_index` on the fixture dates: 2002-01-01 → 9, and 2001-12-31 → 8. t_cut < k0 + 3 or H < `v6_horizons[1]` → `ConfigError`. `persistence` repeats y[:, t_cut − 1] for all H steps. `backtest_errors` matches hand RMSE per h, pooled h 3..4, pooled 1..H and BT (1e-12) |
| E13 | `test_dates.py` | Quarters | `quarters_between(2008-01-01, 2028-01-01)` = 80. A non-quarter end date → `ContractError`. `forecast_dates(2008-01-01, 3)` = [2008-04-01, 2008-07-01, 2008-10-01] |
| E14 | `test_plausibility.py` | GOR and max-epoch flags | The GOR flag at 1.004 × g_max with tol 0.005 is off, and at 1.006 × g_max on. A `trials.csv` with n_epochs "1000;530" and max_epochs 1000 is flagged, and "999;530" is not |
| E15 | `test_store.py` | Schema and triggers | `init_schema` creates the four tables and eight triggers, and is idempotent. A test-split `metrics` row without a claim → `sqlite3.IntegrityError`. A test `predictions` row without a running claim → `sqlite3.IntegrityError`. A second `freeze` or `test_scoring` row without an override → `sqlite3.IntegrityError`; with a reason of ≥ 10 characters it is accepted, and a reason of 5 characters fails the CHECK. UPDATE or DELETE of a `freeze` row, and DELETE of a `test_scoring` row, fail. `write_metrics` refuses an unknown metric, fluid or idx, and a NaN value |
| E16 | `test_store.py` | Store helpers | `resolve_candidates` with `latest` picks the latest ok full-window run with the current hash. It skips a later run with `t_cut` set, one with status `failed`, and one with another hash. With `require_clean`, a dirty run → `ContractError`. `read_rmse_by_date` round-trips a (T, 3) array |
| E17 | `test_guards.py` | Git guards with `FakeGit` | Each of these gives `GuardError` with the right guard ID: no tag (G1); tag not an ancestor of HEAD (G2); dirty tree (G3); a changed frozen path (G4); a freeze row whose tag commit differs from the tag (G5); a config block that differs from E (G7); a second claim without an override (G6). All conditions met → a `TestTicket`, and a `test_scoring` row with status `running` |
| E18 | `test_guards.py` | Real git (`tmp_git_repo`) | `GitOps(repo_dir)`: `resolve_tag` of an annotated tag returns the commit. `is_ancestor(c, c)` is True. After one more commit, `is_ancestor(tag, HEAD)` is True and the reverse False. `changed_paths` lists a changed frozen file. A modified tracked file makes `is_dirty` True. A path that is not a repository → `GuardError` |
| E19 | `test_guards.py` | `load_split` access | `load_split(con, 'test')` without a ticket, with a forged ticket (no claim row), or with a finished claim → `GuardError`, and no test case is returned. `load_split(con, 'val')` holds only validation cases |

### 4.2 Pipeline tests (`tiny_task2_db`)

| ID | File | Test | Pass criterion |
| --- | --- | --- | --- |
| E20 | `test_evaluate.py` | V3 spy for the validation path | Wrappers on `predict`, `predict_std`, `rollout` and `forecast` record every X during `evaluate --split val` and `backtest`. No recorded row equals the X of a test case. No `metrics` row has split `test`. `read_task1_arrays` is called, but no `SplitArrays` with split `test` is built |
| E21 | `test_evaluate.py` | `evaluate --split val` end to end (`cli.main`) | Exit 0. One `runs` row (`task3`, `evaluate`, status ok, `outputs.split` = val, every §1.5.2 key). For each candidate, every catalogue metric of the train and val splits that applies to it exists. `rmse_by_date` has T rows for each fluid. J equals the train run's `j_val` (1e-12). B0's `r2_case` is 0 for every fluid (1e-12), and its `rho_O` is NULL. `model_comparison.md` has the 13 sections of §1.7.1, with "not scored" in the test cells |
| E22 | `test_backtest.py` | `backtest` end to end | Exit 0. New train runs exist with `t_cut` 9 and `settings_from` = the candidate run, and each loaded backtest model has `T_fit` 9. `bt_rmse` has H = 4 rows for each fluid, for each variant and for `b2`. `mapping` covers every lstm variant and `dmd/main`. With `reuse` set to those runs, a second backtest trains nothing and gives equal metrics. A model config that differs from the source run (for example `patience`) → `ConfigError` before any training |
| E23 | `test_selection.py` | `select` end to end | Exit 0. `selection` has one row for each proxy, with at most one primary and one secondary. `selection.md` has the 6 parts of §1.7.2, the `PRIMARY:` line, and the freeze command. With a `freeze` row in the database, `select` exits 3, and with an override reason it exits 0 and logs the reason |
| E24 | `test_freeze_test.py` | Full V3 sequence in `tmp_git_repo` (`freeze.repo_dir`). The test monkeypatches `dwc26.data.db.git_state` to return the temporary repository's HEAD and git_dirty 0, so that the lineage check (§1.5.4 step 3) runs against one repository | (1) `evaluate --split test` before the tag → exit 3 (G1), and no test row anywhere. (2) Tag a commit whose `docs/decisions.md` lacks the IDs; `freeze` → exit 3 (G5). (3) Write the IDs, commit, move the tag (`git tag -f`), and `freeze` → exit 0. (4) A dirty tree → `evaluate --split test` exits 3 (G3). (5) Clean tree → exit 0: test metrics for every candidate and `b2`, test predictions (n_test × T rows for each candidate: 6 × 13 on the fixture), `test_scoring` status ok, and the report shows "scored once". (6) A second run → exit 3 (G6), and the row counts are unchanged. (7) `--override-reason "synthetic rerun for E24"` → exit 0; a second `test_scoring` row with the reason; test predictions compared, not rewritten. (8) Commit a change under a frozen path → exit 3 (G4), even with an override. (9) A failed first attempt (monkeypatched exception after the claim) leaves a `failed` row, and a plain rerun exits 3 |
| E25 | `test_evaluate.py` | Determinism | Two `evaluate --split val` runs give equal `metrics` values (exactly) except `cost_*` |
| E26 | `test_evaluate.py` | Consistency check | Changing one stored validation prediction by 1% of y_factor → `evaluate --split val` exits 1 (`ContractError`), and its `runs` row is `failed` |
| E27 | `test_plots.py` | Plots (matplotlib Agg) | Each plot function writes a non-empty PNG that starts with the PNG signature. With `report.plots: true`, `evaluate --split val` writes the 4 validation figures and `backtest` writes 1. Fewer than 2 s for each figure on the fixture |
| E28 | `test_cli.py` | CLI | `match`, `forecast` and `all` still exit 2. `evaluate --split val --override-reason x` → usage error. A short override reason → exit 1. `--config` defaults to `configs/evaluate.yaml` (checked by parsing, not by running) |
| E29 | `test_config.py` | `load_eval_config` | The deliverable config loads. Each of these raises `ConfigError`: an unknown key, `objective_fluids` [oil], `band` 0.3, `p3_fc_quarters` [16, 9], `end_date` 2028-02-01, `strategy` with a model outside PROXIES, `v6_horizons` [0, 12], and `p2_mode` "warn" |

### 4.3 Raw-data tests (marker `rawdata`; validation path only)

| ID | Test | Pass criterion |
| --- | --- | --- |
| E30 | `load_split` on the real database | train (70, 41, 3), val (15, 41, 3). k0 = 4. K = 37 dates, from 1999-01-01 to 2008-01-01. n_fc = 80. t_cut = 29 and H = 12. There is no test-split call |
| E31 | Mismatch on the real history | O(hist_phys as the prediction) = 0. The 15 validation O_sim values are finite and > 0. σ_obs > 0 on K for oil and water, and the water floor applies on the dates before breakthrough |

No raw-data test, script or notebook may call `load_split(..., 'test')` or `evaluate --split test` before the freeze.

### 4.4 Real runs: the data scientist runs and reports these

**Run, in this order:** `evaluate --split val`, `backtest`, `select`. **Do not run** `freeze` or `evaluate --split test`.

Report:

1. **Run IDs and status.** The run IDs, git commits (`git_dirty` 0), durations, and the resolved candidate IDs. The resolved IDs must equal D-023, with the superseded dmd run absent.
2. **Consistency.** The maximum consistency difference (§1.5.2 step 3). J_val for each candidate against D-023: lstm/A 0.0029, lstm/C 0.0035, rnn/A 0.0038, rnn/B 0.0044, lstm/B 0.0046, rnn/C 0.0049, dmd 0.0093, b1 0.0199, b0 0.0710.
3. **Plausibility checks to confirm.**
   - B0's R²_case = 0.
   - lstm/C fails P2 (15 of 15, ratio about 1.50).
   - All proxies pass P1.
   - The DMD P3 flag and R.
   - Coverage flags (seed ensembles under, DMD over).
   - The GOR ranges.
4. **Backtest.** RMSE at h 9 to 12 for oil and water against B2 for the 7 variants, and the backtest train run IDs and times.
5. **Selection.** `selection.md` with the proposed primary and secondary, and `model_comparison.md`.
6. **Tests.** The figures (paths), and the `pytest` counts and time.

### 4.5 Review checklist (researcher, before the merge)

- **V1.** k0, the mean curve and the GOR range come from train cases. Every selection input is a validation metric. σ_proxy is validation only.
- **V2, V5.** Splits come from the stored column. A grep of `src/dwc26/evaluation`, `src/dwc26/forecast` and `src/dwc26/hm` finds no literal for T, k0, n, 29, 12, 80, 37, a split range or a date. All of these come from config or data.
- **V3.**
  - G1 to G7 and the triggers are covered by E15 to E19 and E24.
  - The test path is read line by line: the claim comes before `load_split('test')`; there is a failure path; the predictions on an override scoring are compared, not rewritten.
  - `git tag -l freeze-proxy` in the project repository is empty.
  - The `metrics` and `predictions` tables of the real database have no test row.
- **V4.** The models are loaded with `con` (scaler check). d = `hist_phys`.
- **V6.**
  - Steps 1 and 2 are applied to every proxy.
  - The report marks each rejected model.
- **V7.**
  - matplotlib 3.10.8 is pinned in `pyproject.toml` and in the lock file.
  - All settings are in `configs/evaluate.yaml`.
  - E25.
- **V8.** One `runs` row and one log per command, and the lineage keys in `outputs`.
- **V9.** §1.9.
- **V10.** One test file for each new module, and `pytest` passes.
- **V13.** Continuations come only from each model's own `rollout` and `forecast`. B2 is a benchmark, not a forecast method.
- **Plausibility.** §4.4 items 2 to 5.
- **One command for each step from a clean clone.** The README lists the Task 3 commands and run times.

## 5. Consequences for later tasks

- **Task 4.**
  - Use `dwc26.hm.objective` (`build_mismatch_spec`, `mismatch`) with `sigma_proxy` = `read_rmse_by_date(con, E, primary)[K][:, F_O]`, where E is the evaluate-val run of the frozen selection.
  - The O_obs of the 100 simulated cases (PRD §8 verification) uses the same function with σ_proxy = None.
  - The std filter (PRD §8 step 2.2) compares with the 95th percentile of the primary's `predict_std` on cases 71 to 85.
  - The recovery test reads test cases only after the freeze, through a guard in `evaluation/guards.py` that the Task 4 spec defines. It requires G1, G2 and G5, but not the once-only rule.
- **Task 5.**
  - Use `forecast/dates.py` for the 80 forecast dates.
  - The primary and secondary come from the active `freeze` row.
  - If DMD is the secondary, state the tail limitation (P3, D-022) wherever its forecast measures the method uncertainty.
  - The backtest is already reported (PRD §10 acceptance).
- **`all` (later).** A clean-clone run retrains, so the run IDs change. It must map the frozen choice by (model, variant), which the `freeze` row stores, and it needs a decision on how a fresh database meets G5 and G6.
- **Frozen paths.** After the freeze, a change under `freeze.frozen_paths` blocks a new test scoring (G4). Task 4 and Task 5 code goes in `hm/` (except `objective.py`) and `forecast/` (except `backtest.py` and `dates.py`). A change to `objective.py` after the freeze needs a researcher decision.
