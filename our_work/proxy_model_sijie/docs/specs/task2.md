# Task 2: Proxy Model Training

Status: draft for implementation, 13 Oct 2026, on branch `task2`. Amended 13 Oct 2026 after review 1: amendment A2 (DMD basis, §2.9a; D-019), the DMD plausibility gate (§2.14; D-020), and the review notes (§2.16; D-021). Review 2 (D-022, D-023): approved for merge; P3 was added for Task 3 (§5).

- **Base:** PRD §6, with PRD §3 (protocol, registry, commands and array contracts) and the Task 1 outputs (`docs/specs/task1.md` §1.5 and §5).
- **Departures from the PRD:** D-012 to D-020 (§0.2), logged in `docs/decisions.md`.

**Label key.**

- "source: PRD §n" means the PRD states it.
- "source: Article N, p. X, Eq. Y" cites an article by its journal page. The text copies in `refs/` carry PDF page markers:
  - Article 2: journal page = PDF page + 982.
  - Article 4: journal page = PDF page + 1100.
  - Article 6: journal page = PDF page − 1, because its PDF starts with a cover page.
- Other sources are in `docs/references.md`.
- "assumption" means that this spec decides it and gives a reason.

## 0. Summary

### 0.1 What Task 2 does

```bash
python -m dwc26 train --model {rnn|lstm|dmd|b0|b1} --config configs/models/<model>.yaml [--t-cut N] [--settings-from RUN_ID]
```

Each call is one run. It does four things:

1. Reads the Task 1 arrays from the database. It keeps the train and validation cases only (V1, V3).
2. Tunes the model on the train cases with the validation objective J (§2.5). The budget is 30 trials.
3. Builds the final model of each variant. There is no refit (PRD §6).
4. Writes the following outputs:
   - `outputs/models/<run_id>/`.
   - The train and validation predictions, in the table `predictions`.
   - A training report and a log.
   - One `runs` row.

| Name | Class | Role | Strategy | Variants | Tuning (budget 30) |
| --- | --- | --- | --- | --- | --- |
| `rnn` | `RNNProxy` | Proxy | S3 | `A`, `B`, `C` (input option) | 30 Latin-hypercube trials, 10 for each option. The cell type (`rnn` or `gru`) is a tuned setting |
| `lstm` | `LSTMProxy` | Proxy | S3 | `A`, `B`, `C` | 30 Latin-hypercube trials, 10 for each option |
| `dmd` | `DMDProxy` | Proxy | S4 | `main` | Full grid of 24 settings |
| `b0` | `MeanCurveBenchmark` | Benchmark only | none | `main` | None |
| `b1` | `KNNBenchmark` | Benchmark only | none | `main` | None (k = 3 with distance weights, fixed by PRD §6) |

**Proxy ID.**

- A fitted model is named by its proxy ID, `<run_id>/<variant>`. Example: `train-20261013T120000000000Z/B`.
- Where CLAUDE.md and PRD §3 write `<run_id>` for `freeze`, `match` and `forecast`, those commands take a proxy ID (D-014).

**Run time.**

- The data scientist reports the measured time.
- Estimate: 42 network fits per neural model, at 10 to 60 s each on one CPU thread. This gives 10 to 40 min per neural model at `n_jobs: 1`.
- DMD: 5 to 20 min. B0 and B1: seconds.
- `n_jobs` > 1 runs fits in parallel and gives identical results (test M17).

### 0.2 Departures from the PRD

| ID | PRD | This spec | Reason |
| --- | --- | --- | --- |
| D-012 | "Loss (neural models): mean squared error on the scaled rates, with equal weight for oil, gas and water" (§6) | Default fluid weights w = (oil 0.25, gas 0.25, water 0.5). They apply to the neural loss (§2.4) and to the DMD snapshot rows (§2.9). Equal weights (1/3 each) stay a tuned option in each model's budget | D-010: gas ≈ 0.3633 × oil, so equal weights give oil about 2/3 of the weight. The user approved the change on 13 Oct 2026 |
| D-013 | DMD snapshot matrix: 41 columns. Sequences over 41 steps | Each model fits the dates k ≥ k0 only. k0 is the first date with a non-zero training rate, found at run time (4 on the real data). Before k0, every model returns exactly 0 | No linear dynamics maps a zero state to a non-zero state. All 100 cases are 0 before k0 (C10). The match window (Task 4) starts at k0 |
| D-014 | The RNN and LSTM input (A, B or C) is a "config option". One model for each name | The option splits the 30-trial budget into strata of 10 trials. Each option gets its own final 5-seed ensemble, named by a proxy ID. Task 3 chooses the pair (model, option) from the validation and the backtest | Validation within 1998 to 2008 cannot see the extrapolation in time, and that extrapolation is what separates the options. Only the backtest (PRD §9) can see it |
| D-015 | 30 trials. Settings selected by the validation error. Early stopping on cases 71 to 85 | Seeded Latin-hypercube search in NumPy (no new dependency). The objective is J, the mean validation NRMSE of oil and water (the Task 3 ranking metric). J is used for early stopping and for the selection. Each trial uses 1 seed, and the final ensemble adds the other 4 seeds. DMD evaluates its full grid of 24 settings (≤ 30). B0 and B1 are not tuned | §2.14 |
| D-016 | DMD steps 1 to 6 | (a) The time index is the integer quarter step. (b) Eigenvalues with \|λ\| > 1 move to the unit circle before the least squares. (c) A real cos/sin basis, equivalent to Re Σ c λ^t. (d) One GP for each fluid and basis coefficient, with a Matérn 5/2 ARD kernel by default. (e) The GP target is c (PRD) or its orthonormal projection (tuned). (f) `predict_std` = GP variance mapped through the basis, plus the basis residual | §2.9 |
| D-017 | `Proxy` protocol (PRD §3) | Adds `rollout`, the keyword `val` in `fit`, and the attributes `variant`, `T_fit`, `k0` and `last_n_clipped`. `predict` returns clipped values, and X must lie in [0, 1]. Benchmarks have the same methods with `extrapolates_time = False`, and their `forecast` raises | §1.3, §2.11, §2.12 |
| D-018 | `predictions`: run_id, case_id, date_idx, oil, gas, water | Adds the columns variant, split, date and the std columns. Values are in physical units. No foreign key to `parameters`. Task 2 writes train and validation rows only. `runs.packages` adds torch and scikit-learn | §1.6 |
| D-019 | DMD step 2: "Fit exact DMD with rank r". Settings: "Rank r for 99% of the energy" (§6) | Amendment A2. The default basis is optimized DMD with real eigenvalues in [λ_min, 1], fitted by variable projection. Exact DMD at the same rank gives its first start. The rank is tuned directly over {2, 3, 4, 5, 6, 8}. The 99%-energy rank is reported only. The kernel is fixed to Matérn 5/2 | §2.9a. Review 1: the exact basis (r = 4) took the oil forecast below 0 in 2009 to 2012 for all 15 validation cases. The energy grid also gave duplicate points |
| D-020 | Selection by the validation error (§6); PRD §7 selection steps 1 to 7 | DMD tuning selects the lowest J among the grid points whose validation continuation clips no value (P1). Task 3 adds P1 and P2 (rising oil) to the selection, plus a fallback secondary | §2.14, §5 |

### 0.3 Facts from Task 1 used here

- `Y[:, k]` is the mean rate over the quarter that ends at `dates[k]` (R1, D-004).
  - On the real data, `Y[:, 0:4] = 0` for all 100 cases (C10), so k0 = 4 (§2.2).
  - On 1999-01-01, all training cases share one oil rate (15,507.77 STB/d) (task1 §5).
- Gas ≈ 0.3633 × oil (C7). Thus Y_gas ≈ Y_oil in scaled units (D-010). Water is 0 before breakthrough.
- Y is scaled per fluid by the train maximum: y_factor = [15678.18, 5695.88, 9166.93].
  - Validation and test values reach 1.003. No output activation may cap the prediction at 1.
- X ∈ [0, 1]⁴. The design minimum and maximum map to exactly 0 and 1 (D-003).
- Real sizes are read at run time and never hard-coded (V5): n_fit = 70, n_val = 15, T = 41, p = 4, T_p = T − k0 = 37.

## 1. Inputs and outputs

### 1.1 Inputs

| Input | Content | Read by |
| --- | --- | --- |
| `data/processed/dwc26.sqlite` | Task 1 tables, through `read_task1_arrays(con)` (F2 guard) | `load_dataset` (§1.3) |
| `configs/models/<model>.yaml` | All settings (V7), §1.2 | `load_model_config` |
| `outputs/models/<src_run>/<variant>/meta.json` | Settings of an earlier run. Read only with `--settings-from` (§2.13) | `run_train` |

`arrays.npz` is not read: the database is the single source (PRD §5).

### 1.2 Configs (one YAML for each model in `configs/models/`)

`load_model_config(path)` validates every key (§3). A missing key or a value out of range raises `ConfigError`. The values below are the deliverable defaults.

`configs/models/lstm.yaml`:

```yaml
model: lstm                      # must equal --model
seed: 0                          # seed of the trial plan (runs.seed)
paths:
  db: data/processed/dwc26.sqlite
  models: outputs/models
  logs: outputs/logs
data:
  t_cut: null                    # null = all T dates; int = first t_cut dates (Task 5 backtest); --t-cut overrides
objective:
  fluids: [oil, water]           # J = mean NRMSE of these fluids (§2.5)
fluid_weight_sets:               # [oil, gas, water]; each set sums to 1 (D-012)
  d012: [0.25, 0.25, 0.5]
  equal: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
network:
  input_options: [A, B, C]       # strata of the plan; one final ensemble for each (D-014)
  x_affine: true                 # u = 2x - 1 (Article 2, p. 989)
training:
  seeds: [0, 1, 2, 3, 4]         # ensemble members (PRD §6)
  tune_seeds: 1                  # members trained for each trial: seeds[:tune_seeds] (D-015)
  max_epochs: 1000
  patience: 50
  min_delta: 0.0
  grad_clip_norm: 1.0            # null = no clipping
  adam: {beta1: 0.9, beta2: 0.999, eps: 1.0e-8, weight_decay: 0.0}
  dtype: float32
  torch_threads: 1
  n_jobs: 1                      # worker processes for the fits; results must not depend on it
  predict_batch: 2048            # rows per forward pass in predict, rollout and forecast
tuning:
  n_trials: 30
  sampler: lhs
  space:                         # dimension order = draw order (§2.14)
    num_layers:    {type: choice, values: [1, 2]}
    hidden_size:   {type: choice, values: [16, 32, 48, 64]}
    lr:            {type: loguniform, low: 3.0e-4, high: 1.0e-2}
    batch_size:    {type: choice, values: [4, 8, 16, 32]}
    fluid_weights: {type: choice, values: [d012, equal]}
report:
  plateau_frac: 0.98             # PRD §7 plateau end
  breakthrough_frac: 0.01        # PRD §7 water breakthrough
  forecast_steps: 80             # continuation check in the report (§1.7)
  coverage_z: 1.959963984540054  # 95% two-sided normal quantile
speed:
  n_batch_rows: 10000            # one predict call (PRD §7 cost metric)
  n_calls: 200                   # timed calls of call_rows rows
  call_rows: 32                  # walkers (PRD §8)
  mcmc_calls: 5000               # steps (PRD §8)
  limit_s: 600                   # PRD §6
```

`configs/models/rnn.yaml` is the same with `model: rnn` and one more dimension, added last:

```yaml
    cell:          {type: choice, values: [rnn, gru]}
```

`configs/models/dmd.yaml`:

```yaml
model: dmd
seed: 0                          # GP optimiser restarts (random_state)
paths: {db: data/processed/dwc26.sqlite, models: outputs/models, logs: outputs/logs}
data: {t_cut: null}
objective: {fluids: [oil, water]}
fluid_weight_sets:
  d012: [0.25, 0.25, 0.5]
  equal: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
dmd:
  basis: real                    # real (A2, §2.9a; default) | exact (§2.9 steps 4 and 5; experiments and the 13 Oct record)
  real_kappa_max: 3.0            # A2: upper bound of each decay-rate increment d_j (lambda_{j+1}/lambda_j >= e^-3 = 0.05)
  real_kappa_sep: 0.005          # A2: lower bound of d_j, j >= 2 (consecutive lambdas differ by >= 0.5%)
  energy_report: 0.99            # A2: r_99, the PRD section 6 energy rank, reported only
  unstable: clip                 # clip | keep (§2.9 step 5; exact basis only)
  stab_tol: 1.0e-9
  imag_tol: 1.0e-10
  sigma_rel_tol: 1.0e-10
  dup_tol: 1.0e-8
  r_max: null
gp:
  alpha: 1.0e-10
  normalize_y: true
  n_restarts_optimizer: 5
  constant_bounds: [1.0e-3, 1.0e+3]
  length_scale_bounds: [1.0e-2, 1.0e+2]
  noise_init: 1.0e-4
  noise_bounds: [1.0e-10, 1.0]
  std_includes_basis_residual: true
tuning:
  n_trials: 30                   # cap; the grid below has 6 x 1 x 2 x 2 = 24 points (A2)
  sampler: grid
  space:                         # product order: last dimension fastest
    rank:          {type: choice, values: [2, 3, 4, 5, 6, 8]}   # A2: replaces energy
    kernel:        {type: choice, values: [matern25]}           # A2: fixed (matern15 changed J by < 1e-4 in review 1)
    fluid_weights: {type: choice, values: [d012, equal]}
    gp_target:     {type: choice, values: [coef, proj]}
report: {plateau_frac: 0.98, breakthrough_frac: 0.01, forecast_steps: 80, coverage_z: 1.959963984540054}
speed: {n_batch_rows: 10000, n_calls: 200, call_rows: 32, mcmc_calls: 5000, limit_s: 600}
```

`configs/models/b0.yaml` has the keys `model: b0`, `seed`, `paths`, `data`, `objective` and `report`. `configs/models/b1.yaml` has the same keys plus:

```yaml
knn: {n_neighbors: 3, weights: distance, algorithm: brute}
```

Benchmarks have no `speed` block, because they never enter the MCMC.

### 1.3 Modules and signatures

The layout follows PRD §3. All arrays are `np.float64` unless stated otherwise. V9 assertions raise `ContractError` (from `dwc26.data.check`) and never use `assert`. `FLUIDS` comes from `dwc26.data.load`.

| Module | Content |
| --- | --- |
| `src/dwc26/models/base.py` | `Proxy` protocol, `onset_index`, `clip_nonneg`, `check_X`, `check_XY`, `TrialFailed` |
| `src/dwc26/models/registry.py` | `PROXIES`, `BENCHMARKS`, `get_class`, `parse_proxy_id`, `load_model` |
| `src/dwc26/models/dataset.py` | `Dataset`, `load_dataset` |
| `src/dwc26/models/tuning.py` | `lhs_unit`, `make_plan`, `resolve_settings` |
| `src/dwc26/models/seq.py` | `SeqNet`, `MemberResult`, `train_member`, `SequenceProxy` (shared by RNN and LSTM) |
| `src/dwc26/models/rnn.py`, `lstm.py` | `RNNProxy` (cell from the settings), `LSTMProxy` (cell `lstm`) |
| `src/dwc26/models/dmd.py` | `energy_rank`, `exact_dmd_eigs`, `clean_eigs`, `real_basis`, `DMDProxy` |
| `src/dwc26/models/baselines.py` | `MeanCurveBenchmark` (b0), `KNNBenchmark` (b1) |
| `src/dwc26/models/train.py` | `TrialResult`, `TrainOutcome`, `train_model`, `run_train` |
| `src/dwc26/models/train_report.py` | `diagnostics`, `speed_benchmark`, `write_train_report` |
| `src/dwc26/evaluation/metrics.py` | `rmse`, `nrmse`, `objective_j` (Task 3 reuses them) |
| `src/dwc26/config.py` | Adds `load_model_config(path) -> dict` |
| `src/dwc26/data/db.py` | Adds the `predictions` DDL to `SCHEMA`, `write_predictions`, and torch and scikit-learn in `PACKAGES` |
| `src/dwc26/cli.py` | Subcommand `train` |

```python
# base.py
class TrialFailed(RuntimeError): ...       # numerical failure of one trial (NaN loss, singular basis)

@runtime_checkable
class Proxy(Protocol):                     # PRD §3, extended (D-017)
    name: str                              # registry name: 'rnn' | 'lstm' | 'dmd' | 'b0' | 'b1'
    variant: str                           # 'A' | 'B' | 'C' | 'main'
    extrapolates_time: bool                # True for PROXIES, False for BENCHMARKS (V13)
    T_fit: int                             # Y.shape[1] at fit
    k0: int                                # onset index (§2.2), set by fit
    last_n_clipped: int                    # negatives set to 0 by the last predict/rollout/forecast call

    def fit(self, X: np.ndarray, Y: np.ndarray, cfg: dict, *,
            val: tuple[np.ndarray, np.ndarray] | None = None) -> "Proxy": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...                 # (n, T_fit, 3) scaled, >= 0
    def predict_std(self, X: np.ndarray) -> np.ndarray | None: ...      # (n, T_fit, 3) scaled, >= 0; None for benchmarks
    def rollout(self, X: np.ndarray, n_total: int) -> np.ndarray: ...   # (n, n_total, 3), steps 0..n_total-1, >= 0
    def forecast(self, X: np.ndarray, n_steps: int) -> np.ndarray: ...  # (n, n_steps, 3), steps T_fit..T_fit+n_steps-1
    def save(self, path: Path) -> None: ...                             # path = <models>/<run_id>/<variant>/
    @classmethod
    def load(cls, path: Path) -> "Proxy": ...

def onset_index(Y: np.ndarray) -> int: ...                               # (n, T, 3) -> k0 (§2.2)
def clip_nonneg(a: np.ndarray) -> tuple[np.ndarray, int]: ...            # (max(a, 0), count of a < 0)
def check_X(X: np.ndarray, p: int) -> np.ndarray: ...                    # V9, §1.8; returns float64 X
def check_XY(X: np.ndarray, Y: np.ndarray) -> None: ...                  # V9, §1.8
```

- `cfg` in `fit` is the flat settings dict from `resolve_settings` (§2.14), not the YAML.
- The neural models require `val`, for early stopping. If `val` is None, they raise `ContractError`.
- DMD and the benchmarks ignore `val`.
- **Suggested structure (not binding).** A base class implements the input checks, the onset, the clipping and the count. It also implements `predict = rollout(X, T_fit)` and `forecast = rollout(X, T_fit + n_steps)[:, T_fit:]`. Each subclass implements a raw rollout of the producing steps k0..n_total−1.

```python
# registry.py: lazy imports, so that `dmd`, `b0` and `b1` run without importing torch
PROXIES = {"rnn": "dwc26.models.rnn:RNNProxy", "lstm": "dwc26.models.lstm:LSTMProxy",
           "dmd": "dwc26.models.dmd:DMDProxy"}
BENCHMARKS = {"b0": "dwc26.models.baselines:MeanCurveBenchmark",
              "b1": "dwc26.models.baselines:KNNBenchmark"}
MODEL_NAMES = (*PROXIES, *BENCHMARKS)

def get_class(name: str) -> type: ...
    # Imports the class. Raises ContractError unless extrapolates_time is True
    # for PROXIES and False for BENCHMARKS. Unknown name -> ConfigError.
def parse_proxy_id(proxy_id: str) -> tuple[str, str]: ...        # '<run_id>/<variant>' -> (run_id, variant)
def load_model(proxy_id: str, models_dir: Path, *, con: sqlite3.Connection | None = None,
               proxies_only: bool = True) -> Proxy: ...
    # Reads <models_dir>/<run_id>/<variant>/meta.json, then get_class(meta['model']).load(dir).
    # With proxies_only, a benchmark raises ContractError (Tasks 4 and 5 call it this way).
    # With con, the y_factor, x_lower and x_upper in meta.json must equal load_scalers(con) exactly
    # (np.array_equal), else ContractError (V4).
```

```python
# dataset.py
@dataclass(frozen=True)
class Dataset:
    X_fit: np.ndarray          # (n_fit, p) scaled, split 'train'
    Y_fit: np.ndarray          # (n_fit, T_fit, 3) scaled
    X_val: np.ndarray          # (n_val, p)
    Y_val: np.ndarray          # (n_val, T_fit, 3)
    case_fit: np.ndarray       # (n_fit,) int64, ascending
    case_val: np.ndarray       # (n_val,) int64, ascending
    dates: np.ndarray          # (T_fit,) datetime64[D]
    t_days: np.ndarray         # (T_fit,)
    T_full: int                # T of the Task 1 arrays
    t_cut: int | None
    y_factor: np.ndarray       # (3,) STB/d, MSCF/d, STB/d
    x_lower: np.ndarray        # (p,)
    x_upper: np.ndarray        # (p,)
    x_names: tuple[str, ...]
    fluids: tuple[str, ...]
    prepare_run_id: str
    arrays_sha256: str

def load_dataset(con: sqlite3.Connection, t_cut: int | None = None) -> Dataset: ...
```

`load_dataset` does six things, in this order:

1. It calls `a = read_task1_arrays(con)`, which carries the F2 guard.
2. It computes `dwc26.data.prepare.arrays_sha256(a)` and compares the value with `runs.outputs.arrays_sha256` of the prepare run `a['run_id']`. A mismatch raises `ContractError`. Reason: the DB tables must be the ones that the logged prepare run wrote (V8).
3. It selects rows by the stored `split` column: `train` gives the fit arrays and `val` the validation arrays. It never uses case-number ranges (V2, V5).
4. It deletes the `test` rows before it returns. The `Dataset` holds no test array, and no test case ID (V3).
5. If `t_cut` is not None, it slices `Y_fit`, `Y_val`, `dates` and `t_days` to the first `t_cut` dates. It raises `ContractError` unless k0 + 3 ≤ t_cut ≤ T_full, with k0 from the train cases of the full arrays.
6. It runs the V9 checks of §1.8.

```python
# tuning.py
def lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray: ...   # (n, d) in [0, 1), §2.14
def make_plan(cfg: dict) -> list[dict]: ...
    # [{'trial': int, 'variant': str, 'settings': {dim: value}}], trial = 0..N-1 in stratum order
def resolve_settings(cfg: dict, variant: str, settings: dict) -> dict: ...  # flat fit cfg, §2.14
```

```python
# seq.py
class SeqNet(torch.nn.Module):
    def __init__(self, cell: str, option: str, p: int, num_layers: int, hidden_size: int): ...
    def forward(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
        # u: (n, p) in [-1, 1]; t: (L,) time values of the L steps; returns (n, L, 3) raw

@dataclass
class MemberResult:
    seed: int
    state: dict                    # best-epoch state_dict, CPU tensors
    best_epoch: int
    n_epochs: int
    j_val: float                   # J of this member at best_epoch
    history: np.ndarray            # (n_epochs, 3): epoch, train loss, J_val
    fit_s: float

def train_member(settings: dict, seed: int, X: np.ndarray, Y: np.ndarray,
                 X_val: np.ndarray, Y_val: np.ndarray, k0: int) -> MemberResult: ...   # §2.7
                 # module-level function, so that worker processes can run it

class SequenceProxy:               # implements Proxy; RNNProxy and LSTMProxy subclass it
    extrapolates_time = True
    @classmethod
    def from_members(cls, variant: str, settings: dict, members: list[MemberResult],
                     k0: int, T_fit: int, p: int) -> "SequenceProxy": ...
    # fit(X, Y, cfg, val=...) == from_members over [train_member(cfg, s, ...) for s in cfg['seeds']]
```

```python
# dmd.py
def energy_rank(sigma: np.ndarray, energy: float, sigma_rel_tol: float,
                r_max: int | None = None) -> tuple[int, int]: ...        # (r, q), §2.9 step 3
def exact_dmd_eigs(D: np.ndarray, r: int) -> np.ndarray: ...            # (r,) complex128, §2.9 step 4
def clean_eigs(lam: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]: ...
    # (reps (m,) complex128, kinds (m,) int8 [0 real, 1 pair], info), §2.9 step 5
def real_basis(reps: np.ndarray, kinds: np.ndarray, tau: np.ndarray) -> np.ndarray: ...
    # (len(tau), r_eff), §2.9 step 6
def varpro_real_eigs(D: np.ndarray, m: int, cfg: dict) -> tuple[np.ndarray, dict]: ...
    # A2, §2.9a: (lam (m,) float64, strictly descending, in (0, 1]; info with lam_exact, varpro {...})

class DMDProxy:                    # implements Proxy
    extrapolates_time = True
    coef_train: np.ndarray         # (n_fit, 3, r_eff) least-squares coefficients a (§2.9 step 7)
    diagnostics: dict              # §2.10
    def rollout_from_coef(self, A: np.ndarray, n_total: int) -> np.ndarray: ...
        # (n, 3, r_eff) -> (n, n_total, 3) raw, unclipped; used by the tests
```

```python
# baselines.py
class MeanCurveBenchmark: ...      # name 'b0', extrapolates_time = False (§2.11)
class KNNBenchmark: ...            # name 'b1', extrapolates_time = False (§2.11)
```

```python
# evaluation/metrics.py: y arrays (n, T, F), physical or scaled (NRMSE and J do not depend on the scale)
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...     # (F,)
def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...    # (F,); ContractError if a range is 0
def objective_j(y_true: np.ndarray, y_pred: np.ndarray, fluid_idx: tuple[int, ...]) -> float: ...
```

```python
# train.py
@dataclass
class TrialResult:
    trial: int
    variant: str
    settings: dict                 # tuned values only
    status: str                    # 'ok' | 'failed'
    error: str | None
    j_val: float                   # +inf if failed
    nrmse_val: np.ndarray          # (3,), NaN if failed
    extra: dict                    # neural: best_epoch, n_epochs (per seed); DMD: r, r_eff, n_stabilised, cond_B
    fit_s: float
    members: list[MemberResult] | None    # neural
    model: "DMDProxy | None"              # DMD

@dataclass
class TrainOutcome:
    models: dict[str, Proxy]       # variant -> final model
    trials: list[TrialResult]
    best_trial: dict[str, int]     # variant -> trial id

def train_model(name: str, cfg: dict, data: Dataset, *,
                settings: dict[str, dict] | None = None) -> TrainOutcome: ...
    # settings=None: tune (§2.14). settings={variant: tuned values}: no tuning, one fit
    # for each variant (Task 5 backtest).
def run_train(config_path: Path, model: str, *, t_cut: int | None = None,
              settings_from: str | None = None) -> str: ...      # returns run_id; §1.4
```

```python
# db.py additions
def write_predictions(con, run_id: str, variant: str, case_id: np.ndarray, split: np.ndarray,
                      dates: np.ndarray, Y_phys: np.ndarray, S_phys: np.ndarray | None,
                      *, allow_test: bool = False) -> int: ...
    # Returns the number of rows written. No commit. ContractError if any split == 'test'
    # and not allow_test. Task 2 never sets allow_test; Task 3 may, after the freeze (V3).
```

**CLI.**

- Syntax: `train --model NAME --config PATH [--t-cut N] [--settings-from RUN_ID]`.
- Success: prints the `run_id` and exits with 0.
- `ConfigError` or `ContractError`: prints the message to stderr and exits with 1.
- Any other exception propagates.
- In every failure after `start_run`, the `runs` row ends with status `failed`.
- `--model` must equal `cfg['model']`. Otherwise it raises `ConfigError`.
- The other commands still exit with 2 ("not implemented").

### 1.4 Processing order in `run_train`

1. **Config.**
   - Run `cfg = load_model_config(config_path)`. Apply `--t-cut` to `cfg['data']['t_cut']`. The resolved config is the logged config.
   - For `rnn` and `lstm`, also run `torch.set_num_threads(training.torch_threads)` and `torch.use_deterministic_algorithms(True)`.
2. **Run row and log.**
   - Run `connect`, then `init_schema` (which now creates `predictions`).
   - Run `start_run(con, task='task2', command='train', cfg=cfg, config_path=..., seed=cfg['seed'], model=name)`.
   - Open `outputs/logs/<run_id>.log` (Task 1 format, UTC).
3. **Data.**
   - Run `data = load_dataset(con, t_cut)`.
   - Log n_fit, n_val, T_fit, k0, the prepare run and `arrays_sha256`.
4. **Settings.**
   - With `--settings-from SRC`, run §2.13: no plan.
   - Otherwise run `plan = make_plan(cfg)`.
5. **Trials and final models.** Run `outcome = train_model(name, cfg, data, settings=...)` (§2.14).
6. **Diagnostics for each variant** (§1.7, §2.10, §2.15):
   - Predictions of the train and validation cases, with the clip counts.
   - Metrics.
   - The coverage preview.
   - The validation continuation check.
   - The DMD diagnostics.
   - The speed benchmark (proxies only).
7. **Save.** For each variant, run `model.save(<models>/<run_id>/<variant>/)`. Then run `load_model(proxy_id, models_dir, con=con)`. On the validation cases:
   - `predict`, `predict_std` and `rollout(X_val, T_fit + forecast_steps)` must equal the in-memory model bit for bit (`np.array_equal`).
   - Otherwise raise `ContractError`.
8. **Files.** Write `config.yaml`, `trials.csv`, `histories/`, `report.md` (§1.5).
9. **Database.** In one transaction:
   - Run `write_predictions` for every variant (train and validation cases).
   - Then run `finish_run(status='ok', outputs=...)` (§1.6), which commits.

**Failure path.**

- Any exception after step 2:
  - Rolls back the open transaction.
  - Calls `finish_run(status='failed', outputs={'error': str(e)})`.
  - Then re-raises.
- A failed run writes no `predictions` rows.
- Partial files may remain in its model directory. Tasks 3 to 5 use only runs with status `ok`.

### 1.5 Model directory `outputs/models/<run_id>/`

| Path | Content |
| --- | --- |
| `config.yaml` | Resolved config, plus the CLI arguments `t_cut` and `settings_from` |
| `trials.csv` | One row per trial. Columns: `trial`, `variant`, `status`, `error`, one column for each tuned setting, `j_val`, `nrmse_val_oil`, `nrmse_val_gas`, `nrmse_val_water`, `fit_s`. Neural models add `best_epoch` and `n_epochs` (`;`-joined over seeds). DMD adds `r`, `r_eff`, `n_stabilised`, `cond_B`, and (A2) `basis`, `n_clip_cont`, `plausible` |
| `histories/trial_<i>_seed_<s>.csv` | Neural only: `epoch`, `train_loss`, `j_val` |
| `report.md` | Training report (§1.7) |
| `<variant>/meta.json` | See below |
| `<variant>/weights.pt` | RNN, LSTM: `torch.save({'seeds': [...], 'states': [state_dict, ...]})`. Loaded with `torch.load(..., weights_only=True)` |
| `<variant>/dmd.npz` | DMD (`allow_pickle=False`): `lam_raw`, `reps` (complex128), `kinds` (int8), `B_fit` (T_p, r_eff), `R` (r_eff, r_eff), or empty for `coef`, `coef_train` (n_fit, 3, r_eff), `resid_var` (T_p, 3), `sigma` (q,), `X_fit` (n_fit, p) |
| `<variant>/gps.pkl` | DMD: `pickle` (protocol 5) of the fitted `GaussianProcessRegressor` objects, as a list over fluids of lists over coefficients |
| `<variant>/b0.npz`, `b1.npz` | B0: `mean_curve` (T_fit, 3). B1: `X_fit`, `Y_fit`. B1 rebuilds the k-NN on load, so no pickle is needed |

**`meta.json` keys.**

| Group | Keys |
| --- | --- |
| Format and identity | `format_version` (1), `model`, `class` (`module:Class`), `variant`, `proxy_id`, `run_id`, `extrapolates_time` |
| Settings | `settings` (the flat fit cfg), `k0`, `T_fit`, `t_cut`, `settings_from` |
| Data | `p`, `x_names`, `fluids`, `y_factor`, `x_lower`, `x_upper` (the scalers, PRD §6 "Save"), `prepare_run_id`, `arrays_sha256` |
| Environment and files | `packages` {numpy, scipy, torch, scikit-learn}, `files` {name: sha256}, `created` (UTC ISO 8601) |

**Load.**

- `load` checks `format_version` and the sha256 of every file before it reads the file. A mismatch raises `ContractError`.
- `gps.pkl` is unpickled only after that check. Assumption: the file is a trusted local artefact of this repository, and the versions are pinned (V7).

### 1.6 Database

**New table.** Added to `SCHEMA` in `db.py`, so that `init_schema` creates it:

```sql
CREATE TABLE IF NOT EXISTS predictions (
  run_id    TEXT NOT NULL REFERENCES runs(run_id),
  variant   TEXT NOT NULL,                   -- 'A','B','C' (rnn, lstm) or 'main'
  case_id   INTEGER NOT NULL,                -- no FK: Task 1 deletes and refills `parameters` (task1 §1.6)
  split     TEXT NOT NULL CHECK (split IN ('train','val','test')),   -- Task 2 writes train and val only (V3)
  date_idx  INTEGER NOT NULL,                -- 0..T_fit-1
  date      TEXT NOT NULL,                   -- 'YYYY-MM-DD'
  oil       REAL NOT NULL, gas REAL NOT NULL, water REAL NOT NULL,        -- physical (STB/d, MSCF/d, STB/d), clipped >= 0
  oil_std   REAL, gas_std REAL, water_std REAL,                          -- physical; NULL when predict_std is None
  PRIMARY KEY (run_id, variant, case_id, date_idx)
);
```

**Rows for each run.** Σ over variants of (n_fit + n_val) × T_fit. On the real data: 3 × 85 × 41 = 10,455 for `rnn` and `lstm`, and 3,485 for the others.

**`runs` row** (V8):

- `task = 'task2'`, `command = 'train'`, `model = name`, `seed = cfg['seed']`.
- `packages` adds `torch` and `scikit-learn`. `PACKAGES` in `db.py` gets `"torch": "torch"` and `"scikit-learn": "scikit-learn"`.
- `package_versions()` records these for every run, including `prepare` runs. It writes "not installed" if a package is absent. It reads the versions through `importlib.metadata`, so it imports neither package (D-018).

**`runs.outputs`** JSON keys:

| Group | Keys |
| --- | --- |
| Files and provenance | `files` {model_dir, report, log, trials}; `prepare_run_id`; `arrays_sha256` |
| Data window | `t_cut`, `T_fit`, `k0`, `n_fit`, `n_val` |
| Run setup | `settings_from`, `n_trials`, `n_failed_trials`, `torch_threads`, `n_jobs` |
| Per variant | `variants` {variant: {...}}, keys below |
| Summary | `best_variant_by_val` (lowest `j_val`; information only, Task 3 chooses) |

Keys of each entry in `variants`:

- Identity and settings: `proxy_id`, `best_trial`, `settings`.
- Validation objective: `j_val_trial` (the trial's score), `j_val` (the final model).
- Errors in physical units: `rmse_train`, `rmse_val`, `nrmse_train`, `nrmse_val`, each {oil, gas, water}.
- Clipping: `n_clipped` {train, val, forecast_val}.
- `coverage_val` {oil, gas, water}, or null for benchmarks.
- `speed` {predict_n_batch_rows_s, per_call_s, mcmc_extrapolated_s, pass}, or null for benchmarks.
- DMD only: `dmd` {q, r, r_eff, n_stabilised, n_merged, max_abs_lam_raw, cond_B}, plus (A2) `basis`, `rank`, `r_99` (rank for 99% of the energy, PRD §6, report only), `lam_used`, `varpro_cost` and `plausible_selected`.
- `files` {name: sha256}.

### 1.7 Training report `outputs/models/<run_id>/report.md`

1. **Header:**
   - Run ID, model, git commit (`+dirty`), UTC start, config path and sha256.
   - Prepare run and `arrays_sha256`.
   - n_fit, n_val, T_fit, k0, `t_cut`, `settings_from`.
   - Seeds, `tune_seeds`, `torch_threads`, `n_jobs`.
   - Duration.
2. **Search space and plan:** the space table, and the number of trials for each variant.
3. **Trials:** all trials sorted by `j_val`, with the failed trials and their errors at the end.
4. **For each variant:**
   - The best settings.
   - J of the trial and J of the final ensemble.
   - RMSE (physical units) and NRMSE for each fluid, on train and validation.
   - The best epoch of each member.
5. **Benchmarks (information only; V6 is applied in Task 3):**
   - Source runs: the latest `runs` rows with model `b0` and `b1`, status `ok`, and the same `arrays_sha256` and `t_cut`.
   - Their validation RMSE of oil and water.
   - For each variant, whether it is lower than both.
   - If a benchmark run is missing, the report says so.
6. **Clipping:** counts on train, validation, and the validation continuation (PRD §6).
7. **Uncertainty preview:** the share of validation values (k ≥ k0) with |y − ŷ| ≤ z · std, for each fluid (Task 3 computes the official coverage), and the mean std ÷ RMSE.
8. **Continuation check on the validation cases**, with `rollout(X_val, T_fit + forecast_steps)`:
   - The minimum and maximum of the forecast for each fluid.
   - The number of cases whose oil rate at any forecast step exceeds the oil rate at T_fit − 1.
   - The largest ratio of a forecast oil rate to the oil rate at T_fit − 1.
   - The clip count.
   - Plausibility: the field has no new wells after the history (assumption), so a rising oil rate is a warning, not an error.
9. **GOR check (D-010):** the range of predicted gas ÷ oil on the validation cases, at dates where the predicted oil is above 1% of its maximum, against the C7 range.
10. **Plateau and breakthrough errors (§2.10).** All models. Required for DMD (PRD §6 step 6).
11. **DMD only:**
    - A2: `basis`, `rank`, `r_99`, `lam_exact` (S0), the VarPro cost of each start and the best start, and `plausible_selected` (§2.14).
    - The singular values and the cumulative energy, with q and r.
    - The eigenvalue table: λ before and after the cleaning, |λ|, angle, period (quarters) and half-life (quarters).
    - n_stabilised, n_merged, r_eff and cond(B).
    - The GP hyperparameters for each (fluid, coefficient), flagged when at a bound, and the count of convergence warnings.
12. **Speed (§2.15):** the times, and pass or fail against `limit_s`.

### 1.8 V9 assertions at module boundaries

| Boundary | Assertion |
| --- | --- |
| `load_dataset` → | `X_fit` (n_fit, p), `X_val` (n_val, p): finite, 0 ≤ X ≤ 1. `Y_fit`, `Y_val`: (n, T_fit, 3), finite, ≥ 0. n_fit ≥ 2, n_val ≥ 1. No `test` case in `case_fit` ∪ `case_val`. `dates` strictly increasing |
| `fit` input (`check_XY`) | X (n, p), float64, finite, 0 ≤ X ≤ 1. Y (n, T, 3), float64, finite, ≥ 0. `val`, if given, has the same p and T. T − k0 ≥ 3 |
| `predict`, `predict_std`, `rollout`, `forecast` input (`check_X`) | X (n, p) with p = fitted p, n ≥ 1, finite, 0 ≤ X ≤ 1 exactly. Task 4 must therefore evaluate the prior first and call the proxy only for rows inside the box (D-003). n_total ≥ 1, n_steps ≥ 1, both integers |
| `predict`, `rollout`, `forecast` → | float64, expected shape, finite, ≥ 0. Exactly 0 for k < k0 |
| `predict_std` → | (n, T_fit, 3) float64, finite, ≥ 0, exactly 0 for k < k0. None only for benchmarks |
| `write_predictions` input | `Y_phys` and `S_phys` (n, T_fit, 3), finite, ≥ 0. No `test` row |

## 2. Equations and methods

### 2.1 Notation

- i = case, f = fluid (oil, gas, water), k = date index.
- k = 0..T_fit − 1 is the fitted window. k ≥ T_fit is the continuation: on the real data T_fit = 41, so the 80 forecast dates are k = 41..120.
- Y_{i,k,f} is the scaled rate. s_f is the Y factor, and the physical rate is q = Y · s_f.
- k0 is the onset (§2.2). τ = k − k0 is the step since the onset, and T_p = T_fit − k0.
- p is the number of parameters (4). n_fit and n_val are the case counts.

### 2.2 Onset k0 (D-013)

$$
k_0 = \min\{\,k : \max_{i \in \text{fit},\, f} Y_{i,k,f} > 0\,\}
$$

- `onset_index` raises `ContractError` if every training value is 0, or if T_fit − k0 < 3. Source: assumption (D-013). Three steps give DMD at least two snapshot pairs.
- k0 comes from the fit cases only (V1). It is saved in `meta.json`.
- Every model returns exactly 0 for k < k0, and fits only the dates k0..T_fit − 1.
- The report states how many validation values with k < k0 are > 0. Expected: 0 (C10).
- Real data: k0 = 4, because production starts in 1998-Q4 (task1 §2.5).

### 2.3 Time variable (RNN and LSTM)

$$
t_k = \frac{k}{T_{\mathrm{fit}} - 1}
$$

Source: PRD §6 ("t = step index ÷ 40" when T = 41). T_fit is read from Y at fit time (V5).

- **Training range:** t ∈ [k0 ÷ (T_fit − 1), 1].
- **Continuation:** k ≥ T_fit gives t > 1. With T_fit = 41, the 80 forecast steps give t = 1.025 to 3, as in PRD §6.
- **Backtest** (`t_cut` = 29): t = k ÷ 28, and the 12 backtest steps reach t = 40 ÷ 28 ≈ 1.43.

### 2.4 Loss and fluid weights (D-012)

$$
\mathcal{L}(\theta) = \sum_{f} w_f \,\frac{1}{|\mathcal{B}|\, T_p} \sum_{i \in \mathcal{B}} \sum_{k=k_0}^{T_{\mathrm{fit}}-1} \big(\hat y_{i,k,f}(\theta) - Y_{i,k,f}\big)^2, \qquad \sum_f w_f = 1
$$

- **MSE:** source Article 2, p. 985, Eq. 1, and PRD §6.
- **Weights:** source D-012.
  - Default `d012` = (0.25, 0.25, 0.5).
  - Tuned option `equal` = (1/3, 1/3, 1/3). It equals the PRD's plain MSE over the three fluids.
  - The config validation normalises each set to sum 1 after checking |Σ − 1| ≤ 1e-9.
- ℬ is the minibatch, and ŷ is the raw (unclipped) network output. Assumption: clipping inside the loss would remove the gradient of negative outputs.

### 2.5 Validation objective J (D-015)

For predictions Ŷ (clipped, §2.12) of the n cases of a split, over all T_fit dates (zeros before k0 included, as in PRD §7):

$$
\mathrm{RMSE}_f = \sqrt{\frac{1}{n\,T_{\mathrm{fit}}}\sum_{i,k}\big(\hat y_{i,k,f}-y_{i,k,f}\big)^2}, \qquad
\mathrm{NRMSE}_f = \frac{\mathrm{RMSE}_f}{\max_{i,k} y_{i,k,f} - \min_{i,k} y_{i,k,f}}, \qquad
J = \frac{1}{|F_J|}\sum_{f \in F_J} \mathrm{NRMSE}_f
$$

- **Source:** PRD §7 (the RMSE formula, NRMSE, and selection rule step 3). F_J = `objective.fluids` = {oil, water}.
- **Scale invariance:** NRMSE does not change with the scale of the fluid, so J is the same in scaled and physical units. Reports state the RMSE in physical units (PRD §5).
- **Uses of J:**
  - Early stopping (§2.7).
  - Trial ranking (§2.14).
  - Task 3 model ranking.
- **Caveat:** the validation J of a final model is optimistically biased, because the same cases chose the epoch and the settings. Task 3 adds the backtest, and the test split after the freeze.

### 2.6 RNN and LSTM networks

**Parameter input.** u = 2x − 1 ∈ [−1, 1]^p (`x_affine: true`). Source: Article 2, p. 989 (min-max normalisation of the parameters to (−1, 1)).

**Steps.** k = k0..K − 1, with K = T_fit for `predict` and K = n_total for `rollout`.

| Option | Input z_k at each step | Initial state | Input size | Source |
| --- | --- | --- | --- | --- |
| A | [u, t_k] | h = 0, c = 0 | p + 1 | PRD §6. Article 2, p. 984: the LSTM takes 11 inputs at each of its 100 steps (11 × 100). Table 1 (p. 989) lists 10 parameters, so our reading is parameters plus time at each step |
| B | [t_k] | For each layer l: h₀⁽ˡ⁾ = tanh(W_h⁽ˡ⁾ u + b_h⁽ˡ⁾), and c₀⁽ˡ⁾ = W_c⁽ˡ⁾ u + b_c⁽ˡ⁾ (LSTM only) | 1 | PRD §6. The tanh on h₀ is an assumption: tanh RNN, GRU and LSTM hidden states lie in (−1, 1). Implementation: one `Linear(p, num_layers·H)` for h₀ (and one for c₀), reshaped to (num_layers, n, H) |
| C | [u] | h = 0, c = 0 | p | PRD §6 |

**Cell.**

- `torch.nn.LSTM`, `torch.nn.GRU`, or `torch.nn.RNN(nonlinearity='tanh')`.
- Settings: `num_layers`, `hidden_size`, `batch_first=True`, `dropout=0`, `bias=True`.
- Gate equations: PyTorch 2.14 documentation; Hochreiter and Schmidhuber (1997) for the LSTM; Cho et al. (2014) for the GRU.
- `rnn` tunes `cell` ∈ {rnn, gru} (PRD §6: "`nn.RNN` or `nn.GRU`"). A GRU is a simplified LSTM with fewer parameters (Article 6, p. 8). `lstm` always uses `nn.LSTM`.

**Output.** ŷ_k = W_o h_k⁽ᴸ⁾ + b_o ∈ ℝ³: a linear layer with 3 units (source: PRD §6). Article 2 also puts fully connected layers between the LSTM and the regression output (p. 991, Fig. 8, Table 3; p. 994–995). This spec keeps the single linear layer. Assumption: we have at most 64 units and 70 cases, while Article 2 had about 580 units and 300 cases. A dense head can become a tuned option later, with a spec change.

**Other settings.**

- **Initialisation:** PyTorch defaults (all weights U(−√k, √k), k = 1/hidden_size; PyTorch 2.14 documentation), after `torch.manual_seed(seed)`.
- **Precision:** parameters and computation in `training.dtype` (float32). Outputs are cast to float64 at the boundary.
- **Option C:** the input is constant over the steps, so the hidden state carries the step count.
  - In the continuation, the recurrence runs past the trained length (extrapolation in state).
  - Options A and B see t > 1 instead (extrapolation in input).
  - The backtest compares them (PRD §9).

### 2.7 Training one member (`train_member`)

1. **Setup:**
   - `torch.manual_seed(seed)` and `rng = np.random.default_rng(seed)`.
   - Build `SeqNet`.
   - Adam with `lr`, betas (β₁, β₂), `eps` and `weight_decay` from the config.
2. **Epoch loop**, for e = 1..`max_epochs`:
   - `order = rng.permutation(n_fit)`.
   - Consecutive minibatches of `batch_size` cases; the last one may be shorter. A `batch_size` ≥ n_fit gives full-batch training.
   - For each minibatch: forward pass over k0..T_fit − 1, loss (§2.4), backward, `clip_grad_norm_(parameters, grad_clip_norm)` (skipped if null), step, zero the gradients.
   - A non-finite loss raises `TrialFailed`.
3. **After each epoch** (network in eval mode, no gradient):
   - Compute J_e (§2.5) on the clipped validation outputs over k = 0..T_fit − 1.
   - If J_e < J_best − `min_delta`: set J_best = J_e and `best_epoch` = e, and deep-copy the state_dict.
   - Stop when e − `best_epoch` ≥ `patience`.
4. **Return:** load the best state. Return `MemberResult`, with the history (epoch, mean minibatch loss, J_e).

**Sources.**

- Adam: Kingma and Ba (2015). Article 2 trains with minibatch stochastic gradient descent (p. 984–985), and the PRD does not name an optimiser. Adam is an assumption: a standard choice, with little sensitivity to the step size.
- Early stopping on cases 71 to 85: PRD §6; Prechelt (1998).
- Gradient-norm clipping: Pascanu et al. (2013). The threshold 1.0 is an assumption: a guard against exploding gradients in recurrent networks.

### 2.8 Ensemble, `predict` and `predict_std` (RNN, LSTM)

Members m = 1..M (M = `len(seeds)` ≥ 2). g_m(x, k) is the raw output of member m.

$$
\mu(x,k) = \frac{1}{M}\sum_{m} g_m(x,k), \qquad \hat y = \max(\mu, 0), \qquad s(x,k) = \sqrt{\frac{1}{M-1}\sum_m \big(g_m(x,k) - \mu(x,k)\big)^2}
$$

- Source: PRD §6 ("The prediction is the mean. The spread is `predict_std`"). Deep ensembles: Lakshminarayanan et al. (2017).
- ddof = 1: assumption (an unbiased variance estimate for M = 5).
- Both are 0 for k < k0.
- Inference runs in chunks of `predict_batch` rows under `torch.inference_mode()`.

### 2.9 DMD (D-013, D-016, D-019)

Settings: `rank`, `kernel`, `fluid_weights` (w), `gp_target`, plus the fixed `dmd` and `gp` blocks.

**Amendment A2 (D-019).** With `dmd.basis: real` (the default), step 3 sets r = min(`rank`, q, `r_max`), and §2.9a replaces steps 4 and 5. With `dmd.basis: exact`, steps 3 to 5 apply as written, with r = min(`rank`, q, `r_max`) in place of the energy rule. Steps 1, 2 and 6 to 11 apply to both bases. The energy rule of step 3 still gives `r_99` (energy `dmd.energy_report` = 0.99) for the report.

**Step 1. Snapshot matrix** (PRD §6 step 1, with D-013 and D-012). For f = 0..2, i = 0..n_fit − 1 and j = 0..T_p − 1:

$$
D_{f\,n_{\mathrm{fit}}+i,\; j} = \sqrt{w_f}\; Y_{i,\,k_0+j,\,f}
$$

The row-weighted squared Frobenius norm gives fluid f the weight w_f, as in the loss (§2.4).

**Step 2. Snapshot pairs.** X₁ = D[:, 0:T_p − 1] and X₂ = D[:, 1:T_p]. Assumption (D-016): one step = one quarter.

- Quarter lengths are 90 to 92 days, at most 2.2% from their mean.
- R1 rates are quarter means, so the step index is the natural clock.
- The recurrent networks use the same index.
- The report gives the minimum and maximum Δt_days.

**Step 3. Rank** (`energy_rank`):

- Thin SVD X₁ = U Σ Vᵀ (`np.linalg.svd(full_matrices=False)`).
- q = #{σ_i > `sigma_rel_tol` · σ₁} (numerical rank).
- r = min{ρ ≤ q : Σ_{i≤ρ} σ_i² ÷ Σ_{i≤q} σ_i² ≥ energy}, then r = min(r, `r_max`).
- energy = 1.0 gives r = q.
- Source: PRD §6 ("Rank r for 99% of the energy"). The energy as σ² follows Article 4, p. 1106 (POD vectors kept up to a share of the explained variance, for example 95%).

**Step 4. Exact DMD** (`exact_dmd_eigs`; source: Tu et al. 2014, Algorithm 2):

$$
\tilde A = U_r^{\mathsf T} X_2 V_r \Sigma_r^{-1} \in \mathbb{R}^{r\times r}, \qquad \tilde A\, W = W \Lambda, \quad \lambda = \operatorname{diag}(\Lambda)
$$

- `np.linalg.eig`.
- Only λ is kept (PRD §6 step 2). The modes are not needed, because step 7 fits the coefficients of each curve. Algorithms 1 and 2 of Tu et al. share these eigenvalues.

**Step 5. Eigenvalue cleaning** (`clean_eigs`; assumption, D-016). Ã is real, so λ is closed under conjugation.

1. **Real or complex:** λ is real if |Im λ| ≤ `imag_tol` · max(1, |λ|). Then set Im λ = 0.
2. **Stabilise** (`unstable: clip`): if |λ| > 1 + `stab_tol`, set λ ← λ ÷ |λ|. Record `n_stabilised` and `max_abs_lam_raw`.
   - Reason: a mode with |λ| > 1 grows without bound. For example, 1.02^116 ≈ 9.9 over τ = 0..116 (36 history and 80 forecast steps on the real data).
   - The field produces from fixed wells with no new development (assumption), so the rates cannot grow without bound.
   - The step comes before the least squares, so the history fit and the forecast use one basis ("the same fitted state", PRD §6).
   - `keep` leaves λ unchanged. It is for experiments only, and the report flags it.
3. **Representatives:** each real λ, and each complex λ with Im λ > 0.
   - Each complex representative needs a partner with |λ̄ − λ_partner| ≤ 1e-8 · max(1, |λ|). Otherwise raise `ContractError`.
4. **Order:** by descending |λ|, then ascending |arg λ|.
5. **Merge duplicates** (after the order; review 1): representatives within `dup_tol` of an earlier one in the order are dropped. Record `n_merged`. This can happen after stabilisation.

`n_stabilised` counts eigenvalues: a stabilised conjugate pair counts 2 (review 1).

**Step 6. Real basis** (`real_basis`). m representatives give r_eff = n_real + 2 n_pair columns:

$$
b(\tau) = \Big[\;\lambda_j^{\tau}\ \ (\text{real } j);\quad \rho_j^{\tau}\cos(\tau\theta_j),\ \rho_j^{\tau}\sin(\tau\theta_j)\ \ (\text{pair } j,\ \lambda_j = \rho_j e^{\mathrm{i}\theta_j},\ 0<\theta_j<\pi)\;\Big]
$$

- 0⁰ = 1, and negative real λ is allowed (τ is an integer).
- **Equivalence with PRD §6 step 5.** Let the conjugate eigenvalue carry the conjugate coefficient. This holds for the least-squares fit of a real curve on a conjugate-closed basis. Then:

$$
\mathrm{Re}\sum_{k} c_k \lambda_k^{\tau} = \sum_{\text{real } j} c_j \lambda_j^{\tau} + \sum_{\text{pairs } j} \big[\,2\,\mathrm{Re}(c_j)\,\rho_j^{\tau}\cos\tau\theta_j - 2\,\mathrm{Im}(c_j)\,\rho_j^{\tau}\sin\tau\theta_j\,\big]
$$

  So the real coefficients a = (c_real, 2 Re c, −2 Im c) are a fixed linear map of the Re and Im parts in PRD §6 step 4.
- Assumption (D-016): the real form avoids the redundant imaginary outputs, and its prediction is real by construction.

**Step 7. Least-squares coefficients** (PRD §6 step 3).

- B = [b(0); …; b(T_p − 1)] ∈ ℝ^{T_p × r_eff}.
- For every training curve: a_{i,f} = argmin_a ‖B a − Y_{i, k0:T_fit, f}‖².
- One call: `np.linalg.lstsq(B, Ymat, rcond=None)`, on the unweighted scaled curves. The fluid weights act only through step 3.
- Record cond(B), and the residual variance ē_f(τ)² = (1/n_fit) Σ_i (B a_{i,f} − y_{i,f})_τ².

**Step 8. GP targets** (`gp_target`, tuned; D-016):

- `coef` (PRD §6 step 4): g_{i,f} = a_{i,f}, with M = I.
- `proj`:
  - B = QR (reduced QR) and g_{i,f} = Qᵀ y_{i,f} = R a_{i,f}, with M = R⁻¹.
  - The GPs then regress orthonormal projections, as Article 4 does with its POD components (p. 1106, Eq. 17). Independent GP errors then reach the history-window prediction through Q without amplification.
  - If min_j |R_jj| ÷ max_j |R_jj| < 1e-12, raise `TrialFailed`.

**Step 9. GP** (PRD §6 step 4). One `GaussianProcessRegressor` for each fluid f and target column j, from X_fit to g_{·,f,j}: 3 · r_eff GPs. Source: Article 4, p. 1106, Eq. 17 (one meta-model built independently for each basis component).

Kernel:

$$
k(x,x') = C\cdot \mathrm{Mat\acute ern}_\nu\!\Big(\sqrt{\textstyle\sum_d (x_d-x'_d)^2/\ell_d^2}\Big) + \sigma_n^2\,\delta_{xx'}
$$

| Kernel parameter | Value | Source |
| --- | --- | --- |
| `kernel` | `matern25` (ν = 2.5, default) or `matern15` (ν = 1.5), with one length scale ℓ_d for each parameter (ARD) | Article 4, p. 1103: Matérn 5/2 with one range for each parameter. Article 4 uses a product of 1-D Matérn kernels. scikit-learn's anisotropic `Matern` uses the scaled Euclidean distance above. Assumption: we use scikit-learn's form |
| C | `ConstantKernel(1.0, constant_bounds)` | Assumption |
| ℓ_d | `length_scale=np.ones(p)`, `length_scale_bounds` (1e-2, 1e2) in scaled X | Assumption |
| σ_n² | `WhiteKernel(noise_init, noise_bounds)` | Assumption: the coefficients carry fit noise from steps 6 and 7 |
| Hyperparameters | Maximum log marginal likelihood (L-BFGS-B), `n_restarts_optimizer` = 5 restarts from log-uniform points within the bounds, `random_state = cfg['seed']` | Article 4, pp. 1103–1104, Eq. 4–7 (maximum likelihood); scikit-learn 1.7 documentation and source (`_gpr.py`) |
| `normalize_y` | `true`: targets are standardised before the fit, and the mean and variance are mapped back | scikit-learn 1.7 source. Assumption: stands in for the constant regression mean h(x)ᵀβ of Article 4, Eq. 2 |
| `alpha` | 1e-10 (scikit-learn default) | scikit-learn 1.7 documentation |

- The GP fits capture `ConvergenceWarning`s and count them for the report.
- The GP maps X to the coefficients only. Time stays in the basis (PRD §6 step 4).

**Step 10. Prediction** (PRD §6 step 5):

$$
\hat y_f(x,k) = \begin{cases} 0 & k < k_0\\ b(k-k_0)^{\mathsf T} M\, \hat g_f(x) & k \ge k_0 \end{cases}
$$

- ĝ_f(x) is the vector of GP means.
- The same formula gives the continuation (Task 5, F2). Then clip (§2.12).

**Step 11. `predict_std`** (D-016):

$$
s_f(x,k)^2 = \sum_j \Big(\big[b(k-k_0)^{\mathsf T} M\big]_j\Big)^2 \hat\sigma_{f,j}(x)^2 \;+\; \bar e_f(k-k_0)^2 \quad (k_0 \le k < T_{\mathrm{fit}}),\qquad s_f = 0 \ (k<k_0)
$$

- σ̂_{f,j} is the GP standard deviation (`predict(return_std=True)`). It includes the `WhiteKernel` variance, because scikit-learn computes it from `kernel_.diag(X)` (scikit-learn 1.7 source).
- The GPs are treated as independent (Article 4, p. 1106: built independently).
- The residual term ē² is included when `std_includes_basis_residual` is true. Assumption: without it, the std ignores the basis error, which is largest on the rate-limited plateau (PRD §6 step 6).
- `predict_std` covers the fitted window only. Task 5 does not need a forecast std.

### 2.9a Amendment A2: optimized DMD with real eigenvalues (D-019)

**Problem (review 1).** Run `train-20261012T235740838203Z` used exact DMD at r = 4. It gave two conjugate pairs. The slow pair (|λ| = 0.9936, period 399 quarters) is about 0.58 rad of a cycle over the 37 fitted quarters. The basis therefore models the oil decline as part of an oscillation. Continued, the oil went below 0 at k = 47 to 59 for all 15 validation cases: 2,022 values were clipped, and 84% of the forecast oil values were 0.

- An oscillation with a period of more than twice the fitted window is not observed as an oscillation. Its continuation has no support in the data (assumption).
- The field has no new wells after 2008 (assumption, D-016; the question is still open with the user). Its decline is then of Arps type (source: Arps 1945): exponential q_i e^{−Dt}, or hyperbolic q_i (1 + bDt)^{−1/b}.
- The hyperbolic form is a positive mixture of exponential declines: (1 + bDt)^{−1/b} = ∫₀^∞ e^{−st} g(s) ds, with g the gamma density of shape 1/b and scale bD. Source: derivation (the Laplace transform of the gamma density).
- In quarter steps, e^{−sτ} = λ^τ with λ = e^{−s} ∈ (0, 1]. Real eigenvalues in (0, 1] thus span the decline forms of reservoir engineering, and they cannot oscillate.

**Step 4R. Real eigenvalues by variable projection** (`basis: real`; replaces steps 4 and 5).

1. **Data.** Z = Dᵀ ∈ ℝ^{T_p × 3 n_fit}: the weighted snapshot matrix of step 1, with time down the rows. All T_p snapshots are used; the pairs of step 2 are not needed.
2. **Model.** Z ≈ B(λ) C, with B(λ)_{τj} = λ_j^τ, τ = 0..T_p − 1 and j = 1..m, where m = r (step 3).
3. **Parameters.** κ_j = d_1 + … + d_j (decay rates) and λ_j = exp(−κ_j).
   - Bounds (boxes on d): d_1 ∈ [0, κ_max], and d_j ∈ [δ, κ_max] for j ≥ 2. Here κ_max = `real_kappa_max` and δ = `real_kappa_sep`.
   - Thus 1 ≥ λ_1 > λ_2 > … > λ_m > 0, and e^{−κ_max} ≤ λ_{j+1}/λ_j ≤ e^{−δ}. Each neighbour is at least 0.5% smaller, so the λ are distinct by construction.
   - The duplicate merge of step 5 is not applied (n_merged = 0), so r_eff = m for every grid point. This fixes the duplicate grid points of review 1.
   - A numerically singular basis raises `TrialFailed`: under `proj`, by the check of step 8; under `coef`, if cond(B) > 1e12.
   - λ = 1 is allowed (d_1 = 0). It is the constant mode, which a water rate needs to level off.
   - Source: assumption. Optimized DMD constrains the eigenvalues within its optimisation (Askham and Kutz 2018); here the constraint is real λ in (0, 1].
4. **Objective** (variable projection: Golub and Pereyra 1973; optimized DMD: Chen, Tu and Rowley 2012; Askham and Kutz 2018):

$$
\min_{d}\ \big\| Z - B(\lambda(d))\, C^\star(d) \big\|_F^2, \qquad C^\star(d) = \arg\min_C \|Z - B(\lambda(d))\,C\|_F = \texttt{lstsq}(B, Z)
$$

   - `scipy.optimize.least_squares(fun, d0, bounds=(lo, hi), method="trf", jac="2-point", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)`.
   - `fun(d)` = (B C* − Z).ravel(). It is deterministic (V7).
5. **Starts.** Run the starts in this order. Keep the result with the lowest final cost; a tie goes to the earlier start.
   - **S0 (exact DMD).** Take λ_exact = `exact_dmd_eigs(D, m)` (step 4). Set κ⁰ = sort ascending of −ln(clip(|λ_exact|, e^{−κ_max}, 1)).
   - **S1 to S3.** κ⁰_j = a (b/a)^{(j−1)/(m−1)}, log-spaced, for (a, b) = (1e-3, 0.3), (1e-2, 1.0) and (3e-3, 3.0). For m = 1, κ⁰ = a.
   - **Feasibility of each start.** κ⁰_1 ← max(κ⁰_1, 0) and κ⁰_j ← max(κ⁰_j, κ⁰_{j−1} + δ). Then d0 = (κ⁰_1, diff κ⁰), clipped to the bounds.
   - The exact-DMD eigenvalues of S0 are recorded as `lam_exact` (PRD §6 step 2 stays as the first start).
6. **Output.** reps = λ (complex128 with imaginary part 0, in descending order), kinds = 0. `info` has n_stabilised = 0, n_merged = 0, max_abs_lam_raw = max |λ_exact|, and `varpro` {cost, start_costs, best_start, nfev, status, d_at_bound}.
   - Steps 6 to 11 then run unchanged.
   - `varpro_real_eigs` re-raises any exception from `least_squares` as `TrialFailed`, so the trial is recorded as failed (§2.14), not the run.
   - A non-finite cost or λ in every start raises `TrialFailed`.
   - `status` 0 (`max_nfev` reached) is not a failure. It is recorded in `varpro.status`, and the report shows it.

**Guarantees.**

- Each fitted or continued curve is Σ_j c_j λ_j^τ with real λ_j ∈ (0, 1]. It does not oscillate.
- It has at most m − 1 sign changes in τ. Source: derivation. Multiply by λ_1^{−τ}, then apply Rolle's theorem m − 1 times; each step removes one term.
- As τ → ∞, the sign of each curve is the sign of the coefficient of its slowest mode. This is not forced to be positive, so the gate of §2.14 checks it on the validation cases.

**Not guaranteed.** The history fit can be worse than the exact basis at the same r, because the real basis cannot bend a curve by rotation. Task 3 applies V6 (B0, B1) to the result as usual.

### 2.10 Plateau and breakthrough errors (PRD §6 step 6)

**Periods for each case i**, on the physical true rates:

- **Plateau end** (PRD §7 event definition, `plateau_frac` = 0.98): k_end,i = the first k ≥ k0 with q_oil,i(k) < 0.98 · max_k q_oil,i(k). If there is none, k_end,i = T_fit.
- **Plateau:** P_i = [k0, k_end,i). **Post-plateau:** [k_end,i, T_fit).
- **Water breakthrough** (`breakthrough_frac` = 0.01): k_bt,i = the first k with q_w,i(k) > 0.01 · max_k q_w,i(k).
- **Pre- and post-breakthrough:** [k0, k_bt,i) and [k_bt,i, T_fit).

**What is reported:** the RMSE in physical units of each fluid over the union of each period across the cases, for three sets:

- (a) DMD only: the least-squares reconstruction B a_{i,f} on the train cases. This is the error of the linear-dynamics basis alone.
- (b) The predictions on the train cases.
- (c) The predictions on the validation cases.

Report only. It is computed for every model, so that the DMD plateau error has a comparison.

### 2.11 Benchmarks B0 and B1 (D-017)

| Benchmark | Prediction | Source |
| --- | --- | --- |
| B0 `MeanCurveBenchmark` | ŷ(x, k, f) = (1/n_fit) Σ_{i∈fit} Y_{i,k,f}, the same for every x | PRD §6, V6 |
| B1 `KNNBenchmark` | `KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='brute', metric='euclidean')`, from scaled X to Y reshaped to (n, T_fit · 3). With weights 1/d_j, a query at distance 0 from a training point returns that point's curve (scikit-learn behaviour) | PRD §6. `brute` is an assumption: exact and deterministic |

- Both have `extrapolates_time = False`.
- `forecast`, and `rollout` with n_total > T_fit, raise `NotImplementedError("benchmark: no time continuation (PRD §6)")`.
- `predict_std` returns None.
- They are in `BENCHMARKS`, never in `PROXIES`. `load_model(..., proxies_only=True)` refuses them, so they cannot reach Tasks 4 and 5 (PRD §6, V13).
- Their outputs are non-negative by construction. The clip and its count still run, with count 0.

### 2.12 Negative rates (D-017)

- `rollout`, `predict` and `forecast` apply ŷ ← max(ŷ, 0) to the raw output for k ≥ k0. They store the number of values changed in `last_n_clipped`.
- This happens in scaled units. Because s_f > 0, the result is bit for bit the clip after the inverse transform (PRD §6): max(y, 0) · s_f = max(y · s_f, 0).
- V9 needs rates ≥ 0 at the `predict` boundary (§1.8), so `predict` returns clipped values.
- **Logging.** The pipeline logs and stores the counts for:
  - The train predictions.
  - The validation predictions.
  - The validation continuation (`forecast_steps`).
  - Tasks 4 and 5 log their own counts.
- The ensemble clips the mean, not the members (§2.8). The loss uses raw outputs (§2.4).

### 2.13 Rollout, forecast and the truncated window

**Rollout.**

- `rollout(X, n_total)` returns the steps k = 0..n_total − 1 from the fitted state, with 0 for k < k0.
- `predict(X) = rollout(X, T_fit)`.
- `forecast(X, n_steps) = rollout(X, T_fit + n_steps)[:, T_fit:]`.
- Task 5 F1 (121 steps, keep steps 42 to 121) is `forecast(X, 80)`, and so is F2 (PRD §9). There is no refit (PRD §6).

**Truncated window (Task 5 backtest).**

- `load_dataset(con, t_cut)` slices the fit and validation curves to the first `t_cut` dates.
- Early stopping therefore sees no date after the cutoff.
- `fit` takes T_fit = `t_cut` from `Y.shape[1]`. The time scaling (§2.3), k0 and the DMD window all follow from it.
- Task 5 converts its cutoff date into `t_cut` = the number of history dates ≤ the cutoff (29 for 1 Jan 2005).

**`--settings-from SRC`.**

- Reads the `runs` row of SRC. It must have status `ok` and the same `model`.
- Reads `outputs/models/SRC/<variant>/meta.json` for each variant of SRC.
- Calls `train_model(..., settings={variant: meta['settings'] tuned values})`. There is no tuning.
- Each variant trains all its `seeds` directly, and `trials.csv` gets one row for each variant.
- `runs.outputs.settings_from` = SRC.
- Known limitation, to report in Task 5: the SRC settings were chosen with full-window validation data, so the backtest hyperparameters are not blind to the post-cutoff dates. PRD §9 accepts this ("with the Task 2 settings").

### 2.14 Tuning (D-015)

**Plan** (`make_plan`).

- **`lhs` (RNN, LSTM):**
  - The strata are `network.input_options`. N = `n_trials` ÷ (number of strata) must be an integer. Otherwise `ConfigError`.
  - For stratum s (config order), `rng = np.random.default_rng([cfg['seed'], s])`.
  - For each dimension d in config order: `perm = rng.permutation(N)`, `jit = rng.random(N)`, `u[:, d] = (perm + jit) / N`. This is a Latin hypercube. Source: McKay et al. (1979).
  - Mapping:
    - `choice` with m values: `values[min(floor(u·m), m−1)]`.
    - `loguniform`: exp(ln low + u (ln high − ln low)).
  - Trial IDs run 0..`n_trials` − 1, stratum by stratum.
- **`grid` (DMD):** the Cartesian product of the `choice` dimensions in config order (`itertools.product`, last dimension fastest). It needs grid size ≤ `n_trials`. The deliverable grid (A2) has rank 6 × kernel 1 × weights 2 × target 2 = 24 points. `rank` values are distinct integers ≥ 1.
- **B0, B1:** no plan.

**Why a NumPy Latin hypercube, not optuna (D-015).**

- Article 2 tuned its LSTM with Bayesian optimisation (p. 985; Table 3, p. 991).
- With 10 trials in each stratum, optuna's TPE sampler would sample at random throughout: its default `n_startup_trials` = 10 uses random sampling first (optuna documentation). A new dependency would add nothing.
- Random search is efficient when few dimensions matter (Bergstra and Bengio 2012).
- The Latin hypercube spreads each dimension evenly over 10 trials (McKay et al. 1979).
- The plan is fixed before any fit, so it is reproducible (V7) and can run in parallel.

**Resolved settings** (`resolve_settings`). The flat fit cfg is the tuned values plus:

- **Neural:**
  - `cell` (`lstm` for LSTM; tuned for RNN).
  - `input_option` = the variant, and `fluid_weights` resolved to its 3 numbers, with `fluid_weights_name`.
  - `seeds`, `max_epochs`, `patience`, `min_delta`, `grad_clip_norm`, `adam`, `dtype`, `x_affine`, `predict_batch`.
  - `objective_fluids`.
- **DMD:** the tuned values, plus `seed`, the `dmd` block and the `gp` block.
- **B1:** `knn`. **B0:** {}.

**Trial evaluation.**

- **Neural:**
  - Train `seeds[:tune_seeds]` with `train_member`.
  - Score = J of the clipped mean of these members on the validation cases (§2.5).
  - Tasks are (trial, seed) pairs. With `n_jobs` > 1, they run in a `ProcessPoolExecutor` (spawn context). Each worker sets `torch.set_num_threads(torch_threads)` and deterministic algorithms.
  - Results are gathered in task order, so they do not depend on `n_jobs`.
- **DMD:** one `DMDProxy.fit` for each grid point. Score = J of `predict(X_val)`.
- **Failures:** a trial that raises `TrialFailed`, `np.linalg.LinAlgError` or `ContractError` from the model is recorded with `status = 'failed'`, j_val = +inf and the error text. The run fails only if every trial of a variant fails.

**Selection and final model.**

- **Neural:**
  - For each variant, the best trial = argmin j_val (ties go to the lower trial ID).
  - The final model = `from_members` over all `seeds`, with the best trial's settings.
  - The members already trained in that trial are reused. Determinism makes them identical to a new fit.
  - There is no refit on train + validation (PRD §6).
- **DMD, with the plausibility gate P1 (D-020):**
  - For each ok trial, compute F = `forecast(X_val, report.forecast_steps)`, and set `n_clip_cont` = `last_n_clipped` (all fluids).
  - `plausible` = (`n_clip_cont` == 0). Reason: a rate below 0 in the continuation is physically impossible, and validation J cannot see it (D-014). The gate uses validation X only (V1).
  - Best = argmin (j_val, trial) over the plausible ok trials. Its fitted object is the final model.
  - If no ok trial is plausible, take the argmin over all ok trials and set `plausible_selected` = false. Log a warning, and state it in report §11: "no plausible grid point; Task 3 rejects this proxy under P1". The run still ends `ok`.
  - `trials.csv` and the report §3 show `basis`, `n_clip_cont` and `plausible` for each trial.
- **Default (D-015):** `tune_seeds` = 1.
  - Reason: 30 trials × 5 seeds = 150 fits for each model, against 30 + 3 × 4 = 42.
  - Cost: the trial score is a single-seed estimate of the ensemble's J, which is noisier.
  - The report shows the trial J and the ensemble J side by side.
  - `tune_seeds: 5` is allowed, and is the faithful option when time permits.

### 2.15 Speed benchmark (PRD §6)

**Budget.** `limit_s` = 600 s for 160,000 parameter sets (32 walkers × 5,000 steps, PRD §6, §8). With `vectorize=True`, the MCMC makes one call of 32 rows per step (PRD §8). The per-call budget is therefore `limit_s` ÷ `mcmc_calls` = 0.12 s.

**Pipeline measurement** (each proxy variant, after the final fit; X from `np.random.default_rng(cfg['seed']).random((rows, p))`):

- t_batch: wall time of one `predict` call on `n_batch_rows` = 10,000 rows (PRD §7 cost metric).
- t_call: mean wall time of `n_calls` = 200 calls of `predict` on `call_rows` = 32 rows, after 5 warm-up calls.
- Extrapolations:
  - MCMC: `mcmc_calls` · t_call.
  - Batch: 16 · t_batch.
- `pass` = both extrapolations < `limit_s`. Report only. Tests M38 and M39 enforce it.

### 2.16 Review 1 notes (D-021)

The data scientist resolved these points. Each is accepted, and this spec now states it:

- **Seeds.** Every seed (plan, members, GP) is an integer in [0, 2³² − 1], because NumPy `SeedSequence` and scikit-learn `random_state` require it.
- **`n_jobs`.** The deliverable is 6 in `rnn.yaml` and `lstm.yaml`, within the allowed 1 to 16. M17 shows that the results do not depend on it. The runs are reproducible on fewer cores, because the pool size does not change any result.
- **`forecast`.** `last_n_clipped` counts the returned steps only.
- **Failures.**
  - Every trial of a variant failing raises `ContractError` (exit 1).
  - So does a failure of a final ensemble member. There is no fallback to the next trial, because the ensemble would then mix settings.
- **GOR check.** The report compares with the training-case range and with the C7 range.
- **Config keys.** Unknown config keys raise `ConfigError`.
- **DMD cleaning.** Duplicates are merged after the ordering (§2.9 step 5). `n_stabilised` counts eigenvalues.
- **`meta.json`.** The extra keys `members` (neural) and `diagnostics` (DMD) are allowed.
- **Lock.** `cloudpickle` 3.1.2 is in `requirements-lock.txt`, because the lock is the `pip freeze` of `.venv`.

Review 2 (A2 rework; D-022) accepts these as well:

- The cond(B) > 1e12 guard applies to the real basis under `coef` only (§2.9a).
- M23 uses a 1e-12 relative-plus-absolute tolerance for both bases.
- M44 ignores round-off at or below 1e-12 × the curve maximum when it counts sign changes and extrema.
- For `basis: real`, `dmd.npz` `lam_raw` holds λ_exact (the S0 start).
- `varpro` and the diagnostics may carry extra keys (`d`, `nfev_per_start`, `status_per_start`, `n_plausible_trials`, `n_clip_cont`).

## 3. Defaults and tuning ranges

`load_model_config` enforces the "Allowed" column. A value outside it raises `ConfigError`. A change outside the allowed range needs a new entry in `docs/decisions.md`.

### 3.1 RNN and LSTM: tuned settings

| Setting | Default (untuned fit) | Search (deliverable) | Allowed | Source |
| --- | --- | --- | --- | --- |
| `input_option` (stratum) | A | A, B, C (10 trials each) | Non-empty subset of {A, B, C} | PRD §6; D-014 |
| `num_layers` | 1 | {1, 2} | ⊆ {1, 2} | PRD §6. Article 2, p. 994 (architectures with 1 or 2 LSTM layers) |
| `hidden_size` | 32 | {16, 32, 48, 64} | Values in [16, 64] | PRD §6 |
| `lr` | 1.5e-3 | log-uniform [3e-4, 1e-2] | 1e-5 ≤ low < high ≤ 1e-1 | Article 2, Table 3, p. 991: range 1e-5 to 0.4, optimum 0.00144. The narrower range is an assumption, typical for Adam |
| `batch_size` | 8 | {4, 8, 16, 32} | Values ≥ 1 | Article 2, p. 995: small batches gave the lowest error (optimum 2 of 300 cases) |
| `fluid_weights` | `d012` | {d012, equal} | Keys of `fluid_weight_sets` | D-012 |
| `cell` (RNN only) | gru | {rnn, gru} | ⊆ {rnn, gru}; not allowed in `lstm.yaml` | PRD §6. Article 6, p. 8 |

### 3.2 RNN and LSTM: fixed settings

| Setting | Default | Allowed | Source |
| --- | --- | --- | --- |
| `seeds` | [0, 1, 2, 3, 4] | 2 to 10 distinct integers. The deliverable uses 5 | PRD §6 |
| `tune_seeds` | 1 | 1 ≤ value ≤ len(seeds) | D-015 |
| `max_epochs` | 1000 | 1 to 5000 | Assumption, with early stopping. Article 2 searched 30 to 500 and chose 249 (Table 3, p. 991) |
| `patience` | 50 | 1 to 500 | Assumption (Prechelt 1998) |
| `min_delta` | 0.0 | 0 to 0.01 | Assumption |
| `grad_clip_norm` | 1.0 | null, or 0.1 to 10 | Pascanu et al. (2013) |
| `adam` | β₁ 0.9, β₂ 0.999, eps 1e-8, weight decay 0 | β in (0, 1), eps > 0, weight decay 0 to 1e-2 | Kingma and Ba (2015) defaults |
| `dtype` | float32 | float32, float64 | Assumption: speed. Outputs are float64 |
| `x_affine` | true | true, false | Article 2, p. 989 |
| `torch_threads` | 1 | 1 to 16 | Assumption: small networks gain little from threads, and 1 thread keeps the results reproducible |
| `n_jobs` | 6 in the deliverable `rnn.yaml` and `lstm.yaml` (review 1); 1 in the test configs | 1 to 16 | Assumption. Results do not depend on it (M17) |
| `predict_batch` | 2048 | 256 to 65,536 | Assumption: bounds memory for 160,000 rows |
| `n_trials` | 30 | 1 to 30, divisible by the number of strata | PRD §6 |
| `seed` (plan) | 0 | Integer in [0, 2³² − 1] (review 1; also each entry of `seeds`) | V7 |

### 3.3 DMD

| Setting | Default (untuned fit) | Search (deliverable) | Allowed | Source |
| --- | --- | --- | --- | --- |
| `rank` (A2, replaces `energy`) | 4 | {2, 3, 4, 5, 6, 8} | Distinct integers, 1 to 12 | D-019. On the real data, energy 0.99 and 0.999 both gave r = 2, and 0.9999 gave r = 4 (review 1), so the energy dimension did not separate the settings. The 99%-energy rank (PRD §6) is reported as `r_99` |
| `basis` (A2) | real | Fixed: real | real, exact | D-019, §2.9a |
| `energy_report` (A2) | 0.99 | Fixed | [0.9, 1.0] | PRD §6 ("99% of the energy"); report only |
| `real_kappa_max` (A2) | 3.0 | Fixed | 0.5 to 10 | Assumption: a neighbour ratio λ_{j+1}/λ_j below e^{−3} = 0.05 decays within one step |
| `real_kappa_sep` (A2) | 0.005 | Fixed | 1e-4 to 0.05 | Assumption: neighbours differ by ≥ 0.5% (about 17% after 37 steps). The fitted window cannot separate closer decay rates |
| `kernel` | matern25 | {matern25} (A2: fixed; matern15 changed J by < 1e-4 in review 1) | ⊆ {matern25, matern15, rbf} | Article 4, p. 1103 (Matérn 5/2) |
| `fluid_weights` | d012 | {d012, equal} | Keys of `fluid_weight_sets` | D-012 |
| `gp_target` | coef | {coef, proj} | ⊆ {coef, proj} | PRD §6 step 4 (coef); D-016 (proj) |
| `unstable` | clip | Fixed (exact basis only) | clip, keep | D-016 |
| `stab_tol` | 1e-9 | Fixed | 0 to 1e-6 | Assumption: ignores round-off on λ = 1 |
| `imag_tol` | 1e-10 | Fixed | 0 to 1e-6 | Assumption |
| `sigma_rel_tol` | 1e-10 | Fixed | 1e-14 to 1e-6 | Assumption |
| `dup_tol` | 1e-8 | Fixed | 0 to 1e-4 | Assumption |
| `r_max` | null | Fixed | null or ≥ 1 | Assumption |
| `gp.alpha` | 1e-10 | Fixed | 1e-12 to 1e-2 | scikit-learn default |
| `gp.normalize_y` | true | Fixed | true, false | Assumption (§2.9 step 9) |
| `gp.n_restarts_optimizer` | 5 | Fixed | 0 to 20 | Assumption |
| `gp.constant_bounds` | (1e-3, 1e3) | Fixed | 0 < lo < hi | Assumption |
| `gp.length_scale_bounds` | (1e-2, 1e2) | Fixed | 0 < lo < hi | Assumption |
| `gp.noise_init`, `gp.noise_bounds` | 1e-4, (1e-10, 1) | Fixed | lo ≤ init ≤ hi | Assumption |
| `gp.std_includes_basis_residual` | true | Fixed | true, false | D-016 |
| `n_trials` | 30 | Grid of 24 | 1 to 30, ≥ grid size | PRD §6; D-015 |
| `seed` | 0 | Any integer | Any integer | V7 |

### 3.4 Benchmarks and common keys

| Setting | Value | Allowed | Source |
| --- | --- | --- | --- |
| B1 `knn` | n_neighbors 3, weights distance, algorithm brute | Fixed: exactly these values | PRD §6 |
| `objective.fluids` | [oil, water] | Non-empty subset of the fluids, no duplicates | PRD §7 |
| `fluid_weight_sets` | d012, equal | 3 values ≥ 0, sum 1 ± 1e-9, oil > 0 and water > 0 | D-012 |
| `data.t_cut` | null | null or an integer ≥ 3 (checked against k0 and T at run time) | §2.13 |
| `report.plateau_frac` | 0.98 | 0.5 to 1 | PRD §7 |
| `report.breakthrough_frac` | 0.01 | 0 to 0.5 | PRD §7 |
| `report.forecast_steps` | 80 | 1 to 400 | PRD §1 |
| `report.coverage_z` | 1.96 | > 0 | Standard |
| `speed.*` | 10,000; 200; 32; 5,000; 600 s | Positive integers. `limit_s` > 0 | PRD §6, §7, §8 |

### 3.5 Packages (V7)

| Package | Version | Group | Status |
| --- | --- | --- | --- |
| torch | 2.14.1+cpu | `models` (pyproject), CPU index in `requirements-lock.txt` | Installed (main session, 13 Oct 2026) |
| scikit-learn | 1.7.2 | Add to `models` | **Not in `.venv`**: `requirements-lock.txt` does not list it. Install it, pin it and regenerate the lock (with joblib and threadpoolctl). If it fails to import with scipy 1.17.1, report it. A different pin needs a researcher decision |
| numpy, scipy, pandas, pyyaml, openpyxl | As pinned | main | Unchanged |

No optuna, pydmd or matplotlib in Task 2.

`pyproject.toml` registers the marker `slow` and sets `addopts = "-m 'not slow'"`, so `pytest` runs the fast tests. `pytest -m slow` runs the slow ones.

## 4. Acceptance tests

`pytest` must pass (V10). The fast Task 2 tests must finish in under 120 s on the laptop; the data scientist reports the time. Tests that train networks use `hidden_size` 16, `max_epochs` ≤ 5, `patience` ≤ 3, `seeds` [0, 1].

### 4.0 Fixtures (`tests/synth_models.py` and `tests/conftest.py`)

**`make_curves(X, T, k0)`** gives Y (n, T, 3) in [0, 1], smooth in x and 0 for k < k0. For τ = k − k0 ≥ 0:

| Fluid | Formula |
| --- | --- |
| Oil | exp(−(0.08 + 0.12 x₁) · max(0, τ − (2 + 3 x₀))). A plateau at 1, then a decline |
| Gas | 0.99 · oil |
| Water | 0.9 (1 − exp(−(0.15 + 0.25 x₃) · max(0, τ − (1 + 4 x₂)))) |

**`make_exponentials(X, T, lam)`** sets k0 = 0. With ρ = 0.95 and θ = 0.4:

y_f(τ) = s_f · [a₀ + a₁ · 0.9^τ + a₂ · 0.7^τ + ρ^τ (a₃ cos θτ + a₄ sin θτ)]

| Term | Formula |
| --- | --- |
| a₀ | 2 + 0.5 x₀ |
| a₁ | 0.3 + 0.1 x₁ |
| a₂ | −0.3 + 0.1 x₂ |
| a₃ | 0.2 + 0.1 x₃ |
| a₄ | 0.1 + 0.1 x₀ |
| s (oil, gas, water) | (1, 0.98, 0.6) |

All values are > 0. The true eigenvalues are {1, 0.9, 0.7, 0.95 e^{±0.4i}}.

**`synthetic_task1_db(tmp_path)`** is a database with the Task 1 schema (`init_schema`) and plain INSERTs.

- **Cases:** n = 30. Train: 1 to 18. Validation: 19 to 24. Test: 25 to 30.
- **Dates:** T = 13 quarterly dates from 2000-01-01.
- **X:** `default_rng(7).random((30, 4))`, rescaled so that each column has minimum 0 and maximum 1. Physical values between (0.05, 0.8, 0.5, 50) and (0.15, 1.5, 10, 200).
- **Y:** Y_phys = `make_curves(X, 13, k0=2)` × (15000, 5450, 9000).
  - Cumulatives = Σ rate · Δt.
  - `history` copies case 1.
  - `scalers` rows: y factors = train maxima, x bounds = the column minimum and maximum.
- **`runs`:** one prepare row, status `ok`. `outputs` holds `rate_definition` R1 and `arrays_sha256` computed from `read_task1_arrays(con)`.

**`tiny_cfg(tmp_path, model)`** is a valid model config for the fixture.

- All paths are in `tmp_path`.
- Neural: `n_trials` 3 (1 for each option), `hidden_size` [16], `max_epochs` 5, `patience` 3, `seeds` [0, 1], `tune_seeds` 1.
- DMD: a grid of 2 (`gp_target` coef, proj), `n_restarts_optimizer` 0. A2: `basis` real, `rank` [3], `kernel` [matern25].

### 4.1 Unit and contract tests

| ID | File | Test | Pass criterion |
| --- | --- | --- | --- |
| M01 | `test_registry.py` | Registry | `PROXIES` = {rnn, lstm, dmd} and `BENCHMARKS` = {b0, b1}. Each class has the attributes and methods of `Proxy` (`isinstance` on an instance). Proxies have `extrapolates_time` True, and benchmarks False. `load_model(..., proxies_only=True)` on a saved b0 raises `ContractError`. An unknown name raises `ConfigError` |
| M02 | `test_contract.py` | Contract (PRD §6), for each of the 5 models, on `make_curves` (n_fit 12, n_val 4, T 13, k0 2) | `predict` gives (4, 13, 3) float64, finite and ≥ 0. `predict_std` gives (4, 13, 3), finite and ≥ 0 (proxies), or None (benchmarks). Two fits from scratch with the same seeds give `np.array_equal` predictions |
| M03 | `test_contract.py` | Seeds matter (rnn, lstm) | Seeds [0, 1] and [2, 3] give different predictions |
| M04 | `test_contract.py` | Continuation (PRD §6), for each proxy | `forecast(X, 80)` gives (n, 80, 3), finite and ≥ 0. `rollout(X, T + 80)[:, :T]` equals `predict(X)` (DMD atol 1e-12; neural atol 1e-6, scaled units). `forecast(X, 80)` == `rollout(X, T + 80)[:, T:]` exactly. `forecast(X, 10)` equals `forecast(X, 80)[:, :10]` with the same tolerances (causality) |
| M05 | `test_contract.py` | Benchmarks do not continue | `forecast(X, 1)` and `rollout(X, T + 1)` raise `NotImplementedError`. `rollout(X, T)` == `predict(X)` |
| M06 | `test_base.py` | Onset | k0 = 2 is found. Every model's `predict`, `predict_std` and `rollout` are exactly 0 for k < 2. `make_curves` with k0 = 0 works. All-zero Y raises `ContractError`, and so does T − k0 = 2 |
| M07 | `test_base.py` | V9 inputs | X with NaN, X = 1 + 1e-9, X = −1e-9, wrong p, Y < 0, Y with NaN, and a `val` with a different T each raise `ContractError`. A neural `fit` without `val` raises `ContractError` |
| M08 | `test_contract.py` | Save and load, for each model | After `save` and `load`, `predict`, `predict_std` and `rollout(X, T + 80)` are `np.array_equal` to the values before. `meta.json` has every key of §1.5. One changed byte in an artefact makes `load` raise `ContractError`. `load_model(..., con=con)` with a different `y_factor` in the DB raises `ContractError` |
| M09 | `test_metrics.py` | J by hand: 2 cases, 2 dates. Oil true [[0, 10], [0, 20]], predicted [[0, 12], [0, 20]]. Gas = oil (true and predicted). Water true [[0, 4], [0, 8]], predicted [[0, 4], [0, 6]] | RMSE_oil = 1 and NRMSE_oil = 0.05. RMSE_water = 1 and NRMSE_water = 0.125. J = 0.0875 (1e-12). The same arrays ÷ (s_oil, s_water) give the same NRMSE (1e-12). A constant true series raises `ContractError` |
| M10 | `test_metrics.py` | Weighted loss | It agrees with a hand calculation for w = (0.25, 0.25, 0.5) on a 2-case, 2-step tensor (1e-6, float32). `equal` gives the plain MSE over the three fluids. A weight set that sums to 0.9 raises `ConfigError` in `load_model_config` |
| M11 | `test_tuning.py` | LHS plan | `n_trials` 30 with strata [A, B, C] gives 30 trials, 10 in each stratum, in stratum order. The same seed gives an identical plan, and another seed a different one. In each stratum, each `loguniform` dimension has exactly one trial in each of the 10 equal bins of ln(value). Every value is within its range. Every `choice` value appears at least once in each stratum (10 trials, at most 4 values). `n_trials` 31 or 29 raises `ConfigError` |
| M12 | `test_tuning.py` | DMD grid (A2) | The deliverable space gives 24 distinct points (rank 6 × kernel 1 × weights 2 × target 2), last dimension fastest, and each resolved fit cfg has `rank` and no `energy`. `n_trials` 23 raises `ConfigError` |
| M13 | `test_seq.py` | Input options | The input sizes are A: p + 1, B: 1, C: p. For LSTM option B, h₀ and c₀ have shape (num_layers, n, H). For every option, a change in X changes the output. Under option C, the input tensor is the same at every step |
| M14 | `test_seq.py` | Time scaling | The t values passed to the network are k ÷ (T_fit − 1) for k = k0..: k ÷ 12 when T = 13, and k ÷ 8 when Y is cut to 9 dates. `rollout` continues with k ≥ T_fit (t > 1) |
| M15 | `test_seq.py` | Early stopping | `best_epoch` = argmin of the J history. The member's recomputed J after the restore equals the minimum (1e-12). n_epochs ≤ best_epoch + patience |
| M16 | `test_seq.py` | Ensemble | `predict` = max(mean of the member raw outputs, 0). `predict_std` = std with ddof = 1 (1e-12). `fit(seeds [0, 1])` is `np.array_equal` to `from_members` of two `train_member` calls |
| M17 | `test_train.py` | Parallel determinism | `train_model` with `n_jobs` 1 and 2 gives equal j_val for every trial, and `np.array_equal` final predictions |
| M18 | `test_dmd.py` | PRD §6: sum of exponentials. `make_exponentials`, n_fit 30, n_val 8, T 25, `basis` exact, rank 5 (A2; was energy 1.0), `unstable` clip | q = r = 5. The recovered λ, sorted, equal the true values within 1e-8, and n_stabilised = 0. Training reconstruction B a, max abs error < 1e-9. `rollout_from_coef(coef_train, T + 80)` equals the analytic curves within relative 1e-8, inside and after the training dates. End to end (GP, `coef`, matern25): validation NRMSE < 0.02 for each fluid over the T dates, and < 0.05 over the 80 continuation steps against the analytic curves |
| M19 | `test_dmd.py` | `energy_rank` | σ = [3, 2, 1, 0.1]: energy 0.99 gives r = 3, 0.9995 gives r = 4, and 1.0 gives r = q = 4. σ = [1, 1e-12] gives q = 1 |
| M20 | `test_dmd.py` | Real-basis equivalence | For random conjugate-closed λ (2 real, 2 pairs) and conjugate-symmetric c: Re Σ c λ^τ = b(τ)ᵀ a with a = (c_real, 2 Re c, −2 Im c), within 1e-12, for τ = 0..50 |
| M21 | `test_dmd.py` | Eigenvalue handling (`basis` exact). Data y_f = s_f [(1 + 0.5 x₀) + (0.2 + 0.1 x₁) · 1.03^τ + (0.3 + 0.1 x₂) · 0.8^τ], s = (1, 0.98, 0.6), k0 = 0, T_p = 15, rank 3 (A2; was energy 1.0) | `clip`: q = 3, n_stabilised = 1, every \|λ_used\| ≤ 1 + 1e-12, n_merged = 1 (1.03 → 1 merges with 1), r_eff = 2, and the rollout to T + 80 is finite. `keep`: `rollout_from_coef(coef_train, T + 80)` equals the analytic curves within relative 1e-6, and its last step ÷ the last fitted step is > 2 for every case. Every basis column is real, and r_eff = n_real + 2 n_pair |
| M22 | `test_dmd.py` | `proj` target | QR coefficients R⁻¹ Qᵀ y equal the `lstsq` coefficients (relative 1e-10). A basis with min \|R_jj\| ÷ max \|R_jj\| < 1e-12 raises `TrialFailed` |
| M23 | `test_dmd.py` | DMD `predict_std` | For one case, it equals the hand formula of §2.9 step 11 from the GP stds (1e-12). With `std_includes_basis_residual` false, the ē² term is absent. It is 0 for k < k0 |
| M24 | `test_train.py` | Plateau report | On `make_curves`, k_end,i equals the §2.10 definition computed by hand. The diagnostics have RMSE for the plateau, post-plateau, pre-breakthrough and post-breakthrough periods for each fluid, and DMD adds the LS reconstruction set |
| M25 | `test_baselines.py` | B0 | `predict` equals the mean training curve exactly, for any X |
| M26 | `test_baselines.py` | B1 | On 4 training points with hand-set X and distinct curves, the query equals Σ (1/d_j) y_j ÷ Σ 1/d_j over the 3 nearest (1e-12). A query equal to a training X returns that curve exactly |
| M27 | `test_base.py` | Clipping | A model with negative raw outputs (monkeypatched raw rollout) gives `predict` ≥ 0, and `last_n_clipped` = the number of negatives. For random y and factors s > 0, `YScaler.inverse(clip(y))` is bit-equal to `clip(YScaler.inverse(y))` |
| M42 | `test_dmd.py` | A2: real exponentials. `make_real_exponentials` (new fixture): `make_exponentials` without the complex pair, y_f = s_f [a₀ + a₁ 0.9^τ + a₂ 0.7^τ]. k0 = 0, n_fit 30, T 25, `basis` real, rank 3 | The λ used, in descending order, equal (1, 0.9, 0.7) within 1e-6. The kinds are all 0, and n_merged = n_stabilised = 0. The training reconstruction B a has max abs error < 1e-6. `rollout_from_coef(coef_train, T + 80)` equals the analytic curves within relative 1e-5, inside and after the training dates. `varpro.start_costs` has 4 entries (the final cost of each start), and `varpro.cost` equals their minimum |
| M43 | `test_dmd.py` | A2: constraints. `basis` real, `gp_target` proj, on (a) `make_exponentials` (it has a complex pair), rank 5, and (b) `make_curves` (n 12, T 13, k0 2), rank 3 | Every λ is real, and 1 ≥ λ₁ > … > λ_m > 0. Each ratio λ_{j+1}/λ_j lies in [e^{−κ_max}, e^{−δ}] (1e-12). r_eff = m. `lam_exact` (S0) is recorded. Two fits give `np.array_equal` λ, `predict` and `forecast(X, 80)` |
| M44 | `test_dmd.py` | A2: oscillation removed. Data: the review-1 slow pair, y_f = s_f ρ^τ [a(x) cos θτ + b(x) sin θτ], with ρ = 0.9936, θ = 0.0157, a = 1 + 0.1 x₀ and b = −(1 + 0.1 x₁). The snapshot matrix has rank 2, and every curve is > 0 for τ ≤ 36 and crosses 0 at τ ≈ 47 to 53. k0 = 0, T 37, n_fit 20, s = (1, 0.98, 0.6) | `basis` exact, rank 2: the λ are ρ e^{±iθ} within 1e-8, and `rollout_from_coef(coef_train, T + 80)` has a negative value for every training case (it reproduces the review-1 failure). `basis` real, rank 2: over τ = 0..116, every raw curve of `rollout_from_coef` has at most 1 sign change and at most 1 local extremum (the guarantee of §2.9a). The test prints the number of curves that are negative at τ = 116 |
| M45 | `test_train.py` | A2: DMD plausibility gate (§2.14). Monkeypatch `DMDProxy` fits on the tiny grid. Grid point 0 has the lowest J but clips in the validation continuation; grid point 1 does not | The run selects trial 1. `trials.csv` has `basis`, `n_clip_cont` and `plausible`. If every grid point clips, the run selects the lowest J, `runs.outputs.variants.main.dmd.plausible_selected` is false, report §11 says so, and the run status is `ok` |

### 4.2 Pipeline tests (`synthetic_task1_db`)

| ID | File | Test | Pass criterion |
| --- | --- | --- | --- |
| M28 | `test_dataset.py` | `load_dataset` | The fit arrays hold train cases 1 to 18 and the validation arrays cases 19 to 24. No test case or test array is in the `Dataset`. The shapes match §1.3. `arrays_sha256` equals the `runs` value. One altered `production_grid` rate raises `ContractError` (hash). A prepare run with status `failed` raises `ContractError` (F2) |
| M29 | `test_dataset.py` | `t_cut` | `t_cut` 7 gives `Y_fit` (18, 7, 3), `Y_val` (6, 7, 3) and `dates[:7]`. `t_cut` 4 (< k0 + 3) and 14 (> T) raise `ContractError` |
| M30 | `test_train.py` | End to end, for each of the 5 models, through `cli.main(['train', ...])` with `tiny_cfg` | Exit 0. One `runs` row with `task` 'task2', `command` 'train', the model, `status` 'ok', `duration_s` > 0, `packages` with torch and scikit-learn versions, and every key of §1.6 in `outputs`. The files of §1.5 exist. `predictions` rows = (18 + 6) × 13 × the number of variants, with `split` in {train, val}, physical values ≥ 0, std NULL only for b0 and b1. `trials.csv` has 3 rows (neural), 2 (DMD) or 0 (b0, b1). `report.md` has the sections of §1.7 |
| M31 | `test_train.py` | V3 spy | Wrappers on `fit`, `predict`, `predict_std`, `rollout` and `forecast` record every X during an `lstm` and a `dmd` run. No recorded row equals the X of a test case. `predictions` has no case_id from 25 to 30 |
| M32 | `test_train.py` | Run determinism | Two runs with the same config give equal `predictions` values (the run IDs differ), and equal `trials.csv` except for `fit_s` |
| M33 | `test_train.py` | `--settings-from` and `--t-cut` | Run 2 is `--t-cut 9 --settings-from <run 1>`. It has T_fit = 9, the tuned settings of each variant equal to run 1's, one `trials.csv` row for each variant, `predictions` with date_idx < 9, and `meta.json` with `t_cut` 9 and `settings_from` = run 1. `--settings-from` with a run of another model raises `ConfigError` |
| M34 | `test_train.py` | Failure path | With the hash mismatch from M28, `train` exits 1, `runs.status` is 'failed', and `predictions` has no row for that run |
| M35 | `test_config.py` | Model config validation | Each raises `ConfigError`: `--model` ≠ `model`, `n_trials` 31, `hidden_size` 128, `num_layers` 3, a weight set summing to 0.9, `cell` in `lstm.yaml`, `knn.n_neighbors` 5, and (A2) `rank` 0, `rank` 2.5, duplicate `rank` values, an `energy` dimension in the DMD space, `basis` "hankel", and `real_kappa_sep` 0.1. The five deliverable configs load without error |
| M36 | `test_train.py` | Trial failures | A trial forced to fail (monkeypatched `train_member` that raises `TrialFailed` for one trial) is recorded with status `failed` and j_val = inf, and the run completes. If every trial of one variant fails, the run ends `failed` |
| M37 | `test_db.py` | Schema | `init_schema` creates `predictions` with the §1.6 columns, and is idempotent. `write_predictions` with a test row raises `ContractError`. After a `train` run, `run_prepare` on the same database succeeds, because `predictions` has no foreign key to `parameters` |

### 4.3 Speed tests

| ID | Marker | Test | Pass criterion |
| --- | --- | --- | --- |
| M38 | none (fast) | For each proxy at the largest settings: `lstm` and `rnn` (gru) with 2 layers, 64 units, 5 members (untrained: `max_epochs` 1); `dmd` fitted with `basis` real and rank 8 (A2: the largest in the search) and `n_restarts_optimizer` 0, on 70 cases × 41 dates of `make_curves` with k0 = 4. Then 50 calls of `predict` on 32 rows | Mean time per call < `limit_s` ÷ `mcmc_calls` = 0.12 s. The test prints r_eff for DMD |
| M39 | `slow` | The same models: 5,000 calls × 32 rows, and one call with 160,000 rows | Each < 600 s. The test prints the times |

### 4.4 Raw-data tests (marker `rawdata`, fixture `real_prepared`)

| ID | Test | Pass criterion |
| --- | --- | --- |
| M40 | `load_dataset` on the real database | `X_fit` (70, 4), `Y_fit` (70, 41, 3), `X_val` (15, 4), `Y_val` (15, 41, 3). k0 = 4. No test case. `arrays_sha256` = the prepare run value |
| M41 | DMD (A2: `basis` real, rank 4, matern25, d012, proj) and B0 and B1, fitted on the real train cases with no tuning | Finite predictions ≥ 0. Every λ_used is real and in (0, 1]. r_eff = 4. The diagnostics have the plateau errors and `r_99`. Validation J is finite for all three. `forecast(X_val, 80)` is finite, and the test prints its clip count |

### 4.5 Review checklist (researcher, before the merge)

- **V1:** fits use `X_fit` and `Y_fit` only. k0, the DMD basis and the GPs come from train cases. Selection and early stopping use validation cases only. The Y scaler is the Task 1 scaler (no refit).
- **V2, V3:** splits come from the stored `split` column. No test row is loaded (M28, M31) or written (M37).
- **V5:** a grep of `src/dwc26/models` and `src/dwc26/evaluation` finds no literal for T, k0, n, the forecast length, a split range or a date.
- **V7:**
  - Pins: torch 2.14.1+cpu and scikit-learn 1.7.2 are in pyproject and the lock.
  - Seeds are in the YAML.
  - M02, M17, M32.
- **V8:** the `runs` row and the log. `runs.packages` has torch and scikit-learn.
- **V9:** §1.8.
- **V10:** one test file for each new module.
- **V13:** proxies continue in time (M04). Benchmarks cannot (M05), and `load_model` keeps them out of Tasks 4 and 5.
- **Plausibility of the real runs** (the data scientist reports each):
  - k0 = 4 and T_fit = 41.
  - For each variant: J_val and the validation RMSE of oil and water, against B0 and B1.
  - DMD: r at 99%, r_eff, n_stabilised, cond(B), and the plateau error.
  - The continuation check: rising oil, and the clip counts.
  - The coverage preview.
  - The speed extrapolations.
  - Training times.
- **One command from a clean clone:** `prepare`, then `train` for each of the 5 models. The README lists the commands and the run times.
- **A2 (D-019, D-020), on the DMD rerun:**
  - Every λ is real and in (0, 1].
  - r_eff takes 6 distinct values over the grid.
  - `lam_exact` and `r_99` are reported.
  - The selected trial is plausible: no clipped value in the validation continuation.
  - The report gives the rising-oil count and the maximum oil ratio (P2 information), the plateau errors, and the validation RMSE of oil and water against B0 and B1.

## 5. Consequences for later tasks

- **Task 3:**
  - **Candidates:** the proxy IDs `rnn/A..C`, `lstm/A..C` and `dmd/main`. Benchmarks: `b0/main` and `b1/main`.
  - **Requirements:**
    - All compared runs must share `arrays_sha256` and `t_cut` = null.
    - Use `evaluation/metrics.py`, so that the ranking metric equals the tuning objective J.
  - **Biases and checks:**
    - The validation J of each final model is optimistically biased (§2.5).
    - Test predictions are made only after the freeze, by `load_model`, with `write_predictions(..., allow_test=True)`.
    - Add the GOR check (D-010), with a relative tolerance `gor_rel_tol` = 0.005 around the training range (assumption: the oil and gas validation NRMSE are about 0.2% to 0.5%). It is report only. Review 1: rnn/A 0.3661 and lstm/C 0.3655 against a training maximum of 0.3654, which is within the tolerance.
  - **Plausibility rules (D-020), added after PRD §7 selection step 2.** Each uses the 80-step continuation of the validation cases (`forecast(X_val, 80)`).
    - **P1.** Any clipped value in the continuation rejects the proxy as primary and as secondary.
    - **P2.** The proxy fails if its oil exceeds 1.05 × its own predicted rate at T_fit − 1, at any step, in more than half of the validation cases.
      - While D-016's assumption of no new wells after 2008 stands, a proxy that fails P2 is not eligible as primary or secondary.
      - If the user confirms new wells or development after 2008, P2 becomes report only.
      - Review 1: lstm/C fails P2 (15 of 15 cases, ratio up to 1.50). All other neural variants pass (0 cases).
    - **P3 (decline continuity; report only, D-022).**
      - For each validation case, compute D_hist = the mean relative oil decline per quarter over the last 8 history quarters, and D_fc = the same over forecast quarters 9 to 16 (2010 to 2012).
      - Report R = median over the cases of D_fc ÷ D_hist, and flag R < 0.5.
      - Basis: under Arps decline with b ≤ 1, D(t) = D_i ÷ (1 + b D_i t) (Arps 1945). At the observed D_i ≈ 0.02 to 0.035 per quarter, D falls by no more than about 1 ÷ (1 + 0.035 · 16) ≈ 0.64 over 16 quarters, so R < 0.5 means that the decline stalls.
      - Review 2: the A2 DMD (rank 8) has one mode at λ ≈ 1 and all other modes with half-lives ≤ 2.7 quarters, so its oil changes by ≤ 9.8% from 2010 to 2028. The backtest (12 quarters from 2005) cannot see this. Task 5 must state it wherever the DMD forecast measures method uncertainty.
    - The backtest (B2 at horizons 9 to 12) still applies to every candidate.
  - **Fallback secondary (D-020).** If no S4 proxy passes steps 1 and 2 and P1, the secondary is the best passing proxy of the other S3 family (RNN if the primary is LSTM, and the reverse). This departs from PRD §7 step 5. The report then states that the method uncertainty covers S3 only.
  - **Coverage (D-021).**
    - Task 3 reports the 95% coverage of each proxy, flags values below 0.90, and does not recalibrate.
    - Reason: Task 4's σ_proxy comes from the validation RMSE (PRD §8). Its std filter compares each sample with the 95th percentile of the same std over cases 71 to 85, so a global variance scale cancels.
    - Review 1 found 0.65 to 0.92 for the seed ensembles. Five seeds capture only the optimisation spread, not the model error (Lakshminarayanan et al. 2017).
- **Task 4:**
  - Load the primary proxy with `load_model(proxy_id, models_dir, con=con)`, so that the scalers are checked (V4).
  - `predict` refuses X outside [0, 1]: evaluate the prior first and call the proxy only for rows inside the box.
  - The `predict_std` for the sample filter is the seed spread (RNN, LSTM) or the GP plus basis residual (DMD).
  - σ_proxy comes from the Task 3 validation RMSE (PRD §8).
  - Measure the MCMC time against the §2.15 extrapolation.
- **Task 5:**
  - F1 and F2 are `forecast(X, 80)`.
  - Backtest: `python -m dwc26 train --model M --config ... --t-cut <n> --settings-from <run_id>` for each proxy. Each RNN and LSTM option is a variant of that run.
  - The backtest settings saw the full-window validation data (§2.13).
  - B2 (persistence) is defined in the Task 5 spec.
