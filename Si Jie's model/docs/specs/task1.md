# Task 1: Data Preparation

Status: approved 13 Oct 2026 (review of branch `task1` at edd39c4, D-011). Amendment A1 (C10 tolerance, D-009) and follow-up F2 (`read_task1_arrays` run status, §1.3) are implemented on branch `task2` (commits 7973b52 and 2cba183, 13 Oct 2026; 128 tests pass). The researcher reviews them with the Task 2 merge. Base: PRD §5. Departures from the PRD: D-003 to D-007 (§0.2, logged in `docs/decisions.md`). Clarifications from the review are marked "(review)".

Label key. "source: PRD §5" means the PRD states it. "source: profile" means the read-only profile of files 01 and 02 made by the main session on 13 Oct 2026 (see `docs/references.md`). "assumption" means that this spec decides it and gives a reason.

## 0. Summary

### 0.1 What Task 1 does

`python -m dwc26 prepare --config configs/data.yaml` does these steps:

1. Reads files 01 and 02 from `data/raw/`.
2. Runs checks C1 to C10.
3. Aligns every case and `RAMP_History` to the T history dates with one rate definition.
4. Scales X and Y.
5. Writes `data/processed/dwc26.sqlite`, `data/processed/arrays.npz`, `outputs/reports/data_checks.md` and `outputs/logs/<run_id>.log`, plus one row in `runs`.

On the real files T = 41 and n = 100. The code reads both values from the files (V5).

### 0.2 Departures from the PRD

| ID | PRD §5 | This spec | Reason |
| --- | --- | --- | --- |
| D-003 | X min-max on "bounds of file 01" | Bounds = per-parameter min and max of the n design points in file 01, read at run time. An optional config override must carry a citation and must contain every design point | File 01 has no prior bounds (source: profile) |
| D-004 | C6 compares the history rate with the previous-quarter mean. Pass → R1, fail → R2 | C6 compares the reported history rate with the previous-quarter mean and with the next-quarter mean. Pass (either < 1%) → R1, computed from cumulatives for cases and history alike. Fail → stop (under `rates.definition: auto`) | The history rate is the next-quarter mean, so the PRD's C6 fails (max difference 1.0). R2 would then give the cases the simulator's last-timestep rate and the history a quarter mean, which breaks V4 |
| D-005 | R1 at t_0: "use the reported rate" | R1 at t_0 = 0, guarded by the new stop check C8 (cumulative = 0 at t_0) | The reported case rate is the simulator's last-timestep rate. Using it at t_0 mixes definitions (V4) |
| D-006 | Checks C1 to C7 | C1 extended (file 01, units, Date column, history dates). New C8 (stop), C9 and C10 (report only). C7 trend defined | Precision. Tasks 2 to 4 need C9 and C10 |
| D-007 | Table columns in PRD §5 | Extra columns and exact DDL (§1.6). `case_id = 0` is `RAMP_History` in `production_native`. `scalers` uses `lower`/`upper` instead of `bounds`. Task 1 tables are replaced on each successful run | Traceability and an unambiguous schema |

## 1. Inputs and outputs

### 1.1 Input files

| Config key | Default path | Sheet | Use |
| --- | --- | --- | --- |
| `paths.file01` | `data/raw/01 Introduction.xlsx` | `Uncertainty Values` | X for each case |
| `paths.file02` | `data/raw/02 Field Production.xlsx` | 6 data sheets (§1.2 `file02.sheets`) | Native rates and cumulatives of each case and of `RAMP_History` |
| `paths.file02` | same | `Field Production` | Reported history rates for C4, and the water injection rate |

The sheets `Plots` and `Nomenclature` are ignored.

Layout (source: profile). The code must verify this layout through C1 and must not assume it.

- **File 01:** row 1 is the header `Item | Fault Transmissibility | Porosity Multiplier | Permeability Multiplier | Aquifer Pore Volume`. Rows `Case 1` to `Case 100` follow (a space between "Case" and the number).
- **File 02 data sheets:**
  - Excel row 1 holds labels (`X, Y1 ...`), which are ignored.
  - Row 2 holds the names `Date, Case_1 ... Case_100, RAMP_History`.
  - Row 3 holds the units, for example `Oil production rate [STB/d]`.
  - Data start at row 4. Some sheets end with one all-empty row.
  - All six sheets share one Date column: the union of all report dates, 11,096 rows. Some dates carry a time of day.
  - An empty cell means "no report on this date". It is not a missing value and is never imputed (source: PRD §5).
  - (review) Only a cell that openpyxl reads as `None` or `''` is empty. Any other content, including whitespace-only text, is a value. If it does not parse to a finite float, it is a bad cell and C3 stops. Assumption: a whitespace cell is suspicious, so the run stops rather than drop it silently.
- **`Field Production`:** the header is `Identifier, Date, Oil production rate (STB/d), Gas production rate (MSCF/d), Water production rate (STB/d), Water injection rate (STB/d)`, followed by T rows.

Units of the parameters: `fault_trans`, `poro_mult` and `perm_mult` are dimensionless multipliers. File 01 does not state the unit of `aquifer_pv`, so it is stored as given.

### 1.2 Config `configs/data.yaml`

All keys below are required. Relative paths resolve against the current working directory, and every command runs from the repository root.

```yaml
seed: 0                         # logged (V7, V8); Task 1 uses no randomness
paths:
  file01: data/raw/01 Introduction.xlsx
  file02: data/raw/02 Field Production.xlsx
  db: data/processed/dwc26.sqlite
  arrays: data/processed/arrays.npz
  report: outputs/reports/data_checks.md
  logs: outputs/logs
file01:
  sheet: Uncertainty Values
  label_column: Item
  label_pattern: '^Case\s+(\d+)$'
  columns:                      # file header -> internal name; this order is the column order of X
    Fault Transmissibility: fault_trans
    Porosity Multiplier: poro_mult
    Permeability Multiplier: perm_mult
    Aquifer Pore Volume: aquifer_pv
file02:
  names_row: 2                  # 1-based Excel rows
  units_row: 3
  first_data_row: 4
  date_column: Date
  case_pattern: '^Case_(\d+)$'
  history_column: RAMP_History
  sheets:                       # quantity key -> sheet name and expected unit (text inside [...])
    oil_rate:   {name: Oil production rate,         unit: STB/d}
    oil_cum:    {name: Oil production cumulative,   unit: STB}
    gas_rate:   {name: Gas production rate,         unit: MSCF/d}
    gas_cum:    {name: Gas production cumulative,   unit: MSCF}
    water_rate: {name: Water production rate,       unit: STB/d}
    water_cum:  {name: Water production cumulative, unit: STB}
  field_sheet:
    name: Field Production
    columns:                    # file header -> internal name
      Date: date
      Oil production rate (STB/d): oil_rate
      Gas production rate (MSCF/d): gas_rate
      Water production rate (STB/d): water_rate
      Water injection rate (STB/d): water_inj_rate
  date_round: ms                # round parsed timestamps to the millisecond
split:                          # inclusive case-number ranges (PRD §1)
  train: [1, 70]
  val: [71, 85]
  test: [86, 100]
x_bounds:
  source: design                # design | override
  override: null                # {fault_trans: [lo, hi], poro_mult: [...], perm_mult: [...], aquifer_pv: [...]}
  citation: null                # required, non-empty, when source = override
rates:
  definition: auto              # auto | R1 | R2
checks:
  c4_rel_tol: 1.0e-6
  c6_rel_tol: 0.01
  c6_floor_frac: 1.0e-3
  c8_abs_tol: 0.0
  c10_rel_tol: 1.0e-5             # amendment A1 (D-009)
```

Fluid order everywhere: `[oil, gas, water]`. Units: oil and water rates in STB/d, gas rate in MSCF/d. Oil and water cumulatives in STB, gas cumulative in MSCF.

### 1.3 Modules and boundary signatures

Layout from PRD §3. All arrays are `np.float64` unless stated otherwise. V9 assertions raise `ContractError` and never use `assert`, because `python -O` strips `assert` statements.

| Module | Content |
| --- | --- |
| `src/dwc26/__main__.py` | Calls `dwc26.cli.main()` |
| `src/dwc26/cli.py` | `argparse` with subcommand `prepare --config PATH`. Subcommands not yet built exit with code 2 and the message "not implemented". Exit code 1 on `DataCheckError` |
| `src/dwc26/config.py` | `load_config(path: Path) -> dict` (`yaml.safe_load`). `ConfigError(ValueError)` for missing or invalid keys |
| `src/dwc26/data/load.py` | Excel reading |
| `src/dwc26/data/check.py` | C1 to C10, `CheckResult`, `DataCheckError`, `ContractError`, `assert_array`, the report writer |
| `src/dwc26/data/align.py` | Days, grid, interpolation, R1, R2, `align_all` |
| `src/dwc26/data/scale.py` | `XScaler`, `YScaler` |
| `src/dwc26/data/split.py` | `assign_split` |
| `src/dwc26/data/db.py` | Schema, writes, run log, `read_task1_arrays` |
| `src/dwc26/data/prepare.py` | `run_prepare(config_path: Path) -> str` (returns `run_id`). Runs the steps in §1.4 |

```python
# load.py
HISTORY_ID = 0          # case_id given to RAMP_History in long tables
QUANTITIES = ("oil_rate", "oil_cum", "gas_rate", "gas_cum", "water_rate", "water_cum")

def load_parameters(path: Path, cfg01: dict) -> tuple[pd.DataFrame, list[str]]:
    """Columns: case_id int64, then the 4 internal names in config order (float64).
    Rows are sorted by case_id, and all-empty rows are dropped.
    The list holds structural problems (missing sheet or column, bad label,
    duplicate id, non-finite value). C1 reports them. Raises only if the file
    cannot be opened."""

@dataclass
class RawProduction:
    sheets_found: list[str]
    names: dict[str, list[str]]         # quantity -> row-2 names after the Date column
    units: dict[str, list[str]]         # quantity -> unit text inside [...] for each data column; (review) the
                                        # stripped full text if the cell has no [...], so C1(c) fails on it; '' if empty
    date_col: dict[str, np.ndarray]     # quantity -> Date column, datetime64[<date_round>] (ms by default), all-empty rows dropped
    native: pd.DataFrame                # long format: case_id int64 (0 = RAMP_History), quantity str,
                                        # date datetime64[<date_round>], value float64. Only non-empty cells.
                                        # Sorted by (case_id, quantity, date)
    bad_cells: pd.DataFrame             # non-empty cells that do not parse to a finite float: case_id, quantity, date, raw
    field: pd.DataFrame | None          # date datetime64[ms], oil_rate, gas_rate, water_rate, water_inj_rate

def load_production(path: Path, cfg02: dict) -> RawProduction:
    """Never raises on a structural deviation (missing sheet or column, extra
    column, wrong unit); C1 detects those. Columns that match neither
    case_pattern nor history_column appear in `names` but not in `native`."""

# align.py
def to_days(ts: np.ndarray, t0: np.datetime64) -> np.ndarray: ...            # (m,) datetime64[ms] -> (m,) float64
def history_grid(raw: RawProduction) -> np.ndarray: ...                       # (T,) datetime64[ms]; call only after C1 passes
def interp_cum(t_native: np.ndarray, cum_native: np.ndarray, t_grid: np.ndarray) -> np.ndarray: ...  # (m,), (m,), (T,) -> (T,)
def rates_r1(t_grid: np.ndarray, cum: np.ndarray, c8_abs_tol: float = 0.0) -> np.ndarray: ...  # (T,), (..., T, F) -> (..., T, F)
def rates_r2(t_native: np.ndarray, rate_native: np.ndarray, t_grid: np.ndarray) -> np.ndarray: ...  # (m,), (m,), (T,) -> (T,)

@dataclass
class Aligned:
    dates: np.ndarray               # (T,) datetime64[D]
    t_days: np.ndarray              # (T,) float64, days from dates[0]
    case_ids: np.ndarray            # (n,) int64, ascending
    cum: np.ndarray                 # (n, T, 3) STB, MSCF, STB
    rate: np.ndarray                # (n, T, 3) STB/d, MSCF/d, STB/d, by the selected definition
    hist_cum: np.ndarray            # (T, 3)
    hist_rate: np.ndarray           # (T, 3), same definition and same code path as `rate` (V4)
    hist_rate_reported: np.ndarray  # (T, 3), RAMP_History rate sheets as reported
    definition: str                 # 'R1' | 'R2'
    n_interpolated: int             # (case, cumulative sheet, grid date) cells not on a native date (0 when C5 passes).
                                    # (review) Only cumulatives are interpolated, so rate sheets are not counted

def align_all(raw: RawProduction, grid: np.ndarray, case_ids: np.ndarray,
              definition: str, cfg: dict) -> Aligned: ...

# scale.py
@dataclass(frozen=True)
class XScaler:
    names: tuple[str, ...]          # config order
    lower: np.ndarray               # (4,) physical units
    upper: np.ndarray               # (4,)
    source: str                     # "design: min/max of file 01, cases <ids>" or the override citation
    def transform(self, X_phys: np.ndarray) -> np.ndarray: ...   # (n, 4) -> (n, 4)
    def inverse(self, U: np.ndarray) -> np.ndarray: ...          # (n, 4) -> (n, 4)

def fit_x_scaler(params: pd.DataFrame, cfg: dict) -> XScaler: ...

@dataclass(frozen=True)
class YScaler:
    fluids: tuple[str, ...]         # ('oil', 'gas', 'water')
    factor: np.ndarray              # (3,) STB/d, MSCF/d, STB/d
    fitted_on: tuple[int, ...]      # case_ids used: train only
    def transform(self, Y_phys: np.ndarray) -> np.ndarray: ...   # (..., 3) -> (..., 3)
    def inverse(self, Y: np.ndarray) -> np.ndarray: ...          # (..., 3) -> (..., 3); no clipping

def fit_y_scaler(rate: np.ndarray, case_ids: np.ndarray, split: np.ndarray) -> YScaler: ...
def load_scalers(con: sqlite3.Connection) -> tuple[XScaler, YScaler]: ...   # used by Tasks 2 to 5

# split.py
def assign_split(case_ids: np.ndarray, ranges: dict[str, list[int]]) -> np.ndarray: ...  # (n,) '<U5'

# check.py
@dataclass(frozen=True)
class CheckResult:
    check_id: str        # 'C1' .. 'C10'
    status: str          # 'pass' | 'fail' | 'report'
    action: str          # e.g. 'stop', 'continue', 'select R1', 'interpolate'
    detail: dict         # JSON-serialisable

class ContractError(ValueError): ...
class DataCheckError(RuntimeError):
    results: list[CheckResult]

def assert_array(name: str, a: np.ndarray, shape: tuple, *, dtype=np.float64,
                 finite: bool = True, nonneg: bool = False) -> None: ...
def check_c1(params: pd.DataFrame, problems01: list[str], raw: RawProduction, cfg: dict) -> CheckResult: ...
def check_c2(raw, grid, case_ids) -> CheckResult: ...
def check_c3(raw) -> CheckResult: ...
def check_c4(raw, grid, cfg) -> CheckResult: ...
def check_c8(raw, grid, case_ids, cfg) -> CheckResult: ...
def check_c5(raw, grid, case_ids) -> CheckResult: ...
def check_c6(raw, grid, cfg) -> tuple[CheckResult, str | None]: ...   # (result, 'R1' | 'R2' | None = stop)
def check_c7(aligned: Aligned) -> CheckResult: ...
def check_c9(params: pd.DataFrame, xs: XScaler) -> CheckResult: ...
def check_c10(aligned: Aligned, split: np.ndarray, ys: YScaler, cfg: dict) -> CheckResult: ...  # A1: cfg gives checks.c10_rel_tol
def write_report(path: Path, run_meta: dict, results: list[CheckResult]) -> None: ...

# db.py
def connect(path: Path) -> sqlite3.Connection: ...      # creates parent dirs; PRAGMA foreign_keys = ON
def init_schema(con) -> None: ...                       # CREATE TABLE IF NOT EXISTS, all tables in §1.6
def start_run(con, task: str, command: str, cfg: dict, config_path: Path,
              seed: int | None, model: str | None = None) -> str: ...
def finish_run(con, run_id: str, status: str, outputs: dict) -> None: ...
def write_checks(con, run_id: str, results: list[CheckResult]) -> None: ...
def write_task1_tables(con, run_id: str, params: pd.DataFrame, split: np.ndarray,
                       raw: RawProduction, aligned: Aligned, xs: XScaler, ys: YScaler) -> None: ...
def read_task1_arrays(con) -> dict[str, np.ndarray]: ...  # rebuilds the arrays.npz keys from the tables
```

`read_task1_arrays` (review): `rate_definition` comes from `runs.outputs` of the single run in `parameters.run_id`. **F2 (follow-up, Task 2 branch):** it raises `ContractError` unless that run has `status = 'ok'` and `rate_definition` is `R1` or `R2`. Reason: §1.4 commits the tables before it writes the npz. If a later step fails, the tables belong to a run with status `failed`, and today Tasks 2 to 5 would read them with `rate_definition = ''`.

### 1.4 Processing order in `run_prepare`

1. Load the config. Open the DB. Run `init_schema`. Call `start_run(task='task1', command='prepare', ...)`, which writes a `runs` row with status `running`. Open the log file.
2. Run `load_parameters` and `load_production`.
3. Run C1. Then compute `grid = history_grid(raw)`.
4. Run C2, C3, C4 and C8 in this order.
5. Run C5 (report), then C6, which selects the rate definition (§2.5).
6. Run `assign_split`, then `align_all`, then the V9 assertions (§2.8).
7. Fit `XScaler` and `YScaler`. Transform X, Y and hist.
8. Run C7, C9 and C10 (report only).
9. In one transaction, run `write_task1_tables` and `write_checks`. Then write `arrays.npz` and `data_checks.md`. Then call `finish_run(status='ok', outputs=...)`.

**Stop rule:** steps 3 and 4 stop at the first failed check. Step 5 stops when C6 returns `None`. On a stop, the pipeline does five things:

- Writes the `data_checks` rows of the checks that ran.
- Writes `data_checks.md`, which shows `FAILED` and lists the checks that did not run as "not run".
- Calls `finish_run(status='failed')`.
- Leaves the Task 1 data tables and `arrays.npz` untouched.
- Raises `DataCheckError`, so the CLI exits with code 1.

Any other exception also calls `finish_run(status='failed')` and is then re-raised.

### 1.5 Output `data/processed/arrays.npz`

The file is written with `np.savez` (uncompressed) and must load with `np.load(path, allow_pickle=False)`. The row order is ascending `case_id` everywhere.

| Key | Shape | dtype | Content |
| --- | --- | --- | --- |
| `X` | (n, 4) | float64 | Scaled parameters in [0, 1], columns in config order |
| `Y` | (n, T, 3) | float64 | Scaled rates [oil, gas, water] |
| `hist` | (T, 3) | float64 | Scaled `RAMP_History` rates, same definition and factors |
| `dates` | (T,) | datetime64[D] | History dates |
| `t_days` | (T,) | float64 | Days from `dates[0]` |
| `case_id` | (n,) | int64 | Case numbers |
| `split` | (n,) | `<U5` | `train`, `val` or `test` |
| `x_names` | (4,) | str | `fault_trans, poro_mult, perm_mult, aquifer_pv` |
| `fluids` | (3,) | str | `oil, gas, water` |
| `x_lower`, `x_upper` | (4,) | float64 | X bounds, physical units |
| `y_factor` | (3,) | float64 | s_f in STB/d, MSCF/d, STB/d |
| `X_phys` | (n, 4) | float64 | Parameters, physical units |
| `Y_phys` | (n, T, 3) | float64 | Rates, physical units |
| `hist_phys` | (T, 3) | float64 | History rates, physical units |
| `rate_definition` | () | str | `R1` or `R2` |
| `run_id` | () | str | Prepare run that wrote the file |

### 1.6 Database `data/processed/dwc26.sqlite`

Task 1 creates the tables below. It also creates `runs`, which every later task uses. Dates are TEXT in ISO 8601. Task 1 tables (`parameters`, `production_native`, `production_grid`, `history`, `scalers`) are emptied and refilled in one transaction on each successful run. The delete order is `production_grid`, `production_native`, `history`, `scalers`, `parameters`. `runs` and `data_checks` are append-only. Task 1 does not touch tables of later tasks.

```sql
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,          -- '<command>-YYYYmmddTHHMMSSffffffZ' (UTC start)
  task          TEXT NOT NULL,             -- 'task1'
  command       TEXT NOT NULL,             -- 'prepare'
  model         TEXT,                      -- NULL for Task 1
  config        TEXT NOT NULL,             -- resolved config as JSON (sort_keys=True)
  config_path   TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,             -- sha256 of the config JSON text
  seed          INTEGER,
  git_commit    TEXT NOT NULL,             -- `git rev-parse HEAD`, or 'unknown'
  git_dirty     INTEGER,                   -- 1 if `git status --porcelain --untracked-files=no` is non-empty; NULL if unknown
  start         TEXT NOT NULL,             -- 'YYYY-MM-DDTHH:MM:SS.ffffffZ'
  duration_s    REAL,                      -- NULL while running
  status        TEXT NOT NULL CHECK (status IN ('running','ok','failed')),
  outputs       TEXT,                      -- JSON, see below
  python        TEXT,                      -- platform.python_version()
  packages      TEXT                       -- JSON {numpy, pandas, openpyxl, scipy, pyyaml: version}
);

CREATE TABLE IF NOT EXISTS parameters (
  case_id     INTEGER PRIMARY KEY,
  split       TEXT NOT NULL CHECK (split IN ('train','val','test')),
  fault_trans REAL NOT NULL,
  poro_mult   REAL NOT NULL,
  perm_mult   REAL NOT NULL,
  aquifer_pv  REAL NOT NULL,
  run_id      TEXT NOT NULL REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS production_native (
  case_id  INTEGER NOT NULL,               -- 1..n; 0 = RAMP_History
  quantity TEXT NOT NULL CHECK (quantity IN ('oil_rate','oil_cum','gas_rate','gas_cum','water_rate','water_cum')),
  date     TEXT NOT NULL,                  -- 'YYYY-MM-DDTHH:MM:SS.sss' ('.ssssss' under date_round: us)
  t_days   REAL NOT NULL,                  -- days from the first history date
  value    REAL NOT NULL,                  -- STB/d, MSCF/d, STB or MSCF by quantity
  PRIMARY KEY (case_id, quantity, date)
);

CREATE TABLE IF NOT EXISTS production_grid (
  case_id    INTEGER NOT NULL REFERENCES parameters(case_id),
  date_idx   INTEGER NOT NULL,             -- 0..T-1
  date       TEXT NOT NULL,                -- 'YYYY-MM-DD'
  t_days     REAL NOT NULL,
  oil_rate   REAL NOT NULL, gas_rate REAL NOT NULL, water_rate REAL NOT NULL,  -- physical, selected definition
  oil_cum    REAL NOT NULL, gas_cum  REAL NOT NULL, water_cum  REAL NOT NULL,
  PRIMARY KEY (case_id, date_idx)
);

CREATE TABLE IF NOT EXISTS history (
  date_idx   INTEGER PRIMARY KEY,
  date       TEXT NOT NULL,
  t_days     REAL NOT NULL,
  oil_rate   REAL NOT NULL, gas_rate REAL NOT NULL, water_rate REAL NOT NULL,  -- same definition as production_grid (V4)
  oil_cum    REAL NOT NULL, gas_cum  REAL NOT NULL, water_cum  REAL NOT NULL,
  oil_rate_reported   REAL NOT NULL,     -- RAMP_History in the rate sheets, as reported
  gas_rate_reported   REAL NOT NULL,
  water_rate_reported REAL NOT NULL,
  water_inj_rate_reported REAL           -- Field Production sheet, STB/d; definition not verified
);

CREATE TABLE IF NOT EXISTS scalers (
  name      TEXT PRIMARY KEY,              -- x_fault_trans, x_poro_mult, x_perm_mult, x_aquifer_pv, y_oil, y_gas, y_water
  kind      TEXT NOT NULL CHECK (kind IN ('x_minmax','y_max')),
  variable  TEXT NOT NULL,                 -- parameter name or fluid
  fluid     TEXT,                          -- oil/gas/water for y; NULL for x
  factor    REAL,                          -- y: s_f; NULL for x
  lower     REAL,                          -- x only
  upper     REAL,                          -- x only
  unit      TEXT,                          -- y: STB/d or MSCF/d; x: NULL
  source    TEXT NOT NULL,                 -- XScaler.source, or 'max over train cases, all dates'
  fitted_on TEXT NOT NULL,                 -- JSON list of case_ids used. y: train case_ids. x (review): all
                                           -- case_ids of file 01 for design bounds, [] for an override
  run_id    TEXT NOT NULL REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS data_checks (
  run_id   TEXT NOT NULL REFERENCES runs(run_id),
  check_id TEXT NOT NULL,                  -- C1..C10
  status   TEXT NOT NULL CHECK (status IN ('pass','fail','report')),
  action   TEXT NOT NULL,
  detail   TEXT NOT NULL,                  -- JSON of CheckResult.detail
  PRIMARY KEY (run_id, check_id)
);
```

`runs.outputs` for a successful prepare run is JSON with these keys:

- `files`: paths of the DB, the npz, the report and the log.
- `inputs`: `{path: sha256}` of both raw files.
- `arrays_sha256`.
- `rate_definition`.
- `c6_convention`.
- `checks`: `{C1: status, ...}`.
- `n_cases`, `n_dates`.

`arrays_sha256` is the sha256 of the concatenated `np.ascontiguousarray(a).tobytes()` of `X, Y, hist, dates.astype('int64'), t_days, case_id, x_lower, x_upper, y_factor`, in this order. It is deterministic, unlike a hash of the zip file.

### 1.7 Report `outputs/reports/data_checks.md`

The report contains:

- A header with: the run ID, git commit (`+dirty` if dirty), UTC start, config path and sha256, sha256 of both inputs, n, T, first and last date, and the rate definition with the C6 convention.
- One table with the columns Check, Status, Action and one-line Summary, for C1 to C10. Checks that did not run show "not run".
- One section for each check, with the content listed in §2.6.

### 1.8 Run log

The run writes `outputs/logs/<run_id>.log` (Python `logging`, level INFO). It records the config, the input hashes, each check's status, array shapes, the scaler values, the duration and the outputs. The `runs` row is the authoritative record (V8).

## 2. Equations and check definitions

### 2.1 Notation

- i = case (1..n). f = fluid (oil, gas, water). k = 0..T−1 is the history-date index.
- t_k = days from t_0. τ_j = the native report times of one series, in days from t_0.
- C_{i,f}(τ) is the cumulative production. q_{i,f}(τ) is the reported rate.
- d_f is `RAMP_History`. Its reported rate is r_f(t_k), and its cumulative is C_f(t_k).

### 2.2 Time

- Timestamps are parsed with `pd.to_datetime` and rounded to the millisecond (`date_round: ms`). Assumption: this removes float noise from Excel serial dates. Two distinct reports less than 1 ms apart do not occur, and C1 flags any duplicate date that the rounding creates.
- Days: t = (ts − t_0) in integer milliseconds ÷ 86,400,000, as float64. Source: PRD §5, step 4.1, "days ... as floats". It is exact for millisecond timestamps.

### 2.3 History grid

The grid is the set of dates with a non-empty `RAMP_History` cell (source: PRD §1, "the 41 quarterly dates of the observed history"). C1 requires this set to be identical in all six sheets and every date to be at 00:00:00.000. `history_grid` returns it sorted, as datetime64[ms]. `Aligned.dates` is the same array cast to datetime64[D]. T = len(grid), with no constant in the code (V5).

### 2.4 Alignment of cumulatives

C_{i,f}(t_k) = `np.interp(t_k, τ, C(τ))`, applied to each case and to `RAMP_History` by the same function `interp_cum` (source: PRD §5, step 4.2).

- The function requires τ to be strictly increasing and finite. It raises `ContractError` if any t_k < τ_0 or t_k > τ_last, so there is no extrapolation (source: PRD §5). C2 makes this unreachable for valid data.
- When C5 passes, every t_k is a knot, and the piecewise-linear interpolant equals the native value there (source: numpy documentation, `numpy.interp`). Bit-exactness in floating point is an assumption, and acceptance test B08 checks it.

### 2.5 Rate definition

**R1, the mean over the previous quarter, from cumulatives** (source: PRD §5, R1 formula; t_0 rule changed by D-005):

$$
q^{R1}_{f}(t_0) = 0, \qquad q^{R1}_{f}(t_k) = \frac{C_f(t_k) - C_f(t_{k-1})}{t_k - t_{k-1}}, \quad k = 1, \dots, T-1
$$

- The same function `rates_r1` is applied to the case cumulatives and to the `RAMP_History` cumulatives. The history's reported rates are never used for `hist` (V4).
- Mean rate = Δcumulative ÷ Δt. Assumption: the cumulative is the time integral of the rate. Evidence: C6 on the history agrees to 3e-8 (source: profile). R1 is the exact quarter mean, whatever the simulator timestep.
- t_0 rule (D-005): q(t_0) = 0 holds exactly when C(t_0) = 0, that is, when nothing was produced before t_0. C8 checks this. `rates_r1` raises `ContractError` if |C(t_0)| > `c8_abs_tol` for any series and fluid.

**R2, the last reported rate at or before t_k** (source: PRD §5):

$$
q^{R2}_f(t_k) = q_f(\tau_{j^*}), \qquad j^* = \max\{\, j : \tau_j \le t_k \,\}
$$

- `rates_r2` uses `np.searchsorted(tau, t_grid, side='right') - 1` and raises if any index is below 0.
- R2 runs only when the config forces `definition: R2` (see the table below).

**Selection** (D-004):

| `rates.definition` | C6 | Result |
| --- | --- | --- |
| `auto` (default) | pass | R1 |
| `auto` | fail | Stop (`DataCheckError`). A human decision is needed |
| `R1` | any | R1. If C6 fails, the report header says "V4 not verified by C6" |
| `R2` | any | R2. The report header says "V4 not verified: case rates are simulator report rates, history rates are quarter means (C6)" |

Consequence on the real data (source: profile; derived): the history reports the next-quarter mean, so for k = 1..T−1, `hist_phys[k] = r(t_{k−1})` within 3e-8. `hist_phys[0..3] = 0` because history production starts in 1998-Q4. The 37 match-window dates, 1 Jan 1999 to 1 Jan 2008, therefore carry the 37 producing quarters 1998-Q4 to 2007-Q4.

### 2.6 Checks

Relative difference (assumption: symmetric, with zero handled):

$$
\mathrm{rel}(a, b; \varphi) = \begin{cases} 0 & a = b \\ |a-b| \,/\, \max(|a|, |b|, \varphi) & \text{otherwise} \end{cases}
$$

| ID | Definition | Status and action | Detail and report content |
| --- | --- | --- | --- |
| C1 | **Structure.** (a) File 01: the sheet exists, the label column and the 4 configured columns exist, every non-empty label matches `label_pattern`, case IDs are unique, and the 4 values are finite. (b) Each of the 6 data sheets exists. Its row-2 names are `Date` followed by exactly {`Case_i` : i ∈ IDs of file 01} ∪ {`RAMP_History`}, with no duplicate and no other name. (c) Every non-empty row-3 unit cell of a data column has the configured unit inside `[...]`, and at least one is non-empty. (d) The parsed Date column has no duplicates and is identical in the 6 sheets. (e) The non-empty `RAMP_History` dates are identical in the 6 sheets, there are at least 2, and each is at 00:00:00.000. (f) `Field Production` exists with the configured columns. Source: PRD §5 C1. Parts (a) and (c) to (f): assumption (D-006) | `pass`/`fail`. Fail: stop | Every failed item with its sheet and name |
| C2 | For each case i, each of the 6 sheets: min τ ≤ t_0 and max τ ≥ t_{T−1}. Source: PRD §5 C2 | `pass`/`fail`. Fail: stop | Failing (case, sheet, first date, last date) |
| C3 | For each series (cases and `RAMP_History`): no bad cells. Every cumulative ≥ 0 and non-decreasing over the native dates (C(τ_{j+1}) − C(τ_j) ≥ 0, exactly, with no tolerance). Every native rate ≥ 0. Source: PRD §5 C3. Exact comparison: assumption, because rounding preserves order, so exported monotone data stay monotone | `pass`/`fail`. Fail: stop | Count and the first 20 violations (case, quantity, date, value, previous value) |
| C4 | `Field Production` dates = grid (same T, equal timestamps). For each fluid f and k: rel(field_f(t_k), r_f(t_k); 0) ≤ `c4_rel_tol`. Source: PRD §5 C4 | `pass`/`fail`. Fail: stop | Max rel for each fluid. Failing (fluid, date, both values) |
| C8 | For each series (cases and `RAMP_History`) and each cumulative sheet: \|C(t_0)\| ≤ `c8_abs_tol`. Assumption (D-005): needed by R1 at t_0 | `pass`/`fail`. Fail: stop | Failing (case, fluid, value) |
| C5 | For each case and sheet: every t_k is a native date (exact timestamp equality). Source: PRD §5 C5 | `pass` → "use values on the dates". `fail` → "interpolate (np.interp)". Never stops. The code path is the same (§2.4) | Number of (case, sheet, date) cells off the native dates, and the first 20 |
| C6 | With m^next_f(t_k) = (C_f(t_{k+1}) − C_f(t_k)) ÷ (t_{k+1} − t_k) and m^prev_f(t_k) = (C_f(t_k) − C_f(t_{k−1})) ÷ (t_k − t_{k−1}) from the history cumulatives, and φ_f = `c6_floor_frac` · max_k \|r_f(t_k)\|: e_next = max_f max_{k=0..T−2} rel(r_f(t_k), m^next_f(t_k); φ_f) and e_prev = max_f max_{k=1..T−1} rel(r_f(t_k), m^prev_f(t_k); φ_f). Pass iff min(e_next, e_prev) < `c6_rel_tol`. Convention = `next` if e_next ≤ e_prev, else `prev`. Source: PRD §5 C6 (prev). Next-quarter branch and floor: assumption (D-004) | `pass` → R1 under `auto`. `fail` → stop under `auto`. See the selection table in §2.5 | e_next and e_prev for each fluid, the convention, the definition selected, and the (fluid, k) of each maximum |
| C7 | GOR_{i,k} = q_gas ÷ q_oil (MSCF/STB) on the aligned rates, for every case and date with q_oil > 0, and the same for the history. Trend: the OLS slope of GOR on t_k ÷ 365.25 (MSCF/STB per year) for each case with ≥ 3 valid dates. Source: PRD §5 C7. Trend definition: assumption (D-006) | `report`. Never stops | Global min and max with their (case, date). Median. Count excluded (q_oil = 0). Per date: median, min and max across cases. Number of cases with slope > 0, and the median slope. History GOR min, max and slope |
| C9 | X design, for each parameter j: the chosen bounds; sample min and max; end gaps (min − lower) ÷ (upper − lower) and (upper − max) ÷ (upper − lower); quartiles; the KS statistic and p-value of the scaled values against U(0, 1) (`scipy.stats.kstest`); if lower > 0, the same for (ln x − ln lower) ÷ (ln upper − ln lower) (log-uniform); and the 4 × 4 Spearman matrix. Assumption (D-006): informs the Task 4 prior | `report` | All of these |
| C10 | Envelope, with physical aligned rates and training cases only (V1). L_{f,k} = min over train of q_{i,f}(t_k), U_{f,k} = max over train. For each fluid: the count and list of k with hist_f(t_k) < L or > U; max_k hist_f(t_k) ÷ s_f; max over the val and over the test cases of q ÷ s_f. For the dates where all three history rates are 0: max over all cases of q_{i,f}(t_k). Assumption (D-006): tells Task 4 whether the proxy must extrapolate to reach the history. **Amendment A1 (D-009):** the excess at k is e_{f,k} = rel(hist_f(t_k), U_{f,k}; 0) if hist > U, rel(hist_f(t_k), L_{f,k}; 0) if hist < L, else 0. A date counts as outside only if e_{f,k} > `c10_rel_tol` | `report` | All of these. A1 adds the top-level key `tol` (= `c10_rel_tol`), and for each fluid: `n_outside_exact` (the count without tolerance), `max_rel_excess` = max_k e_{f,k}, and `rel_excess` = e_{f,k} in each entry of `outside`. `n_outside`, `n_below`, `n_above` and `outside` use the tolerance |

C1 to C4 and C8 run before the alignment. C5 and C6 run before the alignment, and C6 selects its definition. C7, C9 and C10 run after the scaling.

### 2.7 Scaling

**X, min-max** (source: PRD §5; bounds per D-003):

$$
u_{ij} = \frac{x_{ij} - \ell_j}{h_j - \ell_j}, \qquad x_{ij} = \ell_j + u_{ij}(h_j - \ell_j)
$$

- `x_bounds.source: design` (default): ℓ_j = min_i x_ij and h_j = max_i x_ij over all n cases of file 01, at run time (V5).
  - Assumption (D-003): these design points are inputs fixed before any simulation, and they carry no Y information. PRD §5 classes the X bounds as "a design input, not a learned statistic", so V1 does not apply.
  - `XScaler.source` = `"design: min/max of file 01 cases 1..n"`.
- `x_bounds.source: override`:
  - (ℓ_j, h_j) come from `x_bounds.override` for all 4 names, and `citation` must be non-empty. Otherwise `ConfigError`.
  - Each override must satisfy ℓ_j ≤ min_i x_ij and h_j ≥ max_i x_ij, and ℓ_j < h_j. Otherwise `ConfigError`.
  - `XScaler.source` = the citation.
- Required: h_j > ℓ_j for every j.

**Y, one factor for each fluid** (source: PRD §5):

$$
s_f = \max_{i \in \text{train},\; k} q_{i,f}(t_k), \qquad Y_{i,k,f} = q_{i,f}(t_k)\,/\,s_f, \qquad \text{hist}_{k,f} = q^{\text{hist}}_f(t_k)\,/\,s_f
$$

- `fit_y_scaler` selects the rows with `split == 'train'` and passes only those to the max (V1).
- Required: s_f > 0.
- Scaled values of val, test and hist may exceed 1. They are not clipped.

### 2.8 V9 assertions at module boundaries

| Boundary | Assertion |
| --- | --- |
| `load_parameters` → | case_id int64 unique; 4 float64 columns, finite |
| `history_grid` → | (T,) datetime64[ms], strictly increasing, T ≥ 2, all at midnight |
| `interp_cum` → | (T,) finite |
| `rates_r1`, `rates_r2` → | Output shape = input shape (time on axis −2 for R1), finite, ≥ 0 |
| `align_all` → | `cum`, `rate` (n, T, 3); `hist_cum`, `hist_rate`, `hist_rate_reported` (T, 3); all finite, ≥ 0; `cum` non-decreasing along time |
| `fit_x_scaler`, transform → | X (n, 4), finite, 0 ≤ X ≤ 1 for the n cases |
| `fit_y_scaler`, transform → | `factor` (3,) > 0; Y (n, T, 3), hist (T, 3), finite, ≥ 0 |
| Before `np.savez` | All keys and shapes of §1.5; float arrays float64 |

## 3. Defaults and tuning ranges

Task 1 has no learned hyperparameters, and nothing is tuned on data. These are fixed tolerances and design choices. A change outside the range below needs a new entry in `docs/decisions.md`. (review) `load_config` enforces these ranges: a value outside its range raises `ConfigError`.

| Setting | Default | Allowed range | Source |
| --- | --- | --- | --- |
| `split` | train 1–70, val 71–85, test 86–100 | Fixed | PRD §1 |
| `x_bounds.source` | `design` | `design`, `override` (with citation) | D-003 |
| `rates.definition` | `auto` | `auto`, `R1`, `R2` | D-004 |
| `checks.c4_rel_tol` | 1e-6 | 1e-9 to 1e-4 | PRD §5 C4 |
| `checks.c6_rel_tol` | 0.01 | 1e-6 to 0.05 | PRD §5 C6 |
| `checks.c6_floor_frac` | 1e-3 | 0 to 1e-2 | Assumption: stops near-zero water rates before breakthrough from dominating the relative difference |
| `checks.c8_abs_tol` | 0.0 | 0 to 1 (STB or MSCF) | Assumption: the simulator writes an exact 0 at its start |
| `checks.c10_rel_tol` (A1) | 1e-5 | 0 to 1e-3 | Assumption (D-009): export precision gives excesses up to 8.3e-7 on the real data, so 1e-5 leaves a margin of about 12. This is still 4 orders of magnitude below α = 0.10 in Task 4 |
| `file02.date_round` | `ms` | `ms`, `us` | Assumption, §2.2 |
| `seed` | 0 | Any integer | V7. Unused in Task 1 |

Package pins (V7): numpy 2.3.5, pandas 2.3.3, openpyxl 3.1.5 and scipy 1.17.1 (C9), as installed. The data scientist pins pyyaml and pytest at the versions installed. Task 1 does not import scikit-learn or matplotlib.

## 4. Acceptance tests

`pytest` must pass (V10). The tests in §4.1 use a synthetic pair of xlsx files built by a fixture in `tests/conftest.py`, so they run from a clean clone without raw data. The tests in §4.2 carry the marker `rawdata`, which is registered in `pyproject.toml`. They are skipped when the files in `data/raw/` are missing.

**Synthetic fixture** `synthetic_raw(tmp_path)`. It writes file 01 and file 02 with the real layout:

- File 01: 6 cases labelled `Case 1` to `Case 6`.
- Grid: 5 quarterly history dates. Each case reports on the grid dates plus 2 extra dates in each quarter, one with a time of day (for example 08:41:00.938). Cases 5 and 6 report on different extra dates, so the Date column is a union with empty cells.
- Cumulatives: piecewise linear from known piecewise-constant rates, with C = 0 at t_0 for every case. The history produces from t_1 on.
- Reported history rates: the next-quarter means, as in the real file. At t_{T−1} the value is arbitrary.
- `Field Production` copies the reported history rates. One sheet ends with an all-empty row.
- Config: split train [1, 4], val [5, 5], test [6, 6].
- The fixture returns the two paths and a config dict with all paths in `tmp_path`.

### 4.1 Unit and pipeline tests (synthetic)

| ID | File | Test | Pass criterion |
| --- | --- | --- | --- |
| A01 | `test_load.py` | `load_parameters` on the fixture | case_id 1..6 from "Case N", sorted; columns in config order; float64; no problems |
| A02 | `test_load.py` | `load_production` on the fixture | `native` rows = number of non-empty cells; the all-empty row ignored; time of day kept to the ms; `RAMP_History` → case_id 0; units parsed |
| A03 | `test_align.py` | `to_days` | 2007-12-06 08:41:00.938 from 1998-01-01 gives 3626.361816412037 ± 1e-9 |
| A04 | `test_align.py` | `interp_cum` on a synthetic piecewise-linear cumulative with irregular, fractional knots (PRD §5 test) | Grid points between knots: equal to the analytic value, relative 1e-12. Grid points on knots: exactly equal |
| A05 | `test_align.py` | `interp_cum` with a grid date outside [τ_0, τ_last] | Raises `ContractError` |
| A06 | `test_align.py` | `rates_r1` on cumulatives built from piecewise-constant quarterly rates, shapes (T, 3) and (n, T, 3) | q(t_0) = 0. q(t_k) equals the rate of quarter (t_{k−1}, t_k], relative 1e-12 |
| A07 | `test_align.py` | `rates_r1` with C(t_0) = 1 | Raises `ContractError` |
| A08 | `test_align.py` | `rates_r2` with a grid date between two reports and one on a report | Returns the last rate at or before each grid date |
| A09 | `test_align.py` | V4: fixture case whose cumulatives equal the history's | `aligned.rate[i]` and `aligned.hist_rate` are bitwise equal |
| A10 | `test_check.py` | C1 on corrupted copies: (a) column `Case_3` deleted from `Gas production cumulative`; (b) `RAMP_History` renamed in one sheet; (c) an extra column `Case_999`; (d) unit `[m3/d]` in `Water production rate`; (e) sheet `Water production rate` removed; (f) a file 01 label `Case3` | Each returns `fail` |
| A11 | `test_check.py` | C2: `Oil production cumulative` of case 2 blanked from the last grid date onward | `fail` |
| A12 | `test_check.py` | C3: (a) a case-4 cumulative lowered by 1 at one native date; (b) one native rate set to −1; (c) one cell set to text `"n/a"` | Each returns `fail` |
| A13 | `test_check.py` | C4: one `Field Production` oil value × (1 + 1e-4) | `fail`. The unmodified fixture passes |
| A14 | `test_check.py` | C5: case 5 blanked at one interior grid date in all 6 sheets | C5 `fail`. `run_prepare` completes. The case-5 cumulative at that date equals the linear interpolation of its neighbours; `n_interpolated` > 0 |
| A15 | `test_check.py` | C6: (a) history reported as next-quarter means; (b) as previous-quarter means; (c) one non-zero reported rate × 1.05 | (a) `pass`, `next`; (b) `pass`, `prev`; both select R1. (c) `fail`, and `run_prepare` with `auto` raises `DataCheckError` |
| A16 | `test_check.py` | C8: case 1 oil cumulative at t_0 = 1.0 (still monotone) | `fail`, and `run_prepare` raises `DataCheckError` |
| A17 | `test_check.py` | C7, C9 and C10 on the fixture | Status `report`. The detail holds the keys listed in §2.6 |
| A18 | `test_check.py` | `assert_array` with NaN, a negative value under `nonneg=True`, a wrong shape and a wrong dtype | Each raises `ContractError` |
| A19 | `test_scale.py` | X round trip (PRD §5 test); design bounds | `inverse(transform(X)) == X`, relative 1e-12. lower/upper = column min/max. Each scaled column has min 0 and max 1 |
| A20 | `test_scale.py` | X override | A valid override is accepted, with `source` = citation. An override that excludes a design point → `ConfigError`. `source: override` with an empty citation → `ConfigError` |
| A21 | `test_scale.py` | Y round trip (PRD §5 test) on (n, T, 3) and (T, 3) | Relative 1e-12 |
| A22 | `test_scale.py` | V1: val and test rates × 10; then one train case × 10 | First change: `factor` unchanged. Second: `factor` changes |
| A23 | `test_split.py` | Default split config, case_ids 1..100 (PRD §5 test). Invalid configs | Counts 70, 15, 15. A case in no range or in two ranges → `ConfigError` |
| A24 | `test_db.py` | Schema | `init_schema` twice is idempotent. The columns of each table match §1.6 |
| A25 | `test_prepare.py` | End to end on the fixture | Rows: `parameters` 6, `production_grid` 6·T, `history` T, `scalers` 7, `data_checks` 10, `runs` 1. The `runs` row has `status` `ok`, `duration_s` > 0, non-empty `git_commit`, and `outputs` with every key of §1.6. The npz keys, shapes and dtypes match §1.5, and the file loads with `allow_pickle=False`. `read_task1_arrays(con)` equals the npz arrays exactly. `Y_phys == Y * y_factor` and `hist_phys == hist * y_factor` (relative 1e-12) |
| A26 | `test_prepare.py` | Determinism: run twice | Same `arrays_sha256`. Task 1 table row counts unchanged (replaced, not appended). 2 `runs` rows, 20 `data_checks` rows |
| A27 | `test_prepare.py` | Failure path: the C3 corruption from A12(a), fresh DB and output directory, through `python -m dwc26 prepare` | Exit code 1. `runs.status` is `failed`. `data_checks` has C1, C2 and C3. No npz. Task 1 data tables empty. The report says FAILED |
| A28 (A1) | `test_check.py` | C10 tolerance: on the fixture, set the history rate at one producing date above the training maximum, (a) by a relative 1e-7 and (b) by a relative 1e-3 | (a) `n_outside_exact` = 1, `n_outside` = 0. (b) `n_outside` = 1, `max_rel_excess` ≈ 1e-3. `c10_rel_tol` outside [0, 1e-3] → `ConfigError` |
| A29 (F2) | `test_db.py` | `read_task1_arrays` when the run in `parameters.run_id` has `status = 'failed'` | Raises `ContractError` |

### 4.2 Raw-data tests (marker `rawdata`)

A session-scoped fixture runs `run_prepare` once into a temporary directory, using the real files.

| ID | Test | Pass criterion |
| --- | --- | --- |
| B01 | Shapes and grid | X (100, 4); Y (100, 41, 3); hist (41, 3); dates (41,). `dates[0]` = 1998-01-01, `dates[-1]` = 2008-01-01. 37 dates in [1999-01-01, 2008-01-01]. Split counts 70, 15, 15 |
| B02 | Checks | C1 to C5 and C8 `pass`. C6 `pass` with convention `next` and e_next ≤ 1e-6. C6 detail shows e_prev ≥ 0.5, which documents why the PRD's C6 would fail. `rate_definition` = `R1` |
| B03 | History shift (V4 consequence, §2.5) | `hist_phys[k] == hist_rate_reported[k−1]`, relative 1e-6, for k = 1..40 and all fluids. `hist_phys[0] == 0` |
| B04 | Plausibility (source: profile) | `hist_phys` oil at 1999-01-01 = C_oil(1999-01-01) ÷ 92 ≈ 15,507.77 STB/d, relative 1e-4 |
| B05 | X | Each column of X has min 0 and max 1. `x_lower` and `x_upper` equal the column min and max of file 01 |
| B06 | Y | Over the train cases and all dates, `Y.max()` = 1.0 exactly for each fluid. Y ≥ 0, no NaN |
| B07 | Native store | 41 rows for each quantity with case_id 0. ≥ 41 rows for each case and quantity |
| B08 | C5 exactness | The grid cumulatives of every case equal the native values on those dates bit for bit |

### 4.3 Review checklist (researcher, before merge)

- V1: `fit_y_scaler` uses train rows only (A22). X bounds follow D-003.
- V2: the split is by case only.
- V4: one code path for cases and history (A09, B03).
- V5: a grep of `src/` finds no literal 41, 100, 70, 85, 37, 1998 or 2008, and no hard-coded date.
- V7: pinned versions and all settings in `configs/data.yaml`.
- V8: the `runs` row and the log file exist.
- V9: §2.8.
- V10: one test file for each module.
- V12: `git ls-files data outputs` is empty, and no `.xlsx`, `.sqlite` or `.npz` file is tracked.
- Plausibility: B04. The C7 GOR range. The C9 end gaps and KS. The C10 envelope counts.
- One command from a clean clone: `python -m dwc26 prepare --config configs/data.yaml`, after the two raw files are copied to `data/raw/`. The README states this.

## 5. Consequences for later tasks

- **Task 2:** `Y[:, k]` is the mean rate over the quarter ending at `dates[k]`. If the cases, like the history, start production in 1998-Q4, then `Y[:, 0:4] = 0`. C10 reports the case rates on the zero-history dates.
  - (review) Confirmed by run `prepare-20261012T193743103824Z`: `Y[:, 0:4] = 0` for all 100 cases. On 1999-01-01 all training cases share one oil rate (15,507.77 STB/d), and they agree to 2e-7 on 1999-04-01.
  - (review, D-010) Gas ≈ 0.3633 × oil (C7). In scaled units, Y_gas = Y_oil × (0.992 to 1.006). An equal-weight loss over [oil, gas, water] therefore gives oil about 2/3 of the weight. The Task 2 spec must set the fluid weights.
- **Task 4:**
  - d_f(t_k) = `hist_phys`. Never use the reported history rates, which are one quarter earlier.
  - The prior support is the box [ℓ, h]⁴ = [0, 1]⁴ (D-003), so the posterior cannot leave the design range and the proxy is never asked to extrapolate in X.
  - The bound filter (0.02 in scaled units) is measured from the extreme design points.
  - If C10 shows the history outside the training envelope, expect posterior mass at the box faces. Report it.
    - (review, D-009) On the real data, the C10 run without a tolerance flags oil and gas on 1999-01-01 and 1999-04-01 by relative 6e-10 to 8.3e-7. This is export precision, not a real excess. For Task 4, the history lies inside the training envelope.
    - (A1 implemented) Run `prepare-20261012T201436717380Z` (`arrays_sha256` unchanged, 451fcdab…) with `c10_rel_tol` = 1e-5: 0 dates outside the envelope for all three fluids. Without the tolerance: oil 2, gas 2 (`max_rel_excess` gas 8.26e-7, oil 4.3e-8).
  - ~~If C9 suggests that `perm_mult` was sampled log-uniformly, the uniform prior in linear scaled space needs a decision before Task 4.~~ Closed (D-008): C9 shows uniform sampling in linear space. The uniform prior on u ∈ [0, 1]⁴ stands.
- **Task 5 and the later volume task:**
  - A forecast rate dated t_k is the mean over (t_{k−1}, t_k]. The volume in that quarter = q(t_k) · (t_k − t_{k−1}).
  - The reported history value at 2008-01-01 is, by the next-quarter pattern, the 2008-Q1 mean. Assumption: the profile checks the pattern for k = 0..39 only. It is not used in Task 1. It can serve as a one-step check of the forecast at 2008-04-01, because no model is fitted to it.
  - The later task must check the date convention of the file 09 template.
