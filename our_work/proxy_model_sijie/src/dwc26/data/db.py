"""SQLite store `data/processed/dwc26.sqlite`: schema, Task 1 writes, run log (V8).

Spec task1 section 1.6. `runs` and `data_checks` are append-only. The Task 1
tables are emptied and refilled in the caller's transaction on each successful
prepare run. Write functions do not commit, except start_run and finish_run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sqlite3
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from our_work.proxy_model_sijie.src.dwc26.data.check import CheckResult, ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.data.scale import Y_SOURCE, load_scalers

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,          -- '<command>-YYYYmmddTHHMMSSffffffZ' (UTC start)
  task          TEXT NOT NULL,             -- 'task1', 'task2'
  command       TEXT NOT NULL,             -- 'prepare', 'train'
  model         TEXT,                      -- NULL for Task 1; rnn, lstm, dmd, b0, b1 for Task 2
  config        TEXT NOT NULL,             -- resolved config as JSON (sort_keys=True)
  config_path   TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,             -- sha256 of the config JSON text
  seed          INTEGER,
  git_commit    TEXT NOT NULL,             -- `git rev-parse HEAD`, or 'unknown'
  git_dirty     INTEGER,                   -- 1 if `git status --porcelain --untracked-files=no` is non-empty; NULL if unknown
  start         TEXT NOT NULL,             -- 'YYYY-MM-DDTHH:MM:SS.ffffffZ'
  duration_s    REAL,                      -- NULL while running
  status        TEXT NOT NULL CHECK (status IN ('running','ok','failed')),
  outputs       TEXT,                      -- JSON
  python        TEXT,                      -- platform.python_version()
  packages      TEXT                       -- JSON {numpy, pandas, openpyxl, scipy, pyyaml, torch, scikit-learn: version}
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
  date     TEXT NOT NULL,                  -- 'YYYY-MM-DDTHH:MM:SS.sss'
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
  fitted_on TEXT NOT NULL,                 -- JSON list of case_ids used
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

CREATE TABLE IF NOT EXISTS predictions (
  run_id    TEXT NOT NULL REFERENCES runs(run_id),
  variant   TEXT NOT NULL,                   -- 'A','B','C' (rnn, lstm) or 'main'
  case_id   INTEGER NOT NULL,                -- no FK: Task 1 deletes and refills `parameters` (task1 1.6)
  split     TEXT NOT NULL CHECK (split IN ('train','val','test')),   -- Task 2 writes train and val only (V3)
  date_idx  INTEGER NOT NULL,                -- 0..T_fit-1
  date      TEXT NOT NULL,                   -- 'YYYY-MM-DD'
  oil       REAL NOT NULL, gas REAL NOT NULL, water REAL NOT NULL,        -- physical (STB/d, MSCF/d, STB/d), clipped >= 0
  oil_std   REAL, gas_std REAL, water_std REAL,                          -- physical; NULL when predict_std is None
  PRIMARY KEY (run_id, variant, case_id, date_idx)
);

-- Task 3 (spec task3 section 1.6; D-028, D-029). Order: each trigger's tables exist when it is created.
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
"""

TASK1_DELETE_ORDER = ("production_grid", "production_native", "history", "scalers", "parameters")
# Recorded for every run (D-018). Read through importlib.metadata: torch and
# scikit-learn are not imported.
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "openpyxl": "openpyxl", "scipy": "scipy", "pyyaml": "PyYAML",
            "torch": "torch", "scikit-learn": "scikit-learn", "matplotlib": "matplotlib"}

_T0: dict[str, float] = {}   # run_id -> perf_counter at start


def connect(path: Path) -> sqlite3.Connection:
    """Opens (and creates) the database; creates parent dirs; foreign keys on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def _git(*args: str) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def git_state() -> tuple[str, int | None]:
    """(commit, dirty) of the repository that holds this package."""
    head = _git("rev-parse", "HEAD")
    if head is None or not head.strip():
        return "unknown", None
    status = _git("status", "--porcelain", "--untracked-files=no")
    dirty = None if status is None else int(bool(status.strip()))
    return head.strip(), dirty


def package_versions() -> dict[str, str]:
    out = {}
    for key, dist in PACKAGES.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = "not installed"
    return out


def config_json(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def start_run(con, task: str, command: str, cfg: dict, config_path: Path,
              seed: int | None, model: str | None = None) -> str:
    """Inserts a `runs` row with status 'running' and commits. Returns run_id."""
    now = dt.datetime.now(dt.timezone.utc)
    run_id = f"{command}-{now:%Y%m%dT%H%M%S%f}Z"
    text = config_json(cfg)
    commit, dirty = git_state()
    con.execute(
        "INSERT INTO runs (run_id, task, command, model, config, config_path, config_sha256, seed, git_commit, "
        "git_dirty, start, duration_s, status, outputs, python, packages) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'running', NULL, ?, ?)",
        (run_id, task, command, model, text, Path(config_path).as_posix(),
         hashlib.sha256(text.encode("utf-8")).hexdigest(), seed, commit, dirty,
         now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), platform.python_version(),
         json.dumps(package_versions(), sort_keys=True)),
    )
    con.commit()
    _T0[run_id] = time.perf_counter()
    return run_id


def finish_run(con, run_id: str, status: str, outputs: dict) -> None:
    """Sets status ('ok' or 'failed'), duration and outputs, and commits."""
    if status not in ("ok", "failed"):
        raise ValueError(f"finish_run: status must be 'ok' or 'failed', got {status!r}")
    if run_id in _T0:
        duration = time.perf_counter() - _T0.pop(run_id)
    else:
        start = con.execute("SELECT start FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
        t0 = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)
        duration = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
    cur = con.execute("UPDATE runs SET status = ?, duration_s = ?, outputs = ? WHERE run_id = ?",
                      (status, duration, json.dumps(outputs, sort_keys=True), run_id))
    if cur.rowcount != 1:
        raise ContractError(f"finish_run: no run {run_id}")
    con.commit()


def write_checks(con, run_id: str, results: list[CheckResult]) -> None:
    con.executemany(
        "INSERT INTO data_checks (run_id, check_id, status, action, detail) VALUES (?, ?, ?, ?, ?)",
        [(run_id, r.check_id, r.status, r.action, json.dumps(r.detail, sort_keys=True)) for r in results],
    )


def _date_text(d: np.ndarray, unit: str) -> list[str]:
    return np.datetime_as_string(d, unit=unit).tolist()


def write_task1_tables(con, run_id: str, params: pd.DataFrame, split: np.ndarray,
                       raw, aligned, xs, ys) -> None:
    """Replaces the Task 1 tables (no commit: run inside one transaction)."""
    from our_work.proxy_model_sijie.src.dwc26.data.align import to_days, values_on_grid

    for t in TASK1_DELETE_ORDER:
        con.execute(f"DELETE FROM {t}")

    names = list(xs.names)
    cols = ", ".join(names)
    marks = ", ".join("?" for _ in names)
    prm = params.sort_values("case_id")
    if not np.array_equal(prm["case_id"].to_numpy(), aligned.case_ids):
        raise ContractError("write_task1_tables: parameters and aligned case_ids differ")
    con.executemany(
        f"INSERT INTO parameters (case_id, split, {cols}, run_id) VALUES (?, ?, {marks}, ?)",
        [(int(cid), str(s), *map(float, row), run_id)
         for cid, s, row in zip(prm["case_id"], split, prm[names].to_numpy(dtype=np.float64))],
    )

    t0 = aligned.dates[0]
    nat = raw.native
    unit = np.datetime_data(nat["date"].dtype)[0] if len(nat) else "ms"
    d = nat["date"].to_numpy()
    con.executemany(
        "INSERT INTO production_native (case_id, quantity, date, t_days, value) VALUES (?, ?, ?, ?, ?)",
        zip(nat["case_id"].astype(int).tolist(), nat["quantity"].astype(str).tolist(),
            _date_text(d, unit), to_days(d, t0).tolist(), nat["value"].astype(float).tolist()),
    )

    dates = _date_text(aligned.dates, "D")
    t = aligned.t_days.tolist()
    rows = []
    for i, cid in enumerate(aligned.case_ids):
        for k in range(len(dates)):
            rows.append((int(cid), k, dates[k], t[k], *aligned.rate[i, k].tolist(), *aligned.cum[i, k].tolist()))
    con.executemany(
        "INSERT INTO production_grid (case_id, date_idx, date, t_days, oil_rate, gas_rate, water_rate, "
        "oil_cum, gas_cum, water_cum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    inj = [None] * len(dates)
    if raw.field is not None and "water_inj_rate" in raw.field.columns:
        fd = raw.field["date"].to_numpy()
        fv = raw.field["water_inj_rate"].to_numpy(dtype=np.float64)
        order = np.argsort(fd, kind="mergesort")
        grid = aligned.dates.astype(fd.dtype)
        vals = values_on_grid(fd[order], fv[order], grid, "Field Production water_inj_rate")
        inj = [float(v) if np.isfinite(v) else None for v in vals]
    con.executemany(
        "INSERT INTO history (date_idx, date, t_days, oil_rate, gas_rate, water_rate, oil_cum, gas_cum, "
        "water_cum, oil_rate_reported, gas_rate_reported, water_rate_reported, water_inj_rate_reported) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(k, dates[k], t[k], *aligned.hist_rate[k].tolist(), *aligned.hist_cum[k].tolist(),
          *aligned.hist_rate_reported[k].tolist(), inj[k]) for k in range(len(dates))],
    )

    x_fitted = json.dumps([int(i) for i in prm["case_id"]]) if xs.source.startswith("design") else "[]"
    srows = [(f"x_{n}", "x_minmax", n, None, None, float(xs.lower[j]), float(xs.upper[j]), None, xs.source,
              x_fitted, run_id) for j, n in enumerate(names)]
    for f, fluid in enumerate(ys.fluids):
        srows.append((f"y_{fluid}", "y_max", fluid, fluid, float(ys.factor[f]), None, None,
                      _rate_unit(raw, fluid), Y_SOURCE, json.dumps(list(ys.fitted_on)), run_id))
    con.executemany(
        "INSERT INTO scalers (name, kind, variable, fluid, factor, lower, upper, unit, source, fitted_on, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", srows)


def _rate_unit(raw, fluid: str) -> str | None:
    units = [u for u in raw.units.get(f"{fluid}_rate", []) if u]
    return units[0] if units else None


def str_array(values) -> np.ndarray:
    """Unicode array with the minimal itemsize (same construction everywhere)."""
    return np.array([str(v) for v in values])


RATE_DEFINITIONS = ("R1", "R2")


def _task1_run_definition(con, run_id: str) -> str:
    """F2: the run that wrote the Task 1 tables must have status 'ok' and a
    rate_definition of R1 or R2 in its outputs. Returns the definition."""
    row = con.execute("SELECT status, outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ContractError(f"read_task1_arrays: the Task 1 tables name run {run_id}, which has no row in runs")
    status, outputs = row
    if status != "ok":
        raise ContractError(f"read_task1_arrays: run {run_id} that wrote the Task 1 tables has status "
                            f"'{status}', not 'ok'; rerun prepare")
    try:
        definition = json.loads(outputs).get("rate_definition") if outputs else None
    except (json.JSONDecodeError, AttributeError):
        definition = None
    if definition not in RATE_DEFINITIONS:
        raise ContractError(f"read_task1_arrays: run {run_id} has rate_definition {definition!r} in "
                            f"runs.outputs, expected one of {list(RATE_DEFINITIONS)}")
    return definition


def read_task1_arrays(con) -> dict[str, np.ndarray]:
    """Rebuilds the arrays.npz keys from the Task 1 tables. Raises ContractError
    unless the run that wrote them has status 'ok' and rate_definition R1 or R2 (F2)."""
    xs, ys = load_scalers(con)
    names = list(xs.names)
    prm = con.execute(f"SELECT case_id, split, {', '.join(names)}, run_id FROM parameters ORDER BY case_id").fetchall()
    if not prm:
        raise ContractError("read_task1_arrays: the parameters table is empty; run prepare first")
    case_id = np.array([r[0] for r in prm], dtype=np.int64)
    split = np.array([r[1] for r in prm], dtype="<U5")
    X_phys = np.array([r[2:2 + len(names)] for r in prm], dtype=np.float64)
    run_ids = {r[-1] for r in prm}
    if len(run_ids) != 1:
        raise ContractError(f"read_task1_arrays: parameters from several runs {run_ids}")
    run_id = run_ids.pop()
    rate_definition = _task1_run_definition(con, run_id)

    hist_rows = con.execute(
        "SELECT date, t_days, oil_rate, gas_rate, water_rate FROM history ORDER BY date_idx").fetchall()
    dates = np.array([r[0] for r in hist_rows], dtype="datetime64[D]")
    t_days = np.array([r[1] for r in hist_rows], dtype=np.float64)
    hist_phys = np.array([r[2:5] for r in hist_rows], dtype=np.float64)
    T = len(dates)
    g = con.execute(
        "SELECT case_id, date_idx, oil_rate, gas_rate, water_rate FROM production_grid "
        "ORDER BY case_id, date_idx").fetchall()
    if len(g) != len(case_id) * T:
        raise ContractError(f"read_task1_arrays: production_grid has {len(g)} rows, expected {len(case_id) * T}")
    Y_phys = np.array([r[2:5] for r in g], dtype=np.float64).reshape(len(case_id), T, len(FLUIDS))

    return {
        "X": xs.transform(X_phys),
        "Y": ys.transform(Y_phys),
        "hist": ys.transform(hist_phys),
        "dates": dates,
        "t_days": t_days,
        "case_id": case_id,
        "split": split,
        "x_names": str_array(names),
        "fluids": str_array(ys.fluids),
        "x_lower": xs.lower,
        "x_upper": xs.upper,
        "y_factor": ys.factor,
        "X_phys": X_phys,
        "Y_phys": Y_phys,
        "hist_phys": hist_phys,
        "rate_definition": np.array(str(rate_definition)),
        "run_id": np.array(str(run_id)),
    }


SPLITS = ("train", "val", "test")


def write_predictions(con, run_id: str, variant: str, case_id: np.ndarray, split: np.ndarray,
                      dates: np.ndarray, Y_phys: np.ndarray, S_phys: np.ndarray | None,
                      *, allow_test: bool = False) -> int:
    """Writes the predictions of one variant (spec task2 section 1.6). Values are
    physical and clipped (>= 0). No commit. Returns the number of rows.
    ContractError for a test row unless allow_test (Task 3, after the freeze; V3)."""
    case_id = np.asarray(case_id)
    split = np.asarray(split).astype(str)
    dates = np.asarray(dates).astype("datetime64[D]")
    n, T = len(case_id), len(dates)
    if case_id.ndim != 1 or case_id.dtype.kind not in "iu":
        raise ContractError(f"write_predictions: case_id must be a 1-D integer array, got {case_id.dtype} {case_id.shape}")
    if split.shape != (n,):
        raise ContractError(f"write_predictions: split {split.shape}, expected ({n},)")
    bad = sorted(set(split.tolist()) - set(SPLITS))
    if bad:
        raise ContractError(f"write_predictions: unknown split values {bad}")
    if (split == "test").any() and not allow_test:
        raise ContractError("write_predictions: test rows are written only after the freeze (V3)")
    assert_array("write_predictions Y_phys", Y_phys, (n, T, len(FLUIDS)), nonneg=True)
    if S_phys is not None:
        assert_array("write_predictions S_phys", S_phys, (n, T, len(FLUIDS)), nonneg=True)
    ds = np.datetime_as_string(dates, unit="D").tolist()
    rows = []
    for i in range(n):
        for k in range(T):
            s = (None, None, None) if S_phys is None else tuple(float(v) for v in S_phys[i, k])
            rows.append((run_id, variant, int(case_id[i]), str(split[i]), k, ds[k],
                         *(float(v) for v in Y_phys[i, k]), *s))
    con.executemany(
        "INSERT INTO predictions (run_id, variant, case_id, split, date_idx, date, oil, gas, water, "
        "oil_std, gas_std, water_std) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)
