"""Candidates, the `metrics` table and run lookups (spec task3 sections 1.3, 1.5.1 and 1.6; D-029).

`METRICS` is the catalogue of section 1.6: `write_metrics` refuses a name, fluid,
split, idx or model outside it, and a value that is not finite (None = undefined).
Flags are stored as 1.0 and 0.0, counts as floats.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from dwc26.data.check import ContractError
from dwc26.data.load import FLUIDS
from dwc26.evaluation.events import EVENT_STATS, EVENTS

PROXY_MODELS = ("rnn", "lstm", "dmd")
BENCH_MODELS = ("b0", "b1")
NEURAL_MODELS = ("rnn", "lstm")
B2 = "b2"
SPLIT_VALUES = ("train", "val", "test", "none")
FLUID_VALUES = (*FLUIDS, "all")


@dataclass(frozen=True)
class Candidate:
    proxy_id: str              # '<run_id>/<variant>'
    model: str                 # 'b0' | 'b1' | 'rnn' | 'lstm' | 'dmd'
    variant: str
    run_id: str
    is_proxy: bool
    strategy: str | None       # cfg['strategy'][model] for proxies; None for benchmarks
    label: str                 # '<model>/<variant>', e.g. 'lstm/A'

    def to_json(self) -> dict:
        return {"proxy_id": self.proxy_id, "model": self.model, "variant": self.variant, "run_id": self.run_id,
                "is_proxy": self.is_proxy, "strategy": self.strategy, "label": self.label}

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        return cls(**{k: d[k] for k in ("proxy_id", "model", "variant", "run_id", "is_proxy", "strategy", "label")})


@dataclass(frozen=True)
class MetricRow:
    proxy_id: str              # candidate proxy ID, or 'b2'
    split: str                 # 'train' | 'val' | 'test' | 'none'
    fluid: str                 # 'oil' | 'gas' | 'water' | 'all'
    metric: str                # a key of METRICS
    idx: int                   # horizon h, date_idx, or -1
    value: float | None


@dataclass(frozen=True)
class MetricSpec:
    fluids: tuple[str, ...]
    idx: str                   # 'none' (-1) | 'date' (0..T-1) | 'horizon' (1..H)
    splits: tuple[str, ...]
    models: str                # 'all' | 'proxies' | 'proxies_b2' | 'neural'


_F3 = FLUIDS
_ALL = ("all",)
_TVT = ("train", "val", "test")
_VT = ("val", "test")
METRICS: dict[str, MetricSpec] = {
    **{m: MetricSpec(_F3, "none", _TVT, "all") for m in ("rmse", "nrmse", "r2_pooled", "r2_case")},
    "j": MetricSpec(_ALL, "none", _TVT, "all"),
    "n_clipped_pred": MetricSpec(_ALL, "none", _TVT, "all"),
    "rmse_by_date": MetricSpec(_F3, "date", ("val",), "all"),
    **{f"ev_{e}_{s}": MetricSpec(_ALL, "none", _VT, "all") for e in EVENTS for s in EVENT_STATS},
    **{m: MetricSpec(_ALL, "none", _VT, "all") for m in ("rho_O", "r2_O", "local_dO_median")},
    "local_rmse": MetricSpec(_F3, "none", _VT, "all"),
    **{m: MetricSpec(_F3, "none", _VT, "proxies") for m in ("coverage", "interval_width")},
    "p1_n_clipped": MetricSpec(_ALL, "none", _VT, "proxies"),
    **{m: MetricSpec(("oil",), "none", _VT, "proxies")
       for m in ("p2_n_rising", "p2_max_ratio", "p3_R", "p3_n_used", "p3_D_hist_true")},
    **{m: MetricSpec(("gas",), "none", _VT, "all") for m in ("gor_pred_min", "gor_pred_max", "gor_flag")},
    **{m: MetricSpec(("gas",), "none", _VT, "proxies") for m in ("gor_cont_min", "gor_cont_max", "gor_cont_flag")},
    **{m: MetricSpec(_ALL, "none", ("none",), "all") for m in ("cost_fit_s", "cost_run_s", "cost_predict_s")},
    **{m: MetricSpec(_ALL, "none", ("none",), "neural") for m in ("n_trials_max_epochs", "n_members_max_epochs")},
    "bt_rmse": MetricSpec(_F3, "horizon", _VT, "proxies_b2"),
    **{m: MetricSpec(_F3, "none", _VT, "proxies_b2") for m in ("bt_rmse_v6", "bt_rmse_all")},
    "bt_score": MetricSpec(_ALL, "none", _VT, "proxies_b2"),
    "bt_n_clipped": MetricSpec(_ALL, "none", _VT, "proxies"),
    "bt_v6_pass": MetricSpec(("oil", "water"), "none", _VT, "proxies"),
}


def model_allowed(spec: MetricSpec, model: str) -> bool:
    if spec.models == "all":
        return model in PROXY_MODELS + BENCH_MODELS
    if spec.models == "proxies":
        return model in PROXY_MODELS
    if spec.models == "proxies_b2":
        return model in PROXY_MODELS + (B2,)
    return model in NEURAL_MODELS


def expected_metrics(model: str, split: str) -> list[tuple[str, str]]:
    """(metric, fluid) pairs with idx -1 that the catalogue lists for this model and split
    (evaluate rows only: the backtest metrics and the per-date and per-horizon rows excluded)."""
    out = []
    for name, spec in METRICS.items():
        if name.startswith("bt_") or spec.idx != "none" or split not in spec.splits:
            continue
        if model_allowed(spec, model):
            out += [(name, f) for f in spec.fluids]
    return out


# --- candidates (section 1.5.1) ----------------------------------------------------------

def _train_runs(con, model: str) -> list[dict]:
    rows = con.execute("SELECT run_id, status, git_dirty, start, outputs FROM runs WHERE task = 'task2' "
                       "AND command = 'train' AND model = ? ORDER BY start", (model,)).fetchall()
    return [{"run_id": r[0], "status": r[1], "git_dirty": r[2], "start": r[3],
             "outputs": json.loads(r[4] or "{}")} for r in rows]


def _full_window_ok(r: dict, sha: str) -> bool:
    o = r["outputs"]
    return (r["status"] == "ok" and o.get("t_cut") is None and o.get("settings_from") is None
            and o.get("arrays_sha256") == sha)


def resolve_candidates(con, cfg: dict, arrays_sha256: str, models: list[str] | None = None) -> list[Candidate]:
    """One Candidate for each variant of the train run of each model in candidates.runs (config
    order; restricted to `models` if given). ContractError if a run is missing or does not qualify."""
    out = []
    require_clean = cfg["candidates"]["require_clean"]
    for model, choice in cfg["candidates"]["runs"].items():
        if models is not None and model not in models:
            continue
        runs = _train_runs(con, model)
        if choice == "latest":
            ok = [r for r in runs if _full_window_ok(r, arrays_sha256)]
            if not ok:
                raise ContractError(f"candidates: no ok full-window task2 train run of {model} with arrays_sha256 "
                                    f"{arrays_sha256[:12]}...")
            run = max(ok, key=lambda r: r["start"])
        else:
            hit = [r for r in runs if r["run_id"] == choice]
            if not hit:
                raise ContractError(f"candidates.runs.{model}: {choice} is not a task2 train run of {model}")
            run = hit[0]
            if not _full_window_ok(run, arrays_sha256):
                raise ContractError(f"candidates.runs.{model}: {choice} must have status ok, t_cut null, "
                                    f"settings_from null and arrays_sha256 {arrays_sha256[:12]}...")
        if require_clean and run["git_dirty"] != 0:
            raise ContractError(f"candidates: run {run['run_id']} of {model} has git_dirty = {run['git_dirty']}; "
                                "require_clean needs code from a clean commit (V8)")
        variants = sorted(run["outputs"].get("variants", {}))
        if not variants:
            raise ContractError(f"candidates: run {run['run_id']} lists no variants")
        if model in BENCH_MODELS and variants != ["main"]:
            raise ContractError(f"candidates: benchmark run {run['run_id']} must have the single variant 'main', "
                                f"got {variants}")
        is_proxy = model in PROXY_MODELS
        for v in variants:
            out.append(Candidate(proxy_id=f"{run['run_id']}/{v}", model=model, variant=v, run_id=run["run_id"],
                                 is_proxy=is_proxy, strategy=cfg["strategy"][model] if is_proxy else None,
                                 label=f"{model}/{v}"))
    return out


# --- metrics table -------------------------------------------------------------------------

def _model_of(con, proxy_id: str, cache: dict) -> str | None:
    if proxy_id == B2:
        return B2
    if proxy_id not in cache:
        run = proxy_id.split("/")[0]
        row = con.execute("SELECT model FROM runs WHERE run_id = ?", (run,)).fetchone()
        cache[proxy_id] = row[0] if row else None
    return cache[proxy_id]


def _history_T(con) -> int | None:
    try:
        n = con.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    except Exception:  # noqa: BLE001 - no history table
        return None
    return int(n) if n else None


def validate_row(r: MetricRow, model: str | None, T: int | None, H: int | None) -> None:
    spec = METRICS.get(r.metric)
    if spec is None:
        raise ContractError(f"write_metrics: unknown metric {r.metric!r}")
    if r.split not in spec.splits:
        raise ContractError(f"write_metrics: {r.metric} is not defined for split {r.split!r} (allowed {spec.splits})")
    if r.fluid not in spec.fluids:
        raise ContractError(f"write_metrics: {r.metric} is not defined for fluid {r.fluid!r} (allowed {spec.fluids})")
    if isinstance(r.idx, bool) or not isinstance(r.idx, (int, np.integer)):
        raise ContractError(f"write_metrics: idx of {r.metric} must be an integer, got {r.idx!r}")
    idx = int(r.idx)
    if spec.idx == "none" and idx != -1:
        raise ContractError(f"write_metrics: {r.metric} takes idx -1, got {idx}")
    if spec.idx == "date" and not (0 <= idx and (T is None or idx <= T - 1)):
        raise ContractError(f"write_metrics: {r.metric} idx {idx} outside 0..T-1 (T = {T})")
    if spec.idx == "horizon" and not (1 <= idx and (H is None or idx <= H)):
        raise ContractError(f"write_metrics: {r.metric} idx {idx} outside 1..H (H = {H})")
    if r.proxy_id == B2 and spec.models != "proxies_b2":
        raise ContractError(f"write_metrics: {r.metric} is not a B2 metric")
    if model is not None and not model_allowed(spec, model):
        raise ContractError(f"write_metrics: {r.metric} does not apply to model {model!r} ({r.proxy_id})")
    if r.value is not None:
        if isinstance(r.value, bool) or not isinstance(r.value, (int, float, np.integer, np.floating)):
            raise ContractError(f"write_metrics: value of {r.metric} must be a number or None, got {r.value!r}")
        if not math.isfinite(float(r.value)):
            raise ContractError(f"write_metrics: value of {r.metric} ({r.proxy_id}, {r.fluid}) is not finite; "
                                "store an undefined value as None")


def write_metrics(con, run_id: str, rows: list[MetricRow], *, T: int | None = None, H: int | None = None) -> int:
    """Inserts the rows (no commit). V9: every row is in METRICS, the value is finite or None, and
    idx is in range (T from the history table if not given). Returns the number of rows."""
    if T is None:
        T = _history_T(con)
    cache: dict = {}
    data = []
    for r in rows:
        if not isinstance(r, MetricRow):
            raise ContractError(f"write_metrics: expected MetricRow, got {type(r).__name__}")
        validate_row(r, _model_of(con, r.proxy_id, cache), T, H)
        data.append((run_id, r.proxy_id, r.split, r.fluid, r.metric, int(r.idx),
                     None if r.value is None else float(r.value)))
    con.executemany("INSERT INTO metrics (run_id, proxy_id, split, fluid, metric, idx, value) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    return len(data)


def read_metrics(con, run_id: str, *, split: str | None = None) -> list[MetricRow]:
    q = "SELECT proxy_id, split, fluid, metric, idx, value FROM metrics WHERE run_id = ?"
    args: tuple = (run_id,)
    if split is not None:
        q += " AND split = ?"
        args = (run_id, split)
    rows = con.execute(q + " ORDER BY proxy_id, split, metric, fluid, idx", args).fetchall()
    return [MetricRow(r[0], r[1], r[2], r[3], int(r[4]), r[5]) for r in rows]


def index_metrics(rows) -> dict:
    """{(proxy_id, split, fluid, metric, idx): value} for fast lookups."""
    if isinstance(rows, dict):
        return rows
    return {(r.proxy_id, r.split, r.fluid, r.metric, r.idx): r.value for r in rows}


def metric_value(rows, proxy_id: str, split: str, fluid: str, metric: str, idx: int = -1) -> float | None:
    """The value (None = undefined). KeyError if the row is absent. `rows` is a list or index_metrics()."""
    return index_metrics(rows)[(proxy_id, split, fluid, metric, idx)]


def read_rmse_by_date(con, run_id: str, proxy_id: str) -> np.ndarray:
    """(T, 3) validation RMSE by date of one candidate (sigma_proxy of Task 4)."""
    rows = con.execute("SELECT fluid, idx, value FROM metrics WHERE run_id = ? AND proxy_id = ? AND split = 'val' "
                       "AND metric = 'rmse_by_date'", (run_id, proxy_id)).fetchall()
    if not rows:
        raise ContractError(f"read_rmse_by_date: no rows for {proxy_id} in run {run_id}")
    T = max(r[1] for r in rows) + 1
    out = np.full((T, len(FLUIDS)), np.nan)
    for f, k, v in rows:
        out[k, FLUIDS.index(f)] = np.nan if v is None else v
    if not np.isfinite(out).all() or len(rows) != T * len(FLUIDS):
        raise ContractError(f"read_rmse_by_date: incomplete rows for {proxy_id} in run {run_id}")
    return out


def latest_ok_run(con, command: str, *, where: dict | None = None) -> str | None:
    """The latest `start` among the runs with this command, status 'ok' and outputs[key] == value."""
    rows = con.execute("SELECT run_id, outputs FROM runs WHERE command = ? AND status = 'ok' ORDER BY start DESC",
                       (command,)).fetchall()
    for run_id, outputs in rows:
        o = json.loads(outputs or "{}")
        if all(o.get(k) == v for k, v in (where or {}).items()):
            return run_id
    return None


def fluid_rows(proxy_id: str, split: str, metric: str, values, fluids=FLUIDS) -> list[MetricRow]:
    """One row for each fluid; NaN becomes None (undefined)."""
    vals = list(np.asarray(values, dtype=np.float64).ravel())
    if len(vals) != len(fluids):
        raise ContractError(f"fluid_rows: {metric} has {len(vals)} values for {len(fluids)} fluids")
    return [MetricRow(proxy_id, split, f, metric, -1, clean(v)) for f, v in zip(fluids, vals)]


def clean(v) -> float | None:
    """A float (flags 1.0 or 0.0), or None for None and NaN (stored as NULL = undefined). An
    infinity stays, so that write_metrics refuses it (V9)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else 0.0
    v = float(v)
    return None if math.isnan(v) else v
