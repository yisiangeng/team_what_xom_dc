"""Data checks C1 to C10, V9 contract assertions and the report writer.

Spec task1 sections 1.7, 2.6 and 2.8. Stop checks: C1, C2, C3, C4, C8, and
C6 under `rates.definition: auto`. C5 never stops. C7, C9 and C10 report only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LISTED = 20
CHECK_IDS = tuple(f"C{i}" for i in range(1, 11))
CHECK_TITLES = {
    "C1": "Structure",
    "C2": "Coverage of the history dates",
    "C3": "Bad cells, non-negative and non-decreasing values",
    "C4": "Field Production equals RAMP_History",
    "C5": "Cases report on the history dates",
    "C6": "Rate definition of the history",
    "C7": "Gas-oil ratio",
    "C8": "Cumulative is zero at t_0",
    "C9": "X design",
    "C10": "History against the training envelope",
}


class ContractError(ValueError):
    """A V9 assertion at a module boundary failed."""


class DataCheckError(RuntimeError):
    """A stop check failed. `results` holds the checks that ran."""

    def __init__(self, message: str, results: list[CheckResult]):
        super().__init__(message)
        self.results = list(results)


@dataclass(frozen=True)
class CheckResult:
    check_id: str        # 'C1' .. 'C10'
    status: str          # 'pass' | 'fail' | 'report'
    action: str          # e.g. 'stop', 'continue', 'select R1', 'interpolate'
    detail: dict         # JSON-serialisable


def assert_array(name: str, a: np.ndarray, shape: tuple, *, dtype=np.float64,
                 finite: bool = True, nonneg: bool = False) -> None:
    """V9: raises ContractError unless `a` is an ndarray with this shape
    (None = any size on that axis) and dtype, finite (no NaN/inf/NaT) and,
    if nonneg, >= 0. Never uses `assert` (python -O strips it)."""
    if not isinstance(a, np.ndarray):
        raise ContractError(f"{name}: expected np.ndarray, got {type(a).__name__}")
    if len(a.shape) != len(shape) or any(s is not None and s != g for s, g in zip(shape, a.shape)):
        raise ContractError(f"{name}: shape {a.shape}, expected {shape}")
    if dtype is not None and a.dtype != np.dtype(dtype):
        raise ContractError(f"{name}: dtype {a.dtype}, expected {np.dtype(dtype)}")
    if finite:
        if a.dtype.kind in "fc" and not np.isfinite(a).all():
            raise ContractError(f"{name}: {int((~np.isfinite(a)).sum())} non-finite values")
        if a.dtype.kind in "mM" and np.isnat(a).any():
            raise ContractError(f"{name}: NaT values")
    if nonneg and a.dtype.kind in "fiu" and a.size and (a < 0).any():
        raise ContractError(f"{name}: {int((a < 0).sum())} negative values (min {a.min()})")


def rel(a: float, b: float, phi: float) -> float:
    """Symmetric relative difference with floor phi; 0 when a == b (spec 2.6)."""
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), phi)


# --- JSON helpers --------------------------------------------------------------

def _iso(d) -> str:
    d = np.datetime64(d)
    if np.isnat(d):
        return "NaT"
    s = np.datetime_as_string(d, unit="ms")
    return s[:10] if s.endswith("T00:00:00.000") else s


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _f(obj)
    if isinstance(obj, (np.datetime64, pd.Timestamp)):
        return _iso(obj)
    return obj


def _result(check_id, status, action, detail) -> CheckResult:
    detail = _jsonable(detail)
    json.dumps(detail)  # must be serialisable
    return CheckResult(check_id, status, action, detail)


def _stop_result(check_id, failures, detail) -> CheckResult:
    status = "fail" if failures else "pass"
    action = "stop" if failures else "continue"
    return _result(check_id, status, action, detail)


# --- C1 ------------------------------------------------------------------------

def check_c1(params: pd.DataFrame, problems01: list[str], raw, cfg: dict) -> CheckResult:
    """Structure of files 01 and 02 (spec 2.6, C1 parts a to f)."""
    from dwc26.data.align import history_dates
    from dwc26.data.load import HISTORY_ID, QUANTITIES

    f02 = cfg["file02"]
    fails: list[dict] = []

    def fail(part, sheet, name, problem):
        fails.append({"part": part, "sheet": sheet, "name": name, "problem": problem})

    # (a) file 01
    for p in problems01:
        fail("a", cfg["file01"]["sheet"], "", p)
    ids = set(int(i) for i in params["case_id"]) if len(params) else set()
    if not ids:
        fail("a", cfg["file01"]["sheet"], "", "no case in file 01")

    case_rx = re.compile(f02["case_pattern"])
    hist = f02["history_column"]
    date_cols = {}
    hist_dates = {}
    for q in QUANTITIES:
        sheet = f02["sheets"][q]["name"]
        if sheet not in raw.sheets_found:
            fail("b", sheet, "", "sheet missing")
            continue
        if q not in raw.date_col:
            first = raw.names.get(q, [""])[0] if raw.names.get(q) else ""
            fail("b", sheet, first, f"first row-{f02['names_row']} name is not '{f02['date_column']}'")
            continue
        names = raw.names[q]
        # (b) names: Date, then exactly the file-01 cases and the history, no duplicate, nothing else
        seen_names, seen_ids, n_hist = set(), set(), 0
        for nm in names:
            if nm in seen_names:
                fail("b", sheet, nm, "duplicate name")
            seen_names.add(nm)
            m = case_rx.fullmatch(nm)
            if m:
                cid = int(m.group(1))
                if cid == HISTORY_ID:
                    # its cells would merge into the RAMP_History series in the long tables
                    fail("b", sheet, nm, f"case number {cid} collides with the id {HISTORY_ID} "
                                         f"of '{hist}' in the long tables")
                if cid in seen_ids:
                    fail("b", sheet, nm, f"duplicate case {cid}")
                seen_ids.add(cid)
                if cid not in ids:
                    fail("b", sheet, nm, "case not in file 01")
            elif nm == hist:
                n_hist += 1
            else:
                fail("b", sheet, nm, "unexpected name")
        for cid in sorted(ids - seen_ids):
            fail("b", sheet, f"case {cid}", "case of file 01 missing")
        if n_hist == 0:
            fail("b", sheet, hist, "history column missing")
        # (c) units
        want = f02["sheets"][q]["unit"]
        units = raw.units.get(q, [])
        nonempty = [(nm, u) for nm, u in zip(names, units) if u != ""]
        if not nonempty:
            fail("c", sheet, "", "no unit in the units row")
        for nm, u in nonempty:
            if u != want:
                fail("c", sheet, nm, f"unit '{u}', expected '{want}'")
        # (d) Date column
        d = raw.date_col[q]
        if np.isnat(d).any():
            fail("d", sheet, f02["date_column"], f"{int(np.isnat(d).sum())} unparseable dates")
        dd = d[~np.isnat(d)]
        if len(np.unique(dd)) != len(dd):
            fail("d", sheet, f02["date_column"], f"{len(dd) - len(np.unique(dd))} duplicate dates")
        date_cols[sheet] = d
        hist_dates[sheet] = history_dates(raw, q)

    # (d) identical Date columns
    if date_cols:
        ref_sheet, ref = next(iter(date_cols.items()))
        for sheet, d in date_cols.items():
            if d.shape != ref.shape or not np.array_equal(d, ref):
                fail("d", sheet, f02["date_column"], f"Date column differs from '{ref_sheet}'")
    # (e) history dates
    if hist_dates:
        ref_sheet, ref = next(iter(hist_dates.items()))
        for sheet, d in hist_dates.items():
            if d.shape != ref.shape or not np.array_equal(d, ref):
                fail("e", sheet, hist, f"history dates differ from '{ref_sheet}'")
            if len(d) < 2:
                fail("e", sheet, hist, f"{len(d)} history dates (need >= 2)")
            ok = ~np.isnat(d)
            dd = d[ok]
            off = dd[dd != dd.astype("datetime64[D]").astype(dd.dtype)]
            for t in off[:MAX_LISTED]:
                fail("e", sheet, hist, f"history date {_iso(t)} not at 00:00:00.000")
    # (f) Field Production
    fsheet = f02["field_sheet"]["name"]
    if raw.field is None:
        fail("f", fsheet, "", "sheet missing")
    else:
        for head, internal in f02["field_sheet"]["columns"].items():
            if internal not in raw.field.columns:
                fail("f", fsheet, head, "column missing")

    n_dates = len(next(iter(date_cols.values()))) if date_cols else 0
    T = len(next(iter(hist_dates.values()))) if hist_dates else 0
    detail = {
        "n_failures": len(fails),
        "failures": fails,
        "n_cases_file01": len(ids),
        "n_sheets_checked": len(date_cols),
        "n_rows_date_column": n_dates,
        "n_history_dates": T,
    }
    return _stop_result("C1", fails, detail)


# --- C2 ------------------------------------------------------------------------

def check_c2(raw, grid, case_ids) -> CheckResult:
    """Each case covers the first to the last history date in each sheet."""
    from dwc26.data.load import QUANTITIES, native_series

    series = native_series(raw)
    fails = []
    for cid in case_ids:
        for q in QUANTITIES:
            if (int(cid), q) not in series:
                fails.append({"case_id": int(cid), "quantity": q, "first": None, "last": None})
                continue
            d = series[(int(cid), q)][0]
            if d.min() > grid[0] or d.max() < grid[-1]:
                fails.append({"case_id": int(cid), "quantity": q, "first": _iso(d.min()), "last": _iso(d.max())})
    detail = {"t_first": _iso(grid[0]), "t_last": _iso(grid[-1]), "n_failing": len(fails),
              "failing": fails[:MAX_LISTED]}
    return _stop_result("C2", fails, detail)


# --- C3 ------------------------------------------------------------------------

def check_c3(raw) -> CheckResult:
    """No bad cells; cumulatives >= 0 and non-decreasing (exact); rates >= 0."""
    from dwc26.data.load import native_series

    viol = []
    for _, b in raw.bad_cells.iterrows():
        viol.append({"kind": "bad cell", "case_id": int(b.case_id), "quantity": b.quantity,
                     "date": _iso(b.date), "value": str(b.raw), "previous": None})
    n_bad = len(viol)
    for (cid, q), (d, v) in sorted(native_series(raw).items()):
        for j in np.flatnonzero(v < 0):
            viol.append({"kind": "negative", "case_id": cid, "quantity": q, "date": _iso(d[j]),
                         "value": float(v[j]), "previous": float(v[j - 1]) if j > 0 else None})
        if q.endswith("_cum") and len(v) > 1:
            for j in np.flatnonzero(np.diff(v) < 0) + 1:
                viol.append({"kind": "decrease", "case_id": cid, "quantity": q, "date": _iso(d[j]),
                             "value": float(v[j]), "previous": float(v[j - 1])})
    detail = {"n_violations": len(viol), "n_bad_cells": n_bad, "violations": viol[:MAX_LISTED]}
    return _stop_result("C3", viol, detail)


# --- C4 ------------------------------------------------------------------------

def check_c4(raw, grid, cfg) -> CheckResult:
    """Field Production dates = grid; its rates equal the RAMP_History rate sheets."""
    from dwc26.data.align import values_on_grid
    from dwc26.data.load import FLUIDS, HISTORY_ID, native_series

    tol = float(cfg["checks"]["c4_rel_tol"])
    fails, max_rel = [], {}
    field = raw.field
    detail = {"tol": tol}
    if field is None or "date" not in field.columns:
        fails.append({"problem": "Field Production sheet or its Date column missing"})
        detail.update(n_failing=len(fails), failing=fails)
        return _stop_result("C4", fails, detail)
    fd = field["date"].to_numpy().astype(grid.dtype)
    if fd.shape != grid.shape or not np.array_equal(fd, grid):
        fails.append({"problem": "Field Production dates differ from the history grid",
                      "n_field": len(fd), "n_grid": len(grid),
                      "field_first": _iso(fd[0]) if len(fd) else None,
                      "field_last": _iso(fd[-1]) if len(fd) else None})
        detail.update(n_failing=len(fails), failing=fails)
        return _stop_result("C4", fails, detail)
    series = native_series(raw)
    for fluid in FLUIDS:
        col = f"{fluid}_rate"
        d, v = series[(HISTORY_ID, col)]
        r = values_on_grid(d, v, grid, f"RAMP_History {col}")
        fv = field[col].to_numpy(dtype=np.float64)
        rels = np.array([rel(a, b, 0.0) if np.isfinite(a) else np.inf for a, b in zip(fv, r)])
        max_rel[fluid] = float(rels.max()) if len(rels) else 0.0
        for k in np.flatnonzero(rels > tol):
            fails.append({"fluid": fluid, "date": _iso(grid[k]), "field": _f(fv[k]), "history": float(r[k]),
                          "rel": _f(rels[k])})
    detail.update(max_rel=max_rel, n_failing=len(fails), failing=fails[:MAX_LISTED])
    return _stop_result("C4", fails, detail)


# --- C8 ------------------------------------------------------------------------

def check_c8(raw, grid, case_ids, cfg) -> CheckResult:
    """|C(t_0)| <= c8_abs_tol for each series and cumulative sheet (D-005)."""
    from dwc26.data.align import interp_cum, to_days
    from dwc26.data.load import FLUIDS, HISTORY_ID, native_series

    tol = float(cfg["checks"]["c8_abs_tol"])
    series = native_series(raw)
    t0 = grid[0]
    fails, max_abs = [], {f: 0.0 for f in FLUIDS}
    for cid in [*map(int, case_ids), HISTORY_ID]:
        for fluid in FLUIDS:
            d, v = series[(cid, f"{fluid}_cum")]
            c0 = float(interp_cum(to_days(d, t0), v, np.zeros(1))[0])
            max_abs[fluid] = max(max_abs[fluid], abs(c0))
            if abs(c0) > tol:
                fails.append({"case_id": cid, "fluid": fluid, "value": c0})
    detail = {"tol": tol, "t_0": _iso(t0), "max_abs": max_abs, "n_failing": len(fails),
              "failing": fails[:MAX_LISTED]}
    return _stop_result("C8", fails, detail)


# --- C5 ------------------------------------------------------------------------

def check_c5(raw, grid, case_ids) -> CheckResult:
    """Every history date is a native date of each case and sheet (exact)."""
    from dwc26.data.load import QUANTITIES, native_series

    series = native_series(raw)
    off = []
    for cid in case_ids:
        for q in QUANTITIES:
            d = series.get((int(cid), q), (np.array([], dtype=grid.dtype), None))[0]
            for k in np.flatnonzero(~np.isin(grid, d)):
                off.append({"case_id": int(cid), "quantity": q, "date": _iso(grid[k])})
    if off:
        return _result("C5", "fail", "interpolate (np.interp)",
                       {"n_off_native": len(off), "off_native": off[:MAX_LISTED]})
    return _result("C5", "pass", "use values on the dates", {"n_off_native": 0, "off_native": []})


# --- C6 ------------------------------------------------------------------------

def check_c6(raw, grid, cfg) -> tuple[CheckResult, str | None]:
    """Reported history rate against the previous- and next-quarter means of
    the history cumulatives. Returns (result, 'R1' | 'R2' | None = stop)."""
    from dwc26.data.align import to_days, values_on_grid
    from dwc26.data.load import FLUIDS, HISTORY_ID, native_series

    tol = float(cfg["checks"]["c6_rel_tol"])
    floor_frac = float(cfg["checks"]["c6_floor_frac"])
    mode = cfg["rates"]["definition"]
    series = native_series(raw)
    t = to_days(grid, grid[0])
    T = len(grid)
    per_fluid, phis = {}, {}
    best = {"next": (-1.0, None, None), "prev": (-1.0, None, None)}
    for fluid in FLUIDS:
        d, v = series[(HISTORY_ID, f"{fluid}_cum")]
        C = values_on_grid(d, v, grid, f"RAMP_History {fluid}_cum")
        d, v = series[(HISTORY_ID, f"{fluid}_rate")]
        r = values_on_grid(d, v, grid, f"RAMP_History {fluid}_rate")
        m = np.diff(C) / np.diff(t)          # mean over (t_k, t_{k+1}]
        phi = floor_frac * float(np.max(np.abs(r)))
        phis[fluid] = phi
        e_next = [rel(r[k], m[k], phi) for k in range(T - 1)]        # k = 0..T-2
        e_prev = [rel(r[k], m[k - 1], phi) for k in range(1, T)]     # k = 1..T-1
        kn = int(np.argmax(e_next))
        kp = int(np.argmax(e_prev)) + 1
        per_fluid[fluid] = {"e_next": e_next[kn], "k_next": kn, "date_next": _iso(grid[kn]),
                            "e_prev": e_prev[kp - 1], "k_prev": kp, "date_prev": _iso(grid[kp])}
        if e_next[kn] > best["next"][0]:
            best["next"] = (e_next[kn], fluid, kn)
        if e_prev[kp - 1] > best["prev"][0]:
            best["prev"] = (e_prev[kp - 1], fluid, kp)
    e_next, e_prev = best["next"][0], best["prev"][0]
    passed = min(e_next, e_prev) < tol
    convention = "next" if e_next <= e_prev else "prev"

    note = None
    if mode == "auto":
        definition = "R1" if passed else None
        action = "select R1" if passed else "stop"
    elif mode == "R1":
        definition = "R1"
        action = "select R1 (forced)"
        if not passed:
            note = "V4 not verified by C6"
    else:
        definition = "R2"
        action = "select R2 (forced)"
        note = ("V4 not verified: case rates are simulator report rates, "
                "history rates are quarter means (C6)")
    detail = {
        "e_next": e_next, "e_prev": e_prev,
        "argmax_next": {"fluid": best["next"][1], "k": best["next"][2],
                        "date": _iso(grid[best["next"][2]])},
        "argmax_prev": {"fluid": best["prev"][1], "k": best["prev"][2],
                        "date": _iso(grid[best["prev"][2]])},
        "by_fluid": per_fluid, "phi": phis, "tol": tol,
        "convention": convention, "rates_definition_config": mode,
        "definition": definition, "v4_note": note,
    }
    return _result("C6", "pass" if passed else "fail", action, detail), definition


# --- C7 ------------------------------------------------------------------------

def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def check_c7(aligned) -> CheckResult:
    """GOR = q_gas / q_oil (MSCF/STB) where q_oil > 0, and its OLS trend per year."""
    rate, ids, dates = aligned.rate, aligned.case_ids, aligned.dates
    years = aligned.t_days / 365.25
    oil, gas = rate[..., 0], rate[..., 1]
    valid = oil > 0
    gor = np.full(oil.shape, np.nan)
    gor[valid] = gas[valid] / oil[valid]
    detail = {"unit": "MSCF/STB", "n_valid": int(valid.sum()), "n_excluded_oil_zero": int((~valid).sum())}
    if valid.any():
        i_min, k_min = np.unravel_index(np.nanargmin(gor), gor.shape)
        i_max, k_max = np.unravel_index(np.nanargmax(gor), gor.shape)
        detail["gor_min"] = {"value": gor[i_min, k_min], "case_id": ids[i_min], "date": dates[k_min]}
        detail["gor_max"] = {"value": gor[i_max, k_max], "case_id": ids[i_max], "date": dates[k_max]}
        detail["gor_median"] = float(np.median(gor[valid]))
    else:
        detail["gor_min"] = {"value": None, "case_id": None, "date": None}
        detail["gor_max"] = {"value": None, "case_id": None, "date": None}
        detail["gor_median"] = None
    per_date = []
    for k in range(len(dates)):
        col = gor[:, k][valid[:, k]]
        per_date.append({"date": dates[k], "n": len(col),
                         "median": float(np.median(col)) if len(col) else None,
                         "min": float(col.min()) if len(col) else None,
                         "max": float(col.max()) if len(col) else None})
    detail["per_date"] = per_date
    slopes = []
    for i in range(len(ids)):
        m = valid[i]
        if m.sum() >= 3:
            slopes.append(_ols_slope(years[m], gor[i, m]))
    slopes = np.array(slopes)
    detail["slope_unit"] = "MSCF/STB per year"
    detail["n_cases_slope"] = len(slopes)
    detail["n_cases_slope_positive"] = int((slopes > 0).sum())
    detail["median_slope"] = float(np.median(slopes)) if len(slopes) else None
    ho, hg = aligned.hist_rate[:, 0], aligned.hist_rate[:, 1]
    hv = ho > 0
    hgor = hg[hv] / ho[hv]
    detail["history"] = {
        "n_valid": int(hv.sum()),
        "gor_min": float(hgor.min()) if hv.any() else None,
        "gor_max": float(hgor.max()) if hv.any() else None,
        "slope": _ols_slope(years[hv], hgor) if hv.sum() >= 3 else None,
    }
    return _result("C7", "report", "continue", detail)


# --- C9 ------------------------------------------------------------------------

def check_c9(params: pd.DataFrame, xs) -> CheckResult:
    """X design: bounds, end gaps, quartiles, KS against U(0,1) and log-uniform, Spearman."""
    from scipy import stats

    X = params[list(xs.names)].to_numpy(dtype=np.float64)
    U = xs.transform(X)
    out = {}
    for j, name in enumerate(xs.names):
        lo, hi = float(xs.lower[j]), float(xs.upper[j])
        x = X[:, j]
        width = hi - lo
        ks = stats.kstest(U[:, j], "uniform")
        p = {
            "lower": lo, "upper": hi, "min": float(x.min()), "max": float(x.max()),
            "gap_lower": (float(x.min()) - lo) / width, "gap_upper": (hi - float(x.max())) / width,
            "quartiles": np.quantile(x, [0.25, 0.5, 0.75]).tolist(),
            "quartiles_scaled": np.quantile(U[:, j], [0.25, 0.5, 0.75]).tolist(),
            "ks_uniform": {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)},
            "ks_loguniform": None,
        }
        if lo > 0:
            v = (np.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
            ksl = stats.kstest(v, "uniform")
            p["ks_loguniform"] = {"statistic": float(ksl.statistic), "pvalue": float(ksl.pvalue)}
        out[name] = p
    rho = stats.spearmanr(X).statistic
    rho = np.atleast_2d(rho)
    detail = {"n": len(X), "bounds_source": xs.source, "parameters": out,
              "spearman": {"names": list(xs.names), "matrix": rho}}
    return _result("C9", "report", "continue", detail)


# --- C10 -----------------------------------------------------------------------

def check_c10(aligned, split: np.ndarray, ys, cfg: dict) -> CheckResult:
    """History against the min-max envelope of the training cases (V1: train only).

    Amendment A1 (D-009): the excess at k is e = rel(hist, U; 0) above U,
    rel(hist, L; 0) below L, else 0. A date counts as outside only if
    e > checks.c10_rel_tol; `n_outside_exact` counts without the tolerance."""
    tol = float(cfg["checks"]["c10_rel_tol"])
    rate, hist, dates = aligned.rate, aligned.hist_rate, aligned.dates
    train = split == "train"
    L = rate[train].min(axis=0)
    U = rate[train].max(axis=0)
    fluids = {}
    for f, fluid in enumerate(ys.fluids):
        s = float(ys.factor[f])
        below_exact = hist[:, f] < L[:, f]
        above_exact = hist[:, f] > U[:, f]
        e = np.zeros(len(dates))
        for k in np.flatnonzero(above_exact):
            e[k] = rel(float(hist[k, f]), float(U[k, f]), 0.0)
        for k in np.flatnonzero(below_exact):
            e[k] = rel(float(hist[k, f]), float(L[k, f]), 0.0)
        beyond = e > tol
        below, above = below_exact & beyond, above_exact & beyond
        outside = []
        for k in np.flatnonzero(below | above):
            outside.append({"k": int(k), "date": dates[k], "hist": hist[k, f], "lower": L[k, f],
                            "upper": U[k, f], "side": "below" if below[k] else "above", "rel_excess": e[k]})
        val, test = split == "val", split == "test"
        fluids[fluid] = {
            "n_outside": len(outside), "n_below": int(below.sum()), "n_above": int(above.sum()),
            "outside": outside,
            "n_outside_exact": int((below_exact | above_exact).sum()),
            "max_rel_excess": float(e.max()) if len(e) else 0.0,
            "hist_max_scaled": float(hist[:, f].max()) / s,
            "train_max_scaled": float(rate[train][..., f].max()) / s,
            "val_max_scaled": float(rate[val][..., f].max()) / s if val.any() else None,
            "test_max_scaled": float(rate[test][..., f].max()) / s if test.any() else None,
        }
    zero = np.flatnonzero((hist == 0).all(axis=1))
    zh = {"dates": [dates[k] for k in zero],
          "max_case_rate": {fluid: [float(rate[:, k, f].max()) for k in zero]
                            for f, fluid in enumerate(ys.fluids)}}
    detail = {"tol": tol, "n_train": int(train.sum()), "fluids": fluids, "zero_history": zh}
    return _result("C10", "report", "continue", detail)


# --- report --------------------------------------------------------------------

def _fmt(x, nd=4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def summary(r: CheckResult) -> str:
    d = r.detail
    c = r.check_id
    try:
        if c == "C1":
            return (f"{d['n_failures']} failures; {d['n_cases_file01']} cases, {d['n_sheets_checked']} sheets, "
                    f"{d['n_rows_date_column']} Date rows, {d['n_history_dates']} history dates")
        if c == "C2":
            return f"{d['n_failing']} (case, sheet) pairs not covering {d['t_first']} to {d['t_last']}"
        if c == "C3":
            return f"{d['n_violations']} violations ({d['n_bad_cells']} bad cells)"
        if c == "C4":
            if "max_rel" in d:
                mr = ", ".join(f"{k} {_fmt(v, 3)}" for k, v in d["max_rel"].items())
                return f"max rel: {mr} (tol {_fmt(d['tol'])}); {d['n_failing']} failing"
            return f"{d['n_failing']} failing: {d['failing'][0].get('problem')}"
        if c == "C5":
            return f"{d['n_off_native']} (case, sheet, date) cells off the native dates"
        if c == "C6":
            return (f"e_next {_fmt(d['e_next'], 3)}, e_prev {_fmt(d['e_prev'], 3)}; convention "
                    f"{d['convention']}; definition {d['definition'] or 'none (stop)'}")
        if c == "C7":
            return (f"GOR {_fmt(d['gor_min']['value'])} to {_fmt(d['gor_max']['value'])} MSCF/STB, median "
                    f"{_fmt(d['gor_median'])}; slope > 0 in {d['n_cases_slope_positive']}/{d['n_cases_slope']} cases")
        if c == "C8":
            return f"{d['n_failing']} series with |C(t_0)| > {_fmt(d['tol'])}"
        if c == "C9":
            gaps = max(max(p["gap_lower"], p["gap_upper"]) for p in d["parameters"].values())
            pmin = min(p["ks_uniform"]["pvalue"] for p in d["parameters"].values())
            return f"n = {d['n']}; max end gap {_fmt(gaps, 3)}; min KS-uniform p {_fmt(pmin, 3)}"
        if c == "C10":
            parts = [f"{k} {v['n_outside']}" for k, v in d["fluids"].items()]
            exact = [f"{k} {v['n_outside_exact']}" for k, v in d["fluids"].items()]
            return (f"history dates outside the training envelope (rel tol {_fmt(d['tol'])}): "
                    + ", ".join(parts) + "; without tolerance: " + ", ".join(exact))
    except (KeyError, TypeError, ValueError, IndexError):
        pass
    return ""


def _cell(x) -> str:
    """A Markdown table cell: '|' escaped so it does not split the cell."""
    return _fmt(x).replace("|", r"\|")


def _table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return out


def _section(r: CheckResult) -> list[str]:
    d = r.detail
    c = r.check_id
    lines = [f"## {c}: {CHECK_TITLES[c]}", "", f"Status: **{r.status}**. Action: {r.action}.", ""]
    if c == "C6":
        lines += _table(["Fluid", "e_next", "k (date)", "e_prev", "k (date)", "phi"],
                        [[f, v["e_next"], f"{v['k_next']} ({v['date_next']})", v["e_prev"],
                          f"{v['k_prev']} ({v['date_prev']})", d["phi"][f]] for f, v in d["by_fluid"].items()])
        lines += ["", f"Convention: `{d['convention']}`. Definition selected: `{d['definition']}`.", ""]
    elif c == "C7":
        lines += [f"Global GOR min {_fmt(d['gor_min']['value'])} (case {d['gor_min']['case_id']}, "
                  f"{d['gor_min']['date']}); max {_fmt(d['gor_max']['value'])} (case {d['gor_max']['case_id']}, "
                  f"{d['gor_max']['date']}); median {_fmt(d['gor_median'])} MSCF/STB. "
                  f"Excluded (q_oil = 0): {d['n_excluded_oil_zero']}.",
                  f"Cases with slope > 0: {d['n_cases_slope_positive']} of {d['n_cases_slope']}; "
                  f"median slope {_fmt(d['median_slope'])} MSCF/STB per year.",
                  f"History: GOR min {_fmt(d['history']['gor_min'])}, max {_fmt(d['history']['gor_max'])}, "
                  f"slope {_fmt(d['history']['slope'])} MSCF/STB per year.", ""]
        lines += _table(["Date", "n", "median", "min", "max"],
                        [[p["date"], p["n"], p["median"], p["min"], p["max"]] for p in d["per_date"]])
        lines.append("")
    elif c == "C9":
        lines += [f"Bounds: {d['bounds_source']}.", ""]
        lines += _table(["Parameter", "lower", "upper", "min", "max", "gap low", "gap high", "Q1", "Q2", "Q3",
                         "KS U(0,1) D", "p", "KS log-U D", "p"],
                        [[n, p["lower"], p["upper"], p["min"], p["max"], p["gap_lower"], p["gap_upper"],
                          *p["quartiles"], p["ks_uniform"]["statistic"], p["ks_uniform"]["pvalue"],
                          (p["ks_loguniform"] or {}).get("statistic"), (p["ks_loguniform"] or {}).get("pvalue")]
                         for n, p in d["parameters"].items()])
        lines += ["", "Spearman rank correlation:", ""]
        names = d["spearman"]["names"]
        lines += _table(["", *names], [[n, *row] for n, row in zip(names, d["spearman"]["matrix"])])
        lines.append("")
    elif c == "C10":
        lines += [f"A date counts as outside if its relative excess exceeds c10_rel_tol = {_fmt(d['tol'])} "
                  "(amendment A1, D-009). 'outside (exact)' counts without the tolerance.", ""]
        lines += _table(["Fluid", "outside", "below", "above", "outside (exact)", "max rel excess",
                         "max hist / s", "max val / s", "max test / s"],
                        [[f, v["n_outside"], v["n_below"], v["n_above"], v["n_outside_exact"], v["max_rel_excess"],
                          v["hist_max_scaled"], v["val_max_scaled"], v["test_max_scaled"]]
                         for f, v in d["fluids"].items()])
        lines.append("")
        for f, v in d["fluids"].items():
            if v["outside"]:
                lines += [f"{f}: " + ", ".join(f"k={o['k']} {o['date']} ({o['side']}, rel {_fmt(o['rel_excess'], 3)})"
                                               for o in v["outside"]), ""]
        z = d["zero_history"]
        lines += [f"Dates with all three history rates = 0: {', '.join(z['dates']) or 'none'}.", ""]
        if z["dates"]:
            lines += _table(["Date", *z["max_case_rate"]],
                            [[dt, *[z["max_case_rate"][f][j] for f in z["max_case_rate"]]]
                             for j, dt in enumerate(z["dates"])])
            lines.append("")
    lines += ["<details><summary>Detail (JSON)</summary>", "", "```json",
              json.dumps(d, indent=1, sort_keys=True), "```", "", "</details>", ""]
    return lines


def write_report(path: Path, run_meta: dict, results: list[CheckResult]) -> None:
    """Markdown report: header, summary table C1..C10 ('not run' if absent), one section per check."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {r.check_id: r for r in results}
    m = run_meta
    failed = m.get("status") != "ok"
    commit = m.get("git_commit", "n/a")
    if m.get("git_dirty"):
        commit = f"{commit}+dirty"
    lines = ["# Task 1 data checks", ""]
    lines.append(f"**Status: {'FAILED' if failed else 'OK'}**"
                 + (f" (stopped at {m['stopped_at']})" if failed and m.get("stopped_at") else ""))
    lines.append("")
    header = [
        ("Run ID", m.get("run_id")),
        ("Git commit", commit),
        ("Start (UTC)", m.get("start")),
        ("Config", f"{m.get('config_path')} (sha256 {m.get('config_sha256')})"),
    ]
    for p, h in (m.get("inputs") or {}).items():
        header.append((f"Input {p}", f"sha256 {h}"))
    header += [
        ("n cases", m.get("n")),
        ("T history dates", m.get("T")),
        ("First date", m.get("first_date")),
        ("Last date", m.get("last_date")),
        ("Rate definition", f"{m.get('rate_definition')} (C6 convention: {m.get('c6_convention')})"),
    ]
    if m.get("v4_note"):
        header.append(("V4", m["v4_note"]))
    lines += _table(["Item", "Value"], header)
    lines += ["", "## Summary", ""]
    rows = []
    for c in CHECK_IDS:
        r = by_id.get(c)
        rows.append([c, "not run", "not run", ""] if r is None else [c, r.status, r.action, summary(r)])
    lines += _table(["Check", "Status", "Action", "Summary"], rows)
    lines.append("")
    for c in CHECK_IDS:
        if c in by_id:
            lines += _section(by_id[c])
        else:
            lines += [f"## {c}: {CHECK_TITLES[c]}", "", "not run", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
