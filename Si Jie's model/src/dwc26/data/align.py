"""Time, history grid, cumulative interpolation and rate definitions R1, R2.

Spec task1 sections 2.2 to 2.5. Cases and RAMP_History go through the same
functions (V4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dwc26.data.check import ContractError, assert_array
from dwc26.data.load import FLUIDS, HISTORY_ID, RawProduction, native_series


def to_days(ts: np.ndarray, t0: np.datetime64) -> np.ndarray:
    """(m,) datetime64 -> (m,) float64 days from t0: integer units of the
    timestamps (ms by default) divided by the units per day (spec 2.2)."""
    ts = np.asarray(ts)
    if ts.dtype.kind != "M":
        raise ContractError(f"to_days: expected datetime64, got {ts.dtype}")
    if np.isnat(ts).any() or np.isnat(np.datetime64(t0)):
        raise ContractError("to_days: NaT in the timestamps")
    delta = ts - np.datetime64(t0)
    unit = np.datetime_data(delta.dtype)[0]
    per_day = np.timedelta64(1, "D") // np.timedelta64(1, unit)
    return delta.astype(np.int64) / float(per_day)


def _is_midnight(d: np.ndarray) -> np.ndarray:
    return d == d.astype("datetime64[D]").astype(d.dtype)


def history_dates(raw: RawProduction, quantity: str) -> np.ndarray:
    """Dates of the non-empty RAMP_History cells of one sheet (valid or bad)."""
    nat = raw.native
    d = nat.loc[(nat.case_id == HISTORY_ID) & (nat.quantity == quantity), "date"].to_numpy()
    bad = raw.bad_cells
    if len(bad):
        b = bad.loc[(bad.case_id == HISTORY_ID) & (bad.quantity == quantity), "date"].to_numpy()
        d = np.concatenate([d, b.astype(d.dtype)])
    return np.unique(d)


def history_grid(raw: RawProduction) -> np.ndarray:
    """(T,) sorted dates of the non-empty RAMP_History cells. Call only after C1 passes."""
    parts = [history_dates(raw, q) for q in raw.date_col]
    if not parts:
        raise ContractError("history_grid: no data sheet with a Date column")
    grid = np.unique(np.concatenate(parts))
    if grid.ndim != 1 or len(grid) < 2:
        raise ContractError(f"history_grid: need at least 2 history dates, got {len(grid)}")
    if np.isnat(grid).any():
        raise ContractError("history_grid: NaT history date")
    if not (np.diff(grid) > np.timedelta64(0)).all():
        raise ContractError("history_grid: dates not strictly increasing")
    if not _is_midnight(grid).all():
        raise ContractError("history_grid: a history date is not at 00:00:00.000")
    return grid


def interp_cum(t_native: np.ndarray, cum_native: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of a cumulative onto grid times (np.interp).
    No extrapolation: raises ContractError if a grid time lies outside the knots."""
    tau = np.asarray(t_native, dtype=np.float64)
    c = np.asarray(cum_native, dtype=np.float64)
    tg = np.asarray(t_grid, dtype=np.float64)
    if tau.ndim != 1 or c.shape != tau.shape or tg.ndim != 1 or len(tau) == 0:
        raise ContractError(f"interp_cum: bad shapes {tau.shape}, {c.shape}, {tg.shape}")
    if not (np.isfinite(tau).all() and np.isfinite(c).all() and np.isfinite(tg).all()):
        raise ContractError("interp_cum: non-finite input")
    if len(tau) > 1 and not (np.diff(tau) > 0).all():
        raise ContractError("interp_cum: native times not strictly increasing")
    if (tg < tau[0]).any() or (tg > tau[-1]).any():
        raise ContractError(
            f"interp_cum: grid times outside the native range [{tau[0]}, {tau[-1]}] (no extrapolation)")
    out = np.interp(tg, tau, c)
    assert_array("interp_cum", out, tg.shape, finite=True)
    return out


def rates_r1(t_grid: np.ndarray, cum: np.ndarray, c8_abs_tol: float = 0.0) -> np.ndarray:
    """R1: mean rate over the previous quarter, from cumulatives. Time on axis -2.
    q(t_0) = 0 (D-005); requires |C(t_0)| <= c8_abs_tol."""
    t = np.asarray(t_grid, dtype=np.float64)
    c = np.asarray(cum, dtype=np.float64)
    if t.ndim != 1 or c.ndim < 2 or c.shape[-2] != len(t) or len(t) < 2:
        raise ContractError(f"rates_r1: bad shapes t {t.shape}, cum {c.shape}")
    dt = np.diff(t)
    if not (np.isfinite(dt).all() and (dt > 0).all()):
        raise ContractError("rates_r1: grid times not strictly increasing")
    if not np.isfinite(c).all():
        raise ContractError("rates_r1: non-finite cumulative")
    if (np.abs(c[..., 0, :]) > c8_abs_tol).any():
        raise ContractError(f"rates_r1: |C(t_0)| > {c8_abs_tol} (R1 needs C(t_0) = 0, check C8)")
    q = np.zeros_like(c)
    q[..., 1:, :] = np.diff(c, axis=-2) / dt[:, None]
    assert_array("rates_r1", q, c.shape, finite=True, nonneg=True)
    return q


def rates_r2(t_native: np.ndarray, rate_native: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """R2: last reported rate at or before each grid time."""
    tau = np.asarray(t_native, dtype=np.float64)
    r = np.asarray(rate_native, dtype=np.float64)
    tg = np.asarray(t_grid, dtype=np.float64)
    if tau.ndim != 1 or r.shape != tau.shape or tg.ndim != 1:
        raise ContractError(f"rates_r2: bad shapes {tau.shape}, {r.shape}, {tg.shape}")
    if len(tau) > 1 and not (np.diff(tau) > 0).all():
        raise ContractError("rates_r2: native times not strictly increasing")
    idx = np.searchsorted(tau, tg, side="right") - 1
    if (idx < 0).any():
        raise ContractError("rates_r2: a grid time precedes the first report")
    out = r[idx]
    assert_array("rates_r2", out, tg.shape, finite=True, nonneg=True)
    return out


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
    n_interpolated: int             # (case, cumulative sheet, grid date) cells not on a native date


def _series(series: dict, cid: int, q: str):
    if (cid, q) not in series:
        raise ContractError(f"align: no native data for case {cid}, {q}")
    return series[(cid, q)]


def values_on_grid(dates: np.ndarray, values: np.ndarray, grid: np.ndarray, what: str) -> np.ndarray:
    """Exact-date lookup of a native series on the grid (no interpolation)."""
    pos = np.searchsorted(dates, grid)
    ok = (pos < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == grid)
    if not ok.all():
        raise ContractError(f"{what}: missing on grid dates {grid[~ok].astype(str).tolist()}")
    return values[pos]


def align_all(raw: RawProduction, grid: np.ndarray, case_ids: np.ndarray,
              definition: str, cfg: dict) -> Aligned:
    """Aligns every case and RAMP_History to the grid with one rate definition.
    The history is processed as one more row of the same arrays (V4)."""
    if definition not in ("R1", "R2"):
        raise ContractError(f"align_all: unknown rate definition {definition!r}")
    ids = np.asarray(case_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or not (np.diff(ids) > 0).all():
        raise ContractError("align_all: case_ids must be non-empty, unique and ascending")
    grid = np.asarray(grid)
    T = len(grid)
    t0 = grid[0]
    t_grid = to_days(grid, t0)
    series = native_series(raw)
    all_ids = [int(i) for i in ids] + [HISTORY_ID]

    cum = np.empty((len(all_ids), T, len(FLUIDS)), dtype=np.float64)
    n_interp = 0
    for a, cid in enumerate(all_ids):
        for f, fluid in enumerate(FLUIDS):
            d, v = _series(series, cid, f"{fluid}_cum")
            cum[a, :, f] = interp_cum(to_days(d, t0), v, t_grid)
            if cid != HISTORY_ID:
                n_interp += int((~np.isin(grid, d)).sum())

    if definition == "R1":
        rate = rates_r1(t_grid, cum, float(cfg["checks"]["c8_abs_tol"]))
    else:
        rate = np.empty_like(cum)
        for a, cid in enumerate(all_ids):
            for f, fluid in enumerate(FLUIDS):
                d, v = _series(series, cid, f"{fluid}_rate")
                rate[a, :, f] = rates_r2(to_days(d, t0), v, t_grid)

    reported = np.empty((T, len(FLUIDS)), dtype=np.float64)
    for f, fluid in enumerate(FLUIDS):
        d, v = _series(series, HISTORY_ID, f"{fluid}_rate")
        reported[:, f] = values_on_grid(d, v, grid, f"RAMP_History {fluid}_rate")

    out = Aligned(
        dates=grid.astype("datetime64[D]"),
        t_days=t_grid,
        case_ids=ids,
        cum=cum[:-1],
        rate=rate[:-1],
        hist_cum=cum[-1],
        hist_rate=rate[-1],
        hist_rate_reported=reported,
        definition=definition,
        n_interpolated=n_interp,
    )
    n, F = len(ids), len(FLUIDS)
    for name in ("cum", "rate"):
        assert_array(f"align_all.{name}", getattr(out, name), (n, T, F), nonneg=True)
    for name in ("hist_cum", "hist_rate", "hist_rate_reported"):
        assert_array(f"align_all.{name}", getattr(out, name), (T, F), nonneg=True)
    assert_array("align_all.dates", out.dates, (T,), dtype=np.dtype("datetime64[D]"))
    assert_array("align_all.t_days", out.t_days, (T,))
    if (np.diff(out.cum, axis=1) < 0).any() or (np.diff(out.hist_cum, axis=0) < 0).any():
        raise ContractError("align_all: cumulative decreases along time")
    return out
