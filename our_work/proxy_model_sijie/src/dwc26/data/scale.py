"""X min-max scaling and Y per-fluid max scaling (spec task1 section 2.7).

X bounds are a design input (D-003). Y factors are fitted on the training
cases only (V1). History rates use the same factors (V4).
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from our_work.proxy_model_sijie.src.dwc26.config import ConfigError
from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS

Y_SOURCE = "max over train cases, all dates"


def _ids_text(ids) -> str:
    ids = [int(i) for i in ids]
    if ids and ids == list(range(ids[0], ids[0] + len(ids))):
        return f"{ids[0]}..{ids[-1]}"
    return ",".join(str(i) for i in ids)


@dataclass(frozen=True)
class XScaler:
    names: tuple[str, ...]          # config order
    lower: np.ndarray               # (p,) physical units
    upper: np.ndarray               # (p,)
    source: str                     # "design: min/max of file 01, cases <ids>" or the override citation

    def transform(self, X_phys: np.ndarray) -> np.ndarray:
        """(n, p) physical -> (n, p) scaled: (x - lower) / (upper - lower)."""
        X = np.asarray(X_phys, dtype=np.float64)
        assert_array("XScaler.transform input", X, (None, len(self.names)))
        U = (X - self.lower) / (self.upper - self.lower)
        assert_array("XScaler.transform", U, X.shape)
        return U

    def inverse(self, U: np.ndarray) -> np.ndarray:
        """(n, p) scaled -> (n, p) physical: lower + u (upper - lower)."""
        U = np.asarray(U, dtype=np.float64)
        assert_array("XScaler.inverse input", U, (None, len(self.names)))
        return self.lower + U * (self.upper - self.lower)


def fit_x_scaler(params: pd.DataFrame, cfg: dict) -> XScaler:
    """Design bounds = column min and max of file 01 (D-003), or a cited override
    that contains every design point. Raises ConfigError for an invalid override."""
    names = tuple(cfg["file01"]["columns"].values())
    X = params[list(names)].to_numpy(dtype=np.float64)
    assert_array("fit_x_scaler input", X, (None, len(names)))
    if len(X) == 0:
        raise ContractError("fit_x_scaler: no design points")
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    xb = cfg["x_bounds"]
    if xb["source"] == "design":
        lower, upper = xmin.copy(), xmax.copy()
        source = f"design: min/max of file 01, cases {_ids_text(params['case_id'])}"
        bad = [n for n, lo, hi in zip(names, lower, upper) if not hi > lo]
        if bad:
            raise ContractError(f"fit_x_scaler: constant design column(s) {bad}; upper must exceed lower")
    elif xb["source"] == "override":
        cit = xb.get("citation")
        if not isinstance(cit, str) or not cit.strip():
            raise ConfigError("x_bounds.citation must be non-empty when x_bounds.source = override")
        ov = xb.get("override")
        if not isinstance(ov, dict):
            raise ConfigError("x_bounds.override must map each parameter to [lower, upper]")
        lower, upper = np.empty(len(names)), np.empty(len(names))
        for j, n in enumerate(names):
            b = ov.get(n)
            if (not isinstance(b, (list, tuple)) or len(b) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                               for v in b)):
                raise ConfigError(f"x_bounds.override.{n} must be [lower, upper]")
            lo, hi = float(b[0]), float(b[1])
            if not lo < hi:
                raise ConfigError(f"x_bounds.override.{n}: lower {lo} must be < upper {hi}")
            if lo > xmin[j] or hi < xmax[j]:
                raise ConfigError(f"x_bounds.override.{n} = [{lo}, {hi}] excludes design points "
                                  f"(min {xmin[j]}, max {xmax[j]})")
            lower[j], upper[j] = lo, hi
        extra = set(ov) - set(names)
        if extra:
            raise ConfigError(f"x_bounds.override: unknown parameters {sorted(extra)}")
        source = cit.strip()
    else:
        raise ConfigError(f"x_bounds.source must be design or override, got {xb['source']!r}")
    return XScaler(names=names, lower=lower, upper=upper, source=source)


@dataclass(frozen=True)
class YScaler:
    fluids: tuple[str, ...]         # ('oil', 'gas', 'water')
    factor: np.ndarray              # (3,) STB/d, MSCF/d, STB/d
    fitted_on: tuple[int, ...]      # case_ids used: train only

    def transform(self, Y_phys: np.ndarray) -> np.ndarray:
        """(..., 3) physical -> (..., 3) scaled: q / s_f."""
        Y = np.asarray(Y_phys, dtype=np.float64)
        if Y.ndim < 1 or Y.shape[-1] != len(self.fluids):
            raise ContractError(f"YScaler.transform: last axis must be {len(self.fluids)}, got {Y.shape}")
        return Y / self.factor

    def inverse(self, Y: np.ndarray) -> np.ndarray:
        """(..., 3) scaled -> (..., 3) physical: y * s_f. No clipping."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim < 1 or Y.shape[-1] != len(self.fluids):
            raise ContractError(f"YScaler.inverse: last axis must be {len(self.fluids)}, got {Y.shape}")
        return Y * self.factor


def fit_y_scaler(rate: np.ndarray, case_ids: np.ndarray, split: np.ndarray) -> YScaler:
    """s_f = max over the train cases and all dates of q_f (V1: train rows only)."""
    rate = np.asarray(rate)
    ids = np.asarray(case_ids)
    split = np.asarray(split)
    assert_array("fit_y_scaler rate", rate, (len(ids), None, len(FLUIDS)), nonneg=True)
    if split.shape != ids.shape:
        raise ContractError(f"fit_y_scaler: split {split.shape} and case_ids {ids.shape} differ")
    train = split == "train"
    if not train.any():
        raise ContractError("fit_y_scaler: no training case")
    factor = rate[train].max(axis=(0, 1))
    if not (factor > 0).all():
        raise ContractError(f"fit_y_scaler: factor must be > 0 for every fluid, got {factor}")
    return YScaler(fluids=FLUIDS, factor=factor.astype(np.float64),
                   fitted_on=tuple(int(i) for i in ids[train]))


def load_scalers(con: sqlite3.Connection) -> tuple[XScaler, YScaler]:
    """Rebuilds both scalers from the `scalers` table (used by Tasks 2 to 5)."""
    rows = con.execute(
        "SELECT name, kind, variable, fluid, factor, lower, upper, source, fitted_on "
        "FROM scalers ORDER BY rowid"
    ).fetchall()
    xr = [r for r in rows if r[1] == "x_minmax"]
    yr = {r[3]: r for r in rows if r[1] == "y_max"}
    if not xr or set(yr) != set(FLUIDS):
        raise ContractError("load_scalers: the scalers table is empty or incomplete; run prepare first")
    sources = {r[7] for r in xr}
    if len(sources) != 1:
        raise ContractError(f"load_scalers: inconsistent X sources {sources}")
    xs = XScaler(
        names=tuple(r[2] for r in xr),
        lower=np.array([r[5] for r in xr], dtype=np.float64),
        upper=np.array([r[6] for r in xr], dtype=np.float64),
        source=sources.pop(),
    )
    fitted = {r[8] for r in yr.values()}
    if len(fitted) != 1:
        raise ContractError("load_scalers: inconsistent fitted_on of the Y factors")
    ys = YScaler(
        fluids=FLUIDS,
        factor=np.array([yr[f][4] for f in FLUIDS], dtype=np.float64),
        fitted_on=tuple(int(i) for i in json.loads(fitted.pop())),
    )
    return xs, ys
