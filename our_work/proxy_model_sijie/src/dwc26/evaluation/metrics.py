"""RMSE, NRMSE, the objective J (spec task2 section 2.5) and the Task 3 metrics (spec task3 section 2).

Arrays are (n, T, F) or any shape whose last axis is the fluid. RMSE pools
all cases and dates of the split. NRMSE divides by the range (max - min) of
the true values in the split, so it does not depend on the fluid scale, and
J is the same in scaled and physical units. Task 3 reuses these functions,
so that the ranking metric equals the tuning objective.
"""

from __future__ import annotations

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError


def _pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ContractError(f"metrics: y_true {yt.shape} and y_pred {yp.shape} differ")
    if yt.ndim < 2 or yt.size == 0:
        raise ContractError(f"metrics: expected (..., F) arrays with values, got {yt.shape}")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ContractError("metrics: non-finite values")
    return yt, yp


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(F,): sqrt of the mean squared error over every axis but the last."""
    yt, yp = _pair(y_true, y_pred)
    d = yp - yt
    return np.sqrt(np.mean(d * d, axis=tuple(range(yt.ndim - 1))))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(F,): RMSE / (max - min of y_true). ContractError if a range is 0."""
    yt, yp = _pair(y_true, y_pred)
    axes = tuple(range(yt.ndim - 1))
    span = yt.max(axis=axes) - yt.min(axis=axes)
    if not (span > 0).all():
        raise ContractError(f"nrmse: the true values of fluid(s) {np.nonzero(span <= 0)[0].tolist()} are constant")
    return rmse(yt, yp) / span


def objective_j(y_true: np.ndarray, y_pred: np.ndarray, fluid_idx: tuple[int, ...]) -> float:
    """J = mean NRMSE over the fluids in fluid_idx (oil and water by default)."""
    idx = [int(i) for i in fluid_idx]
    if not idx:
        raise ContractError("objective_j: no fluid selected")
    return float(np.mean(nrmse(y_true, y_pred)[idx]))


# --- Task 3 (spec task3 sections 2.2 to 2.5) ---------------------------------------------
# y arrays are (n, T, F); `window` is a boolean (T,) mask over the dates, or None (all dates).

def _cube(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = _pair(y_true, y_pred)
    if yt.ndim != 3:
        raise ContractError(f"metrics: expected (n, T, F) arrays, got {yt.shape}")
    return yt, yp


def _mask(window, T: int) -> np.ndarray:
    if window is None:
        return np.ones(T, dtype=bool)
    w = np.asarray(window)
    if w.dtype != bool or w.shape != (T,):
        raise ContractError(f"metrics: window must be a boolean ({T},) mask, got {w.dtype} {w.shape}")
    if not w.any():
        raise ContractError("metrics: the window has no date")
    return w


def _ratio(sse: np.ndarray, sst: np.ndarray) -> np.ndarray:
    out = np.full(sse.shape, np.nan)
    ok = sst > 0
    out[ok] = 1.0 - sse[ok] / sst[ok]
    return out


def r2_pooled(y_true, y_pred, window=None) -> np.ndarray:
    """(F,): 1 - SSE / sum (y - y_bar)^2 over the cases and the window dates, y_bar = their mean.
    NaN where the denominator is 0. Form of Article 4, Eq. 16; the window is D-024."""
    yt, yp = _cube(y_true, y_pred)
    w = _mask(window, yt.shape[1])
    t, p = yt[:, w], yp[:, w]
    sse = np.sum((p - t) ** 2, axis=(0, 1))
    sst = np.sum((t - t.mean(axis=(0, 1))) ** 2, axis=(0, 1))
    return _ratio(sse, sst)


def r2_case(y_true, y_pred, mean_curve, window=None) -> np.ndarray:
    """(F,): 1 - SSE / sum (y - m_k)^2 with m the train mean curve (T, F) = SSE of B0.
    NaN where the denominator is 0 (PRD section 7)."""
    yt, yp = _cube(y_true, y_pred)
    m = np.asarray(mean_curve, dtype=np.float64)
    if m.shape != yt.shape[1:] or not np.isfinite(m).all():
        raise ContractError(f"r2_case: mean_curve {m.shape} must be finite with shape {yt.shape[1:]}")
    w = _mask(window, yt.shape[1])
    t, p, mm = yt[:, w], yp[:, w], m[w]
    sse = np.sum((p - t) ** 2, axis=(0, 1))
    sst = np.sum((t - mm[None]) ** 2, axis=(0, 1))
    return _ratio(sse, sst)


def rmse_by_date(y_true, y_pred) -> np.ndarray:
    """(T, F): the RMSE over the cases at each date (sigma_proxy of PRD section 8)."""
    yt, yp = _cube(y_true, y_pred)
    return np.sqrt(np.mean((yp - yt) ** 2, axis=0))


def rmse_pooled(y_true, y_pred) -> np.ndarray:
    """(F,): the RMSE over every axis but the last (= rmse)."""
    return rmse(y_true, y_pred)


def coverage(y_true, y_pred, std, z: float, window=None) -> tuple[np.ndarray, np.ndarray]:
    """(share (F,), mean interval width 2 z s (F,)): the share of |y - y_hat| <= z s over the
    cases and the window dates (PRD section 7; window D-024)."""
    yt, yp = _cube(y_true, y_pred)
    s = np.asarray(std, dtype=np.float64)
    if s.shape != yt.shape or not np.isfinite(s).all() or (s < 0).any():
        raise ContractError(f"coverage: std must be finite and >= 0 with shape {yt.shape}, got {s.shape}")
    if not (np.isfinite(z) and z > 0):
        raise ContractError(f"coverage: z must be > 0, got {z}")
    w = _mask(window, yt.shape[1])
    err, sd = np.abs(yt - yp)[:, w], s[:, w]
    share = np.mean(err <= z * sd, axis=(0, 1))
    width = 2.0 * z * np.mean(sd, axis=(0, 1))
    return share, width


def _vector(name: str, a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ContractError(f"{name}: expected a finite 1-D array, got shape {a.shape}")
    return a


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman's rho: the Pearson correlation of the average ranks (ties take the mean rank).
    None if either vector is constant or n < 3."""
    from scipy.stats import rankdata

    a, b = _vector("spearman a", a), _vector("spearman b", b)
    if a.shape != b.shape:
        raise ContractError(f"spearman: shapes {a.shape} and {b.shape} differ")
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra, rb = rankdata(a, method="average"), rankdata(b, method="average")
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float(np.sum(ra * rb) / np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))


def r2_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """R^2 of Article 4, Eq. 16 on a scalar output: 1 - sum (y_pred - y)^2 / sum (y - mean y)^2.
    None if y_true is constant."""
    t, p = _vector("r2_scalar y_true", y_true), _vector("r2_scalar y_pred", y_pred)
    if t.shape != p.shape or t.size == 0:
        raise ContractError(f"r2_scalar: shapes {t.shape} and {p.shape}")
    sst = float(np.sum((t - t.mean()) ** 2))
    if not sst > 0:
        return None
    return 1.0 - float(np.sum((p - t) ** 2)) / sst


def local_set(O_sim: np.ndarray, case_id: np.ndarray, local_n: int) -> np.ndarray:
    """Indices of the local_n cases with the lowest O_sim; ties go to the lower case ID (D-025)."""
    O = _vector("local_set O_sim", O_sim)
    cid = np.asarray(case_id)
    if cid.shape != O.shape:
        raise ContractError(f"local_set: case_id {cid.shape} and O_sim {O.shape} differ")
    if not 1 <= local_n <= O.size:
        raise ContractError(f"local_set: local_n = {local_n} must lie in [1, n = {O.size}]")
    order = np.lexsort((cid, O))
    return np.sort(order[:local_n])


def local_errors(y_true, y_pred, k_idx: np.ndarray, local: np.ndarray, O_sim: np.ndarray,
                 O_prx: np.ndarray) -> tuple[np.ndarray, float]:
    """(local RMSE (F,) over the local cases and the dates K, median of O_prx - O_sim over them)."""
    yt, yp = _cube(y_true, y_pred)
    sub = np.ix_(np.asarray(local), np.asarray(k_idx))
    d = yp[sub] - yt[sub]
    r = np.sqrt(np.mean(d * d, axis=(0, 1)))
    dO = _vector("local_errors O_prx", O_prx)[local] - _vector("local_errors O_sim", O_sim)[local]
    return r, float(np.median(dO))
