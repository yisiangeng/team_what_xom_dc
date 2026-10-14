"""Mismatch O(x) between a rate curve and RAMP_History (PRD section 8; spec task3 section 2.4; D-025).

O = sum over the fluids F_O and the dates K of ((y_hat - d) / sigma)^2, with
sigma^2 = sigma_obs^2 + sigma_proxy^2 and sigma_obs = max(alpha d, beta max_K d).
d is `hist_phys`: the R1 history rates (D-004), never the reported column.
Task 3 uses O_obs (sigma_proxy = 0) for every model. Task 4 reuses this module
with sigma_proxy = the validation RMSE by date of the frozen primary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dwc26.data.check import ContractError, assert_array
from dwc26.data.load import FLUIDS


def window_indices(dates: np.ndarray, start, end) -> np.ndarray:
    """(|K|,) int64: the indices k with start <= dates[k] <= end (inclusive)."""
    dates = np.asarray(dates).astype("datetime64[D]")
    s, e = np.datetime64(start, "D"), np.datetime64(end, "D")
    return np.nonzero((dates >= s) & (dates <= e))[0].astype(np.int64)


def sigma_obs(d: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """(|K|, F_O) -> (|K|, F_O): max(alpha d, beta max_K d), column by column."""
    d = np.asarray(d, dtype=np.float64)
    assert_array("sigma_obs d", d, (None, None), nonneg=True)
    return np.maximum(alpha * d, beta * d.max(axis=0, keepdims=True))


@dataclass(frozen=True)
class MismatchSpec:
    k_idx: np.ndarray          # (|K|,) int64
    fluid_idx: np.ndarray      # (F_O,) int64, indices into FLUIDS
    d: np.ndarray              # (|K|, F_O) physical history rates, hist_phys[k_idx][:, fluid_idx]
    sigma_obs: np.ndarray      # (|K|, F_O) > 0
    sigma_proxy: np.ndarray    # (|K|, F_O) >= 0; zeros for O_obs
    sigma: np.ndarray          # (|K|, F_O) = sqrt(sigma_obs^2 + sigma_proxy^2) > 0


def build_mismatch_spec(hist_phys: np.ndarray, dates: np.ndarray, cfg_mismatch: dict,
                        sigma_proxy: np.ndarray | None = None) -> MismatchSpec:
    """The window K, d, sigma_obs and sigma from the config block `mismatch`. ContractError if K is
    empty, if a fluid has max_K d = 0, or if sigma_proxy is not (|K|, F_O), finite and >= 0."""
    hist_phys = np.asarray(hist_phys, dtype=np.float64)
    dates = np.asarray(dates).astype("datetime64[D]")
    assert_array("build_mismatch_spec hist_phys", hist_phys, (len(dates), len(FLUIDS)), nonneg=True)
    start, end = cfg_mismatch["window"]
    k_idx = window_indices(dates, start, end)
    if k_idx.size == 0:
        raise ContractError(f"build_mismatch_spec: no history date in the window [{start}, {end}]")
    unknown = [f for f in cfg_mismatch["fluids"] if f not in FLUIDS]
    if unknown or not cfg_mismatch["fluids"]:
        raise ContractError(f"build_mismatch_spec: fluids {cfg_mismatch['fluids']} must be a non-empty subset "
                            f"of {FLUIDS}")
    fluid_idx = np.array([FLUIDS.index(f) for f in cfg_mismatch["fluids"]], dtype=np.int64)
    d = hist_phys[np.ix_(k_idx, fluid_idx)]
    zero = [cfg_mismatch["fluids"][j] for j in range(len(fluid_idx)) if not d[:, j].max() > 0]
    if zero:
        raise ContractError(f"build_mismatch_spec: the history of {zero} is 0 on every date of K, so sigma_obs "
                            "would be 0")
    so = sigma_obs(d, float(cfg_mismatch["alpha"]), float(cfg_mismatch["beta"]))
    if sigma_proxy is None:
        sp = np.zeros_like(so)
    else:
        sp = np.asarray(sigma_proxy, dtype=np.float64)
        assert_array("build_mismatch_spec sigma_proxy", sp, so.shape, nonneg=True)
    sigma = np.sqrt(so ** 2 + sp ** 2)
    if not (sigma > 0).all():
        raise ContractError("build_mismatch_spec: sigma must be > 0 on every date and fluid of K")
    for a in (k_idx, fluid_idx, d, so, sp, sigma):
        a.setflags(write=False)
    return MismatchSpec(k_idx=k_idx, fluid_idx=fluid_idx, d=d, sigma_obs=so, sigma_proxy=sp, sigma=sigma)


def mismatch(Y_phys: np.ndarray, spec: MismatchSpec) -> np.ndarray:
    """(n, T, 3) physical rates -> (n,) O >= 0 (a sum over K and F_O, not a mean)."""
    Y_phys = np.asarray(Y_phys)
    assert_array("mismatch Y_phys", Y_phys, (None, None, len(FLUIDS)), nonneg=True)
    if Y_phys.shape[1] <= int(spec.k_idx.max()):
        raise ContractError(f"mismatch: Y_phys has T = {Y_phys.shape[1]} dates, K needs index {int(spec.k_idx.max())}")
    y = Y_phys[:, spec.k_idx][:, :, spec.fluid_idx]                     # (n, |K|, F_O)
    r = (y - spec.d[None]) / spec.sigma[None]
    O = np.sum(r * r, axis=(1, 2))
    assert_array("mismatch O", O, (Y_phys.shape[0],), nonneg=True)
    return O
