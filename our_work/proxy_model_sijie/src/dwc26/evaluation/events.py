"""Event dates of a rate curve and their errors in days (PRD section 7; spec task3 section 2.3; D-024).

- Plateau end: k_max = the first argmax of q_oil over k >= k0; the first k > k_max
  with q_oil(k) < plateau_frac q_oil(k_max). None if no such k, or if max q_oil <= 0.
- Water breakthrough: the first k >= k0 with q_w(k) > breakthrough_frac max_k q_w(k).
  None if max q_w <= 0.
- Water-oil crossover: the first k >= k0 with q_w(k) > q_oil(k).

Each curve uses its own maximum. -1 means no event. An event in one curve only
is counted (n_mismatch), not scored.
"""

from __future__ import annotations

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS

EVENTS = ("plateau_end", "breakthrough", "crossover")
EVENT_STATS = ("mae_days", "bias_days", "n_both", "n_mismatch", "n_none")
OIL, WATER = FLUIDS.index("oil"), FLUIDS.index("water")


def _first(mask: np.ndarray) -> int:
    hit = np.nonzero(mask)[0]
    return int(hit[0]) if hit.size else -1


def event_indices(Y_phys: np.ndarray, k0: int, cfg_events: dict) -> dict[str, np.ndarray]:
    """(n, T, 3) physical -> {event: (n,) int64 date index, -1 = no event}."""
    Y_phys = np.asarray(Y_phys)
    assert_array("event_indices Y_phys", Y_phys, (None, None, len(FLUIDS)), nonneg=True)
    n, T = Y_phys.shape[:2]
    if not 0 <= k0 < T:
        raise ContractError(f"event_indices: k0 = {k0} outside [0, {T})")
    pf, bf = float(cfg_events["plateau_frac"]), float(cfg_events["breakthrough_frac"])
    out = {e: np.full(n, -1, dtype=np.int64) for e in EVENTS}
    for i in range(n):
        oil, wat = Y_phys[i, :, OIL], Y_phys[i, :, WATER]
        tail = oil[k0:]
        if tail.max() > 0:
            k_max = k0 + int(np.argmax(tail))
            after = _first(oil[k_max + 1:] < pf * oil[k_max])
            out["plateau_end"][i] = -1 if after < 0 else k_max + 1 + after
        w_max = wat.max()
        if w_max > 0:
            k = _first(wat[k0:] > bf * w_max)
            out["breakthrough"][i] = -1 if k < 0 else k0 + k
        k = _first(wat[k0:] > oil[k0:])
        out["crossover"][i] = -1 if k < 0 else k0 + k
    return out


def event_errors(Y_true: np.ndarray, Y_pred: np.ndarray, t_days: np.ndarray, k0: int,
                 cfg_events: dict) -> dict[str, dict]:
    """{event: {mae_days, bias_days, n_both, n_mismatch, n_none}}. Error e = t[k_pred] - t[k_true]
    (positive = the prediction is late). mae and bias are None if n_both = 0."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    if Y_true.shape != Y_pred.shape:
        raise ContractError(f"event_errors: shapes {Y_true.shape} and {Y_pred.shape} differ")
    t_days = np.asarray(t_days, dtype=np.float64)
    assert_array("event_errors t_days", t_days, (Y_true.shape[1],))
    it, ip = event_indices(Y_true, k0, cfg_events), event_indices(Y_pred, k0, cfg_events)
    out = {}
    for e in EVENTS:
        a, b = it[e], ip[e]
        both = (a >= 0) & (b >= 0)
        one = (a >= 0) ^ (b >= 0)
        err = t_days[b[both]] - t_days[a[both]]
        nb = int(both.sum())
        out[e] = {"mae_days": float(np.mean(np.abs(err))) if nb else None,
                  "bias_days": float(np.mean(err)) if nb else None,
                  "n_both": nb, "n_mismatch": int(one.sum()), "n_none": int((~both & ~one).sum())}
    return out
