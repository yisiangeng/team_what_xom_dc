"""Split by case number (V2). Spec task1 section 1.3."""

from __future__ import annotations

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.config import ConfigError

SPLITS = ("train", "val", "test")


def _ranges(ranges: dict) -> dict[str, tuple[int, int]]:
    if not isinstance(ranges, dict) or set(ranges) != set(SPLITS):
        raise ConfigError(f"split must have exactly the keys {list(SPLITS)}")
    out = {}
    for name in SPLITS:
        r = ranges[name]
        if (not isinstance(r, (list, tuple)) or len(r) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in r)):
            raise ConfigError(f"split.{name} must be [first, last] case numbers")
        lo, hi = int(r[0]), int(r[1])
        if lo > hi:
            raise ConfigError(f"split.{name}: first {lo} > last {hi}")
        out[name] = (lo, hi)
    return out


def assign_split(case_ids: np.ndarray, ranges: dict[str, list[int]]) -> np.ndarray:
    """(n,) case numbers -> (n,) '<U5' labels from inclusive case-number ranges.

    Raises ConfigError if a case is in no range or in two ranges."""
    rr = _ranges(ranges)
    ids = np.asarray(case_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ConfigError("case_ids must be a 1-D integer array")
    out = np.empty(ids.shape, dtype="<U5")
    hits = np.zeros(ids.shape, dtype=np.int64)
    for name, (lo, hi) in rr.items():
        m = (ids >= lo) & (ids <= hi)
        out[m] = name
        hits += m
    if (hits == 0).any():
        raise ConfigError(f"cases in no split range: {ids[hits == 0].tolist()}")
    if (hits > 1).any():
        raise ConfigError(f"cases in two split ranges: {ids[hits > 1].tolist()}")
    return out
