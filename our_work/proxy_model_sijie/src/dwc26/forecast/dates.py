"""Quarterly forecast dates (spec task3 section 1.3; Task 5 reuses them).

The history grid is quarterly on the same day of the month. The continuation
dates follow it: last + 3, 6, ... months. No date is hard-coded (V5).
"""

from __future__ import annotations

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError


def _day(d: np.datetime64) -> np.datetime64:
    try:
        return np.datetime64(d, "D")
    except (TypeError, ValueError) as e:
        raise ContractError(f"not a date: {d!r}") from e


def _split(d: np.datetime64) -> tuple[np.datetime64, int]:
    """(month, day offset from the first of the month)."""
    d = _day(d)
    m = d.astype("datetime64[M]")
    return m, int((d - m.astype("datetime64[D]")).astype(np.int64))


def quarters_between(d0, d1) -> int:
    """Whole quarters from d0 to d1. ContractError unless both dates fall on the same day of the
    month and the month difference is divisible by 3. Negative if d1 is before d0."""
    m0, off0 = _split(d0)
    m1, off1 = _split(d1)
    months = int((m1 - m0).astype(np.int64))
    if off0 != off1 or months % 3:
        raise ContractError(f"{_day(d1)} is not a whole number of quarters from {_day(d0)} "
                            f"(same day of the month and a month difference divisible by 3)")
    return months // 3


def forecast_dates(last, n: int) -> np.ndarray:
    """(n,) datetime64[D]: last + 3, 6, ..., 3n months, on the same day of the month."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ContractError(f"forecast_dates: n must be an integer >= 0, got {n!r}")
    m, off = _split(last)
    months = m + 3 * np.arange(1, int(n) + 1)
    out = months.astype("datetime64[D]") + np.timedelta64(off, "D")
    if n and not np.array_equal(out.astype("datetime64[M]"), months):
        raise ContractError(f"forecast_dates: day {off + 1} of the month does not exist in every forecast month")
    return out
