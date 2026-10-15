"""E06: dwc26.evaluation.events (spec task3 section 2.3; D-024)."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.evaluation.events import EVENTS, event_errors, event_indices

CFG = {"plateau_frac": 0.98, "breakthrough_frac": 0.01}
OIL = np.array([0, 0, 95, 100, 100, 99, 90, 80], dtype=float)
WATER = np.array([0, 0, 0, 0, 0.5, 10, 50, 100], dtype=float)
T_DAYS = np.array([0, 91, 182, 274, 366, 456, 547, 639], dtype=float)


def _curve(oil, water):
    return np.stack([oil, 0.36 * oil, water], axis=-1)[None]


def test_e06_plateau_end_from_the_maximum():
    ev = event_indices(_curve(OIL, WATER), 2, CFG)
    # the ramp quarter (95 < 98) at k0 = 2 does not end the plateau; the first k after the maximum
    # (k = 3) with oil < 0.98 x 100 is k = 6 (99 at k = 5 is not below 98)
    assert ev["plateau_end"][0] == 6
    assert event_indices(_curve(np.zeros(8), WATER), 2, CFG)["plateau_end"][0] == -1
    flat = np.array([0, 0, 100, 100, 100, 100, 100, 100], dtype=float)
    assert event_indices(_curve(flat, WATER), 2, CFG)["plateau_end"][0] == -1


def test_e06_breakthrough_and_crossover():
    ev = event_indices(_curve(OIL, WATER), 2, CFG)
    assert ev["breakthrough"][0] == 5                    # first k >= k0 with water > 0.01 x 100
    assert ev["crossover"][0] == 7                       # first k >= k0 with water > oil (100 > 80)
    none = event_indices(_curve(OIL, 0.1 * WATER), 2, CFG)
    assert none["crossover"][0] == -1
    assert event_indices(_curve(OIL, np.zeros(8)), 2, CFG)["breakthrough"][0] == -1


def test_e06_errors_in_days_with_sign():
    true = _curve(OIL, WATER)
    late = OIL.copy()
    late[6] = 99.0                                       # plateau end moves from k = 6 to k = 7
    pred = _curve(late, WATER)
    err = event_errors(true, pred, T_DAYS, 2, CFG)
    pe = err["plateau_end"]
    assert pe["n_both"] == 1 and pe["n_mismatch"] == 0 and pe["n_none"] == 0
    assert pe["mae_days"] == pytest.approx(T_DAYS[7] - T_DAYS[6])
    assert pe["bias_days"] == pytest.approx(T_DAYS[7] - T_DAYS[6])         # positive: the prediction is late
    early = event_errors(pred, true, T_DAYS, 2, CFG)["plateau_end"]
    assert early["bias_days"] == pytest.approx(-(T_DAYS[7] - T_DAYS[6]))
    assert err["breakthrough"]["mae_days"] == 0.0


def test_e06_event_in_one_curve_only_is_counted_not_scored():
    true = _curve(OIL, WATER)
    pred = _curve(OIL, 0.1 * WATER)                      # no crossover in the prediction
    err = event_errors(true, pred, T_DAYS, 2, CFG)
    cx = err["crossover"]
    assert cx == {"mae_days": None, "bias_days": None, "n_both": 0, "n_mismatch": 1, "n_none": 0}
    both_none = event_errors(pred, pred, T_DAYS, 2, CFG)["crossover"]
    assert both_none["n_none"] == 1 and both_none["mae_days"] is None
    assert set(err) == set(EVENTS)
