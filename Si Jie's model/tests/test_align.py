"""A03 to A09: dwc26.data.align."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.data.align import (
    Aligned,
    align_all,
    history_grid,
    interp_cum,
    rates_r1,
    rates_r2,
    to_days,
)
from dwc26.data.check import ContractError
from dwc26.data.load import load_parameters, load_production
from synth import CASES, GRID, HIST_CASE, T, case_rates


def test_a03_to_days():
    ts = np.array(["2007-12-06T08:41:00.938"], dtype="datetime64[ms]")
    t0 = np.datetime64("1998-01-01", "ms")
    d = to_days(ts, t0)
    assert d.dtype == np.float64 and d.shape == (1,)
    assert abs(d[0] - 3626.361816412037) <= 1e-9


def test_to_days_rejects_nat():
    ts = np.array(["NaT"], dtype="datetime64[ms]")
    with pytest.raises(ContractError):
        to_days(ts, np.datetime64("1998-01-01", "ms"))


def _piecewise_linear():
    # irregular, fractional knots; known slopes
    tau = np.array([0.0, 0.37, 1.9, 5.25, 7.0001, 10.5, 13.123456])
    slopes = np.array([3.0, 0.0, 12.5, 7.25, 1.0e-3, 40.0])
    cum = np.concatenate([[0.0], np.cumsum(slopes * np.diff(tau))])

    def analytic(t):
        j = np.searchsorted(tau, t, side="right") - 1
        j = np.clip(j, 0, len(slopes) - 1)
        return cum[j] + slopes[j] * (t - tau[j])

    return tau, cum, analytic


def test_a04_interp_cum_exact():
    tau, cum, analytic = _piecewise_linear()
    between = np.array([0.1, 0.5, 2.0, 5.3, 8.9, 12.0])
    out = interp_cum(tau, cum, between)
    np.testing.assert_allclose(out, analytic(between), rtol=1e-12, atol=0)
    on = tau.copy()
    np.testing.assert_array_equal(interp_cum(tau, cum, on), cum)
    mixed = np.array([0.0, 0.2, 1.9, 6.0, 13.123456])
    out = interp_cum(tau, cum, mixed)
    assert out[0] == cum[0] and out[2] == cum[2] and out[4] == cum[-1]


def test_a05_interp_cum_no_extrapolation():
    tau, cum, _ = _piecewise_linear()
    with pytest.raises(ContractError):
        interp_cum(tau, cum, np.array([-0.001, 1.0]))
    with pytest.raises(ContractError):
        interp_cum(tau, cum, np.array([1.0, tau[-1] + 1e-9]))


def test_interp_cum_requires_increasing_knots():
    with pytest.raises(ContractError):
        interp_cum(np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.5]))


def _quarterly(n=None, nf=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.array([0.0, 90.0, 181.0, 273.0, 365.0, 455.0])
    shape = (len(t) - 1, nf) if n is None else (n, len(t) - 1, nf)
    q = rng.uniform(0.0, 5000.0, size=shape)
    dt = np.diff(t)[:, None]
    inc = q * dt
    zero = np.zeros(shape[:-2] + (1, nf))
    cum = np.concatenate([zero, np.cumsum(inc, axis=-2)], axis=-2)
    return t, q, cum


@pytest.mark.parametrize("n", [None, 4])
def test_a06_rates_r1(n):
    t, q, cum = _quarterly(n)
    r = rates_r1(t, cum)
    assert r.shape == cum.shape and r.dtype == np.float64
    np.testing.assert_array_equal(r[..., 0, :], 0.0)
    np.testing.assert_allclose(r[..., 1:, :], q, rtol=1e-12, atol=0)


def test_a07_rates_r1_nonzero_start():
    t, _, cum = _quarterly()
    cum = cum + 0.0
    cum[0, 0] = 1.0
    cum[1:, 0] += 1.0
    with pytest.raises(ContractError):
        rates_r1(t, cum)


def test_rates_r1_negative_increment_raises():
    t, _, cum = _quarterly()
    cum[2, 1] = cum[1, 1] - 1.0
    with pytest.raises(ContractError):
        rates_r1(t, cum)


def test_a08_rates_r2():
    tau = np.array([0.0, 10.0, 25.5, 40.0])
    rate = np.array([0.0, 5.0, 7.0, 3.0])
    grid = np.array([0.0, 20.0, 25.5, 41.0])  # on, between, on, after the last report
    np.testing.assert_array_equal(rates_r2(tau, rate, grid), [0.0, 5.0, 7.0, 3.0])
    with pytest.raises(ContractError):
        rates_r2(tau, rate, np.array([-1.0, 5.0]))


def _aligned(synthetic_raw, definition="R1"):
    cfg = synthetic_raw.cfg
    params, _ = load_parameters(synthetic_raw.file01, cfg["file01"])
    raw = load_production(synthetic_raw.file02, cfg["file02"])
    grid = history_grid(raw)
    case_ids = params["case_id"].to_numpy()
    return raw, grid, align_all(raw, grid, case_ids, definition, cfg)


def test_history_grid(synthetic_raw):
    raw = load_production(synthetic_raw.file02, synthetic_raw.cfg["file02"])
    grid = history_grid(raw)
    assert grid.dtype == np.dtype("datetime64[ms]")
    np.testing.assert_array_equal(grid, np.array(GRID, dtype="datetime64[ms]"))


def test_align_all_shapes_and_values(synthetic_raw):
    _, grid, al = _aligned(synthetic_raw)
    n = len(CASES)
    assert isinstance(al, Aligned)
    assert al.dates.dtype == np.dtype("datetime64[D]") and al.dates.shape == (T,)
    assert al.t_days[0] == 0.0 and al.t_days.shape == (T,)
    assert al.cum.shape == (n, T, 3) and al.rate.shape == (n, T, 3)
    assert al.hist_cum.shape == (T, 3) and al.hist_rate.shape == (T, 3)
    assert al.hist_rate_reported.shape == (T, 3)
    assert al.definition == "R1"
    assert al.n_interpolated == 0
    np.testing.assert_array_equal(al.case_ids, np.array(CASES, dtype=np.int64))
    for idx, i in enumerate(CASES):
        r = case_rates(i)
        for f, fluid in enumerate(("oil", "gas", "water")):
            assert al.rate[idx, 0, f] == 0.0
            np.testing.assert_allclose(al.rate[idx, 1:, f], r[fluid], rtol=1e-12)
    # history: next-quarter reporting -> hist_rate[k] = reported[k-1]
    np.testing.assert_allclose(al.hist_rate[1:], al.hist_rate_reported[:-1], rtol=1e-12)


def test_a09_v4_same_code_path(synthetic_raw):
    _, _, al = _aligned(synthetic_raw)
    idx = CASES.index(HIST_CASE)
    assert np.array_equal(al.cum[idx], al.hist_cum)
    assert np.array_equal(al.rate[idx], al.hist_rate)  # bitwise


def test_align_all_r2(synthetic_raw):
    _, _, al = _aligned(synthetic_raw, "R2")
    assert al.definition == "R2"
    # the synthetic simulator reports the rate of the quarter ending at t_k
    for idx, i in enumerate(CASES):
        r = case_rates(i)
        np.testing.assert_array_equal(al.rate[idx, 1:, 0], r["oil"])
    # history R2 = its reported rates (last report at or before t_k)
    np.testing.assert_array_equal(al.hist_rate, al.hist_rate_reported)
