"""E07, E08: dwc26.hm.objective, the mismatch O(x) (PRD section 8; spec task3 section 2.4; D-025)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.hm.objective import build_mismatch_spec, mismatch, sigma_obs, window_indices

DATES = np.array(["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"], dtype="datetime64[D]")
CFG = {"window": ["2000-04-01", "2000-10-01"], "fluids": ["oil", "water"], "alpha": 0.10, "beta": 0.02}


def _hist():
    # oil 0, 100, 200, 150; gas = 0.36 oil; water 0, 0, 10, 1000 (the floor applies at 2000-04-01 and 2000-07-01)
    oil = np.array([0.0, 100.0, 200.0, 150.0])
    water = np.array([0.0, 0.0, 10.0, 1000.0])
    return np.stack([oil, 0.36 * oil, water], axis=-1)


def test_e07_window_inclusive():
    assert window_indices(DATES, "2000-04-01", "2000-10-01").tolist() == [1, 2, 3]
    assert window_indices(DATES, "2000-04-02", "2000-09-30").tolist() == [2]
    assert window_indices(DATES, "2001-01-01", "2001-02-01").size == 0


def test_e07_zero_at_history_and_hand_example():
    h = _hist()
    spec = build_mismatch_spec(h, DATES, CFG)
    assert spec.k_idx.tolist() == [1, 2, 3] and spec.fluid_idx.tolist() == [0, 2]
    assert mismatch(h[None], spec)[0] == 0.0
    # sigma_obs: oil max(0.1 d, 0.02 x 200 = 4) = 10, 20, 15; water max(0.1 d, 0.02 x 1000 = 20) = 20, 20, 100
    np.testing.assert_allclose(spec.sigma_obs, [[10, 20], [20, 20], [15, 100]], rtol=0, atol=1e-12)
    y = h.copy()
    y[1, 0] += 5.0        # oil +5 at sigma 10
    y[2, 0] -= 10.0       # oil -10 at sigma 20
    y[2, 2] += 4.0        # water +4 at sigma 20 (the floor)
    y[3, 2] -= 50.0       # water -50 at sigma 100
    O = mismatch(y[None], spec)[0]
    assert O == pytest.approx(0.25 + 0.25 + 0.04 + 0.25, abs=1e-12)


def test_e07_sum_not_mean_and_sigma_proxy():
    h = _hist()
    spec = build_mismatch_spec(h, DATES, CFG)
    y = h + np.array([5.0, 0.0, 5.0])
    O1 = mismatch(y[None], spec)[0]
    # doubling K with identical residuals doubles O: repeat the window dates
    dates2 = np.arange(np.datetime64("2000-01"), np.datetime64("2000-01") + 21, np.timedelta64(3, "M")).astype(
        "datetime64[D]")
    h2 = np.concatenate([h, h[1:]])
    y2 = np.concatenate([y, y[1:]])
    spec2 = build_mismatch_spec(h2, dates2, {**CFG, "window": ["2000-04-01", "2001-07-01"]})
    assert spec2.k_idx.size == 6
    assert mismatch(y2[None], spec2)[0] == pytest.approx(2 * O1, abs=1e-12)
    sp = np.full((3, 2), 7.0)
    spec_p = build_mismatch_spec(h, DATES, CFG, sigma_proxy=sp)
    np.testing.assert_allclose(spec_p.sigma, np.sqrt(spec.sigma_obs ** 2 + 49.0), rtol=0, atol=1e-12)
    assert mismatch(y[None], spec_p)[0] < O1
    np.testing.assert_allclose(sigma_obs(np.array([[0.0, 5.0], [10.0, 0.0]]), 0.1, 0.02),
                               [[0.2, 0.5], [1.0, 0.1]], rtol=0, atol=1e-12)


def test_e07_errors():
    h = _hist()
    zero_w = h.copy()
    zero_w[:, 2] = 0.0
    with pytest.raises(ContractError):
        build_mismatch_spec(zero_w, DATES, CFG)                         # max_K d_water = 0
    with pytest.raises(ContractError):
        build_mismatch_spec(h, DATES, {**CFG, "window": ["2001-01-01", "2001-12-31"]})   # K empty
    with pytest.raises(ContractError):
        build_mismatch_spec(h, DATES, CFG, sigma_proxy=np.ones((2, 2)))
    with pytest.raises(ContractError):
        build_mismatch_spec(h, DATES, CFG, sigma_proxy=-np.ones((3, 2)))
    spec = build_mismatch_spec(h, DATES, CFG)
    with pytest.raises(ContractError):
        mismatch(h[None, :3], spec)                                     # T <= max(k_idx)
    bad = h.copy()[None]
    bad[0, 1, 0] = np.nan
    with pytest.raises(ContractError):
        mismatch(bad, spec)


def test_e08_d_is_the_r1_history(tmp_path):
    """The spec's d equals hist_phys[K] (R1), not the reported column (D-004)."""
    from dwc26.data.db import read_task1_arrays
    from our_work.proxy_model_sijie.tests.synth_models import synthetic_task1_db

    s = synthetic_task1_db(tmp_path)
    con = sqlite3.connect(s["db"])
    con.execute("UPDATE history SET oil_rate_reported = oil_rate * 3 + 1, water_rate_reported = water_rate * 3 + 1")
    con.commit()
    a = read_task1_arrays(con)
    reported = np.array(con.execute("SELECT oil_rate_reported, gas_rate_reported, water_rate_reported FROM history "
                                    "ORDER BY date_idx").fetchall(), dtype=np.float64)
    con.close()
    cfg = {"window": ["2000-07-01", "2003-01-01"], "fluids": ["oil", "water"], "alpha": 0.1, "beta": 0.02}
    spec = build_mismatch_spec(a["hist_phys"], a["dates"], cfg)
    np.testing.assert_array_equal(spec.d, a["hist_phys"][spec.k_idx][:, [0, 2]])
    assert not np.allclose(spec.d, reported[spec.k_idx][:, [0, 2]])
