"""A19 to A22: dwc26.data.scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwc26.config import ConfigError
from dwc26.data.check import ContractError
from dwc26.data.scale import XScaler, YScaler, fit_x_scaler, fit_y_scaler
from synth import synthetic_config

NAMES = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]


def _params(n=20, seed=1):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(0.01, 0.15, n),
        rng.uniform(0.8, 1.5, n),
        rng.uniform(0.5, 9.8, n),
        rng.uniform(50.0, 200.0, n),
    ])
    df = pd.DataFrame(X, columns=NAMES)
    df.insert(0, "case_id", np.arange(1, n + 1, dtype=np.int64))
    return df


def _cfg(**xb):
    cfg = synthetic_config("/tmp/unused")
    cfg["x_bounds"] = {"source": "design", "override": None, "citation": None, **xb}
    return cfg


def test_a19_x_round_trip_design():
    params = _params()
    xs = fit_x_scaler(params, _cfg())
    assert isinstance(xs, XScaler)
    assert xs.names == tuple(NAMES)
    X = params[NAMES].to_numpy()
    np.testing.assert_array_equal(xs.lower, X.min(axis=0))
    np.testing.assert_array_equal(xs.upper, X.max(axis=0))
    U = xs.transform(X)
    assert U.shape == X.shape and U.dtype == np.float64
    np.testing.assert_array_equal(U.min(axis=0), 0.0)
    np.testing.assert_array_equal(U.max(axis=0), 1.0)
    np.testing.assert_allclose(xs.inverse(U), X, rtol=1e-12, atol=0)
    assert xs.source.startswith("design")


def test_a20_x_override():
    params = _params()
    X = params[NAMES].to_numpy()
    lo, hi = X.min(axis=0), X.max(axis=0)
    override = {n: [float(lo[j]) - 0.001, float(hi[j]) + 1.0] for j, n in enumerate(NAMES)}
    xs = fit_x_scaler(params, _cfg(source="override", override=override, citation="Kick-off slide 7"))
    assert xs.source == "Kick-off slide 7"
    np.testing.assert_array_equal(xs.lower, lo - 0.001)
    U = xs.transform(X)
    assert (U > 0).all() and (U < 1).all()

    # an override that excludes a design point
    bad = dict(override)
    bad["perm_mult"] = [float(lo[2]) + 0.1, float(hi[2])]
    with pytest.raises(ConfigError):
        fit_x_scaler(params, _cfg(source="override", override=bad, citation="x"))
    # empty citation
    with pytest.raises(ConfigError):
        fit_x_scaler(params, _cfg(source="override", override=override, citation=""))
    with pytest.raises(ConfigError):
        fit_x_scaler(params, _cfg(source="override", override=override, citation=None))
    # missing parameter, inverted bounds
    with pytest.raises(ConfigError):
        fit_x_scaler(params, _cfg(source="override", override={k: v for k, v in override.items() if k != "poro_mult"},
                                  citation="x"))
    inv = dict(override)
    inv["poro_mult"] = [2.0, 0.5]
    with pytest.raises(ConfigError):
        fit_x_scaler(params, _cfg(source="override", override=inv, citation="x"))


def test_x_constant_column_raises():
    params = _params()
    params["poro_mult"] = 1.0
    with pytest.raises(ContractError):
        fit_x_scaler(params, _cfg())


def _rates(n=10, T=7, seed=2):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, 1000.0, size=(n, T, 3))
    r[:, 0, :] = 0.0
    return r


def test_a21_y_round_trip():
    rate = _rates()
    ids = np.arange(1, 11, dtype=np.int64)
    split = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2, dtype="<U5")
    ys = fit_y_scaler(rate, ids, split)
    assert isinstance(ys, YScaler)
    assert ys.fluids == ("oil", "gas", "water")
    assert ys.fitted_on == tuple(range(1, 7))
    assert ys.factor.shape == (3,) and (ys.factor > 0).all()
    np.testing.assert_array_equal(ys.factor, rate[:6].max(axis=(0, 1)))
    Y = ys.transform(rate)
    np.testing.assert_allclose(ys.inverse(Y), rate, rtol=1e-12, atol=0)
    h = rate[7]
    np.testing.assert_allclose(ys.inverse(ys.transform(h)), h, rtol=1e-12, atol=0)
    assert Y[:6].max() == 1.0


def test_a22_y_uses_train_only():
    rate = _rates()
    ids = np.arange(1, 11, dtype=np.int64)
    split = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2, dtype="<U5")
    f0 = fit_y_scaler(rate, ids, split).factor
    r1 = rate.copy()
    r1[6:] *= 10.0
    np.testing.assert_array_equal(fit_y_scaler(r1, ids, split).factor, f0)
    r2 = rate.copy()
    r2[2] *= 10.0
    assert (fit_y_scaler(r2, ids, split).factor != f0).all()


def test_y_zero_factor_raises():
    rate = _rates()
    rate[..., 2] = 0.0
    ids = np.arange(1, 11, dtype=np.int64)
    split = np.array(["train"] * 10, dtype="<U5")
    with pytest.raises(ContractError):
        fit_y_scaler(rate, ids, split)
