"""M06, M07, M27: dwc26.models.base (onset, V9 input checks, clipping)."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.data.scale import YScaler
from dwc26.models.base import check_X, check_XY, clip_nonneg, onset_index
from synth_models import ALL_MODELS, NEURAL_MODELS, PROXY_MODELS, curves_data, fit_model, make_curves, unit_X


# --- onset (M06) --------------------------------------------------------------

def test_m06_onset_index():
    X = unit_X(5, 0)
    assert onset_index(make_curves(X, 13, 2)) == 2
    assert onset_index(make_curves(X, 13, 0)) == 0
    with pytest.raises(ContractError):
        onset_index(np.zeros((5, 13, 3)))
    with pytest.raises(ContractError):
        onset_index(make_curves(X, 13, 11))          # T - k0 = 2
    assert onset_index(make_curves(X, 13, 10)) == 10  # T - k0 = 3


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m06_models_zero_before_onset(name, tmp_path):
    X, Y, Xv, Yv = curves_data()
    m = fit_model(name, tmp_path, X, Y, (Xv, Yv))
    assert m.k0 == 2 and m.T_fit == 13
    P = m.predict(Xv)
    assert np.all(P[:, :2] == 0.0)
    S = m.predict_std(Xv)
    if name in PROXY_MODELS:
        assert np.all(S[:, :2] == 0.0)
        R = m.rollout(Xv, 13 + 20)
        assert np.all(R[:, :2] == 0.0)
    else:
        assert S is None
    np.testing.assert_array_equal(m.rollout(Xv, 13), P)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m06_onset_zero_works(name, tmp_path):
    X, Xv = unit_X(12, 1), unit_X(4, 2)
    m = fit_model(name, tmp_path, X, make_curves(X, 13, 0), (Xv, make_curves(Xv, 13, 0)))
    assert m.k0 == 0
    P = m.predict(Xv)
    assert P.shape == (4, 13, 3) and np.isfinite(P).all() and (P >= 0).all()


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m06_fit_rejects_bad_onset(name, tmp_path):
    X, Xv = unit_X(12, 1), unit_X(4, 2)
    with pytest.raises(ContractError):
        fit_model(name, tmp_path, X, np.zeros((12, 13, 3)), (Xv, np.zeros((4, 13, 3))))
    with pytest.raises(ContractError):
        fit_model(name, tmp_path, X, make_curves(X, 13, 11), (Xv, make_curves(Xv, 13, 11)))


# --- V9 inputs (M07) ------------------------------------------------------------

def _bad_X():
    X = unit_X(4, 5)
    out = []
    a = X.copy(); a[0, 0] = np.nan; out.append(a)
    a = X.copy(); a[1, 2] = 1 + 1e-9; out.append(a)
    a = X.copy(); a[2, 1] = -1e-9; out.append(a)
    out.append(X[:, :3].copy())
    out.append(X[0])                       # 1-D
    out.append(np.zeros((0, 4)))           # n = 0
    return out


@pytest.mark.parametrize("i", range(6))
def test_m07_check_X(i):
    with pytest.raises(ContractError):
        check_X(_bad_X()[i], 4)


def test_m07_check_X_accepts_box_corners():
    X = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    out = check_X(X, 4)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, X)


def test_m07_check_XY():
    X, Y, _, _ = curves_data()
    check_XY(X, Y)
    bad = Y.copy(); bad[0, 5, 1] = -1e-12
    with pytest.raises(ContractError):
        check_XY(X, bad)
    bad = Y.copy(); bad[0, 5, 1] = np.nan
    with pytest.raises(ContractError):
        check_XY(X, bad)
    with pytest.raises(ContractError):
        check_XY(X[:-1], Y)
    with pytest.raises(ContractError):
        check_XY(X.astype(np.float32), Y)
    with pytest.raises(ContractError):
        check_XY(X, Y[..., :2])


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m07_models_reject_bad_inputs(name, tmp_path):
    X, Y, Xv, Yv = curves_data()
    for bad in _bad_X()[:4]:
        with pytest.raises(ContractError):
            fit_model(name, tmp_path, np.vstack([X[:-4], bad]) if bad.shape[1] == 4 else bad, Y, (Xv, Yv))
    bad = Y.copy(); bad[3, 4, 2] = -0.5
    with pytest.raises(ContractError):
        fit_model(name, tmp_path, X, bad, (Xv, Yv))
    bad = Y.copy(); bad[3, 4, 2] = np.nan
    with pytest.raises(ContractError):
        fit_model(name, tmp_path, X, bad, (Xv, Yv))
    with pytest.raises(ContractError):
        fit_model(name, tmp_path, X, Y, (Xv, Yv[:, :12]))
    m = fit_model(name, tmp_path, X, Y, (Xv, Yv))
    for bad in _bad_X():
        with pytest.raises(ContractError):
            m.predict(bad)
        with pytest.raises(ContractError):
            m.predict_std(bad)
    with pytest.raises(ContractError):
        m.rollout(Xv, 0)
    with pytest.raises(ContractError):
        m.rollout(Xv, 13.0)
    if name in PROXY_MODELS:
        with pytest.raises(ContractError):
            m.forecast(Xv, 0)


@pytest.mark.parametrize("name", NEURAL_MODELS)
def test_m07_neural_fit_requires_val(name, tmp_path):
    X, Y, _, _ = curves_data()
    with pytest.raises(ContractError):
        fit_model(name, tmp_path, X, Y, None)


# --- clipping (M27) ---------------------------------------------------------------

def test_m27_clip_nonneg():
    a = np.array([[-1.0, 0.0, 2.0], [-1e-300, 3.0, -5.0]])
    c, n = clip_nonneg(a)
    assert n == 3
    np.testing.assert_array_equal(c, [[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    assert (c >= 0).all()


def test_m27_clip_commutes_with_inverse():
    rng = np.random.default_rng(11)
    y = rng.normal(size=(50, 13, 3))
    ys = YScaler(fluids=("oil", "gas", "water"), factor=rng.uniform(1.0, 2e4, 3), fitted_on=(1,))
    a = ys.inverse(clip_nonneg(y)[0])
    b = clip_nonneg(ys.inverse(y))[0]
    assert np.array_equal(a, b)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name", ["lstm", "dmd"])
def test_m27_model_clips_and_counts(name, tmp_path, monkeypatch):
    X, Y, Xv, Yv = curves_data()
    m = fit_model(name, tmp_path, X, Y, (Xv, Yv))
    rng = np.random.default_rng(0)

    def raw(self, Xq, n_total):
        return rng.normal(size=(len(Xq), n_total - self.k0, 3))

    monkeypatch.setattr(type(m), "_raw_rollout", raw)
    P = m.predict(Xv)
    rng = np.random.default_rng(0)
    expected = rng.normal(size=(len(Xv), 13 - 2, 3))
    assert (P >= 0).all()
    assert m.last_n_clipped == int((expected < 0).sum()) > 0
    np.testing.assert_array_equal(P[:, 2:], np.maximum(expected, 0.0))
    rng = np.random.default_rng(1)
    F = m.forecast(Xv, 5)
    rng = np.random.default_rng(1)
    full = rng.normal(size=(len(Xv), 13 + 5 - 2, 3))
    assert F.shape == (4, 5, 3) and (F >= 0).all()
    assert m.last_n_clipped == int((full[:, 11:] < 0).sum())     # forecast counts the returned steps
