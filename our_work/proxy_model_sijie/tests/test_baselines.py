"""M25, M26: dwc26.models.baselines (B0 mean curve, B1 kNN)."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.models.baselines import KNNBenchmark, MeanCurveBenchmark
from our_work.proxy_model_sijie.tests.synth_models import make_curves, unit_X

KNN = {"knn": {"n_neighbors": 3, "weights": "distance", "algorithm": "brute"}}


def test_m25_b0_mean_curve():
    X = unit_X(12, 0)
    Y = make_curves(X, 13, 2)
    m = MeanCurveBenchmark().fit(X, Y, {})
    assert m.name == "b0" and m.variant == "main" and m.extrapolates_time is False
    for Xq in (unit_X(5, 1), np.zeros((1, 4)), np.ones((2, 4))):
        P = m.predict(Xq)
        assert P.shape == (len(Xq), 13, 3)
        for i in range(len(Xq)):
            assert np.array_equal(P[i], Y.mean(axis=0))
    assert m.last_n_clipped == 0
    assert m.predict_std(Xq) is None


def test_m26_b1_knn_by_hand():
    X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    T = 6
    rng = np.random.default_rng(4)
    Y = np.zeros((4, T, 3))
    Y[:, 1:] = rng.uniform(0.1, 1.0, size=(4, T - 1, 3))     # distinct curves, k0 = 1
    m = KNNBenchmark().fit(X, Y, KNN)
    assert m.name == "b1" and m.extrapolates_time is False
    q = np.array([[0.3, 0.2, 0.1, 0.0]])
    d = np.linalg.norm(X - q, axis=1)
    near = np.argsort(d)[:3]
    w = 1.0 / d[near]
    expected = (w[:, None, None] * Y[near]).sum(axis=0) / w.sum()
    np.testing.assert_allclose(m.predict(q)[0], expected, rtol=0, atol=1e-12)
    for i in range(4):
        assert np.array_equal(m.predict(X[i:i + 1])[0], Y[i])


def test_benchmarks_do_not_continue():
    X = unit_X(12, 0)
    Y = make_curves(X, 13, 2)
    for m in (MeanCurveBenchmark().fit(X, Y, {}), KNNBenchmark().fit(X, Y, KNN)):
        with pytest.raises(NotImplementedError, match="no time continuation"):
            m.forecast(X[:2], 1)
        with pytest.raises(NotImplementedError):
            m.rollout(X[:2], 14)
        np.testing.assert_array_equal(m.rollout(X[:2], 13), m.predict(X[:2]))
        np.testing.assert_array_equal(m.rollout(X[:2], 5), m.predict(X[:2])[:, :5])
