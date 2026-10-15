"""M09 and M10: dwc26.evaluation.metrics and the weighted training loss."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.evaluation.metrics import nrmse, objective_j, rmse


def _m09():
    oil_t = np.array([[0.0, 10.0], [0.0, 20.0]])
    oil_p = np.array([[0.0, 12.0], [0.0, 20.0]])
    w_t = np.array([[0.0, 4.0], [0.0, 8.0]])
    w_p = np.array([[0.0, 4.0], [0.0, 6.0]])
    yt = np.stack([oil_t, oil_t, w_t], axis=-1)
    yp = np.stack([oil_p, oil_p, w_p], axis=-1)
    return yt, yp


def test_m09_j_by_hand():
    yt, yp = _m09()
    r = rmse(yt, yp)
    n = nrmse(yt, yp)
    assert r.shape == (3,) and n.shape == (3,)
    assert r[0] == pytest.approx(1.0, abs=1e-12) and r[2] == pytest.approx(1.0, abs=1e-12)
    assert n[0] == pytest.approx(0.05, abs=1e-12) and n[2] == pytest.approx(0.125, abs=1e-12)
    assert objective_j(yt, yp, (0, 2)) == pytest.approx(0.0875, abs=1e-12)
    s = np.array([15678.18, 5695.88, 9166.93])
    np.testing.assert_allclose(nrmse(yt / s, yp / s), n, rtol=0, atol=1e-12)
    assert objective_j(yt / s, yp / s, (0, 2)) == pytest.approx(0.0875, abs=1e-12)


def test_m09_constant_true_series_raises():
    yt, yp = _m09()
    yt[..., 2] = 3.0
    with pytest.raises(ContractError):
        nrmse(yt, yp)
    with pytest.raises(ContractError):
        objective_j(yt, yp, (0, 2))


def test_metrics_shape_and_nan_checks():
    yt, yp = _m09()
    with pytest.raises(ContractError):
        rmse(yt, yp[:1])
    yp[0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        rmse(yt, yp)
    with pytest.raises(ContractError):
        objective_j(yt, yt, ())


def test_m10_weighted_loss_by_hand():
    torch = pytest.importorskip("torch")
    from dwc26.models.seq import weighted_mse

    pred = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]],
                         [[2.0, 2.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float32)
    target = torch.tensor([[[0.0, 2.0, 1.0], [0.5, 1.0, 0.0]],
                           [[1.0, 0.0, 0.0], [3.0, 1.0, 2.0]]], dtype=torch.float32)
    # squared errors per fluid over the 2 cases x 2 steps
    d2 = ((pred - target) ** 2).numpy().astype(np.float64)
    mse_f = d2.reshape(-1, 3).mean(axis=0)            # oil (1+0+1+4)/4, gas (0+1+4+0)/4, water (4+1+0+1)/4
    np.testing.assert_allclose(mse_f, [1.5, 1.25, 1.5])
    w = (0.25, 0.25, 0.5)
    hand = 0.25 * 1.5 + 0.25 * 1.25 + 0.5 * 1.5
    got = weighted_mse(pred, target, torch.tensor(w, dtype=torch.float32))
    assert got.dtype == torch.float32
    assert float(got) == pytest.approx(hand, abs=1e-6)
    eq = weighted_mse(pred, target, torch.tensor([1 / 3] * 3, dtype=torch.float32))
    assert float(eq) == pytest.approx(float(torch.nn.functional.mse_loss(pred, target)), abs=1e-6)


def test_m10_weight_set_sum_raises(tmp_path):
    from dwc26.config import ConfigError, load_model_config
    from our_work.proxy_model_sijie.tests.synth_models import tiny_cfg

    path = tiny_cfg(tmp_path, "lstm", fluid_weight_sets__d012=[0.2, 0.2, 0.5])
    with pytest.raises(ConfigError):
        load_model_config(path)


# --- Task 3: E01 to E05, E09 (spec task3 section 4.1) ------------------------------------------------

def _eval_cfg():
    import copy

    from our_work.proxy_model_sijie.tests.synth_eval import tiny_eval_cfg_dict

    t2 = {"db": "x.sqlite", "models": "m", "runs": {"b0": "r0", "b1": "r1", "lstm": "r2", "dmd": "r3"},
          "model_cfgs": {"lstm": "l.yaml", "dmd": "d.yaml"}}
    return copy.deepcopy(tiny_eval_cfg_dict(t2, "out"))


def test_e01_j_identity_in_scaled_units():
    from dwc26.evaluation.evaluate import evaluate_models
    from dwc26.evaluation.splits import TrainRef
    from dwc26.evaluation.store import Candidate
    from our_work.proxy_model_sijie.tests.synth_eval import FakeProxy, split_arrays

    rng = np.random.default_rng(1)
    s = np.array([15678.18, 5695.88, 9166.93])
    Y = rng.random((5, 6, 3))
    P = Y + 0.05 * rng.standard_normal(Y.shape)
    arr = split_arrays(Y * s, split="train", y_factor=s)
    ref = TrainRef(k0=0, mean_curve=(Y * s).mean(axis=0), gor_range=(0.0, 10.0))
    cand = Candidate("r/main", "b1", "main", "r", False, None, "b1/main")
    res = evaluate_models({"r/main": FakeProxy(P, 6, is_proxy=False)}, [cand], arr, ref, None, 0, _eval_cfg())
    j = next(r.value for r in res.rows if r.metric == "j")
    Pc = np.where(P < 0, 0.0, P)
    assert j == pytest.approx(objective_j(Y, Pc, (0, 2)), abs=1e-12)
    yt, yp = _m09()                                             # the task2 M09 example still passes
    assert objective_j(yt, yp, (0, 2)) == pytest.approx(0.0875, abs=1e-12)


def _e02():
    oil_t = np.array([[0.0, 10.0, 8.0], [0.0, 12.0, 6.0]])
    oil_p = np.array([[0.0, 11.0, 8.0], [0.0, 12.0, 5.0]])
    yt = np.stack([oil_t, oil_t, oil_t], axis=-1)
    yp = np.stack([oil_p, oil_p, oil_p], axis=-1)
    mean = np.stack([np.array([0.0, 10.0, 7.0])] * 3, axis=-1)
    return yt, yp, mean


def test_e02_r2_by_hand():
    from dwc26.evaluation.metrics import r2_case, r2_pooled
    from dwc26.evaluation.store import clean

    yt, yp, mean = _e02()
    W = np.arange(3) >= 1
    assert r2_pooled(yt, yp, W)[0] == pytest.approx(0.9, abs=1e-12)
    assert r2_case(yt, yp, mean, W)[0] == pytest.approx(2 / 3, abs=1e-12)
    b0 = np.broadcast_to(mean, yt.shape)
    assert r2_case(yt, b0, mean, W)[0] == 0.0
    assert r2_pooled(yt, yt, W)[0] == 1.0 and r2_case(yt, yt, mean, W)[0] == 1.0
    const = np.ones_like(yt)
    assert np.isnan(r2_pooled(const, yp, W)[0])
    assert np.isnan(r2_case(const, yp, np.ones((3, 3)), W)[0])
    assert clean(r2_pooled(const, yp, W)[0]) is None
    with pytest.raises(ContractError):
        r2_pooled(yt, yp, np.zeros(3, dtype=bool))               # a window with no date


def test_e03_rmse_by_date_and_pooled():
    from dwc26.evaluation.metrics import rmse_by_date, rmse_pooled

    yt = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    yp = yt.copy()
    yp[0, :, 0] += [1.0, 2.0, 3.0]
    yp[1, :, 0] += [3.0, 0.0, 1.0]
    yp[:, :, 2] += 2.0
    r = rmse_by_date(yt, yp)
    assert r.shape == (3, 3)
    np.testing.assert_allclose(r[:, 0], np.sqrt([(1 + 9) / 2, (4 + 0) / 2, (9 + 1) / 2]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(r[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(r[:, 2], 2.0, atol=1e-12)
    pooled = rmse_pooled(yt, yp)
    np.testing.assert_allclose(np.mean(r ** 2, axis=0), pooled ** 2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pooled, rmse(yt, yp), rtol=0, atol=0)


def test_e04_coverage():
    from dwc26.evaluation.metrics import coverage

    yt = np.zeros((1, 3, 3))
    yp = np.zeros((1, 3, 3))
    yp[0, 1, :] = 1.0                                # error 1
    yp[0, 2, :] = 3.0                                # error 3
    yp[0, 0, :] = 100.0                              # k < k0: excluded
    s = np.ones((1, 3, 3))
    s[0, 0, :] = 0.0
    W = np.arange(3) >= 1
    share, width = coverage(yt, yp, s, 2.0, W)
    np.testing.assert_allclose(share, 0.5)
    np.testing.assert_allclose(width, 2 * 2.0 * 1.0)
    share_all, _ = coverage(yt, yp, s, 2.0)
    np.testing.assert_allclose(share_all, 1 / 3)
    with pytest.raises(ContractError):
        coverage(yt, yp, -s, 2.0, W)


def test_e05_spearman_matches_scipy():
    from scipy.stats import spearmanr

    from dwc26.evaluation.metrics import spearman

    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(3, 20))
        a = rng.integers(0, 5, n).astype(float)               # ties
        b = a + rng.integers(-2, 3, n)
        if np.all(a == a[0]) or np.all(b == b[0]):
            assert spearman(a, b) is None
            continue
        assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)
    assert spearman(np.ones(5), np.arange(5.0)) is None
    assert spearman(np.arange(5.0), np.ones(5)) is None
    assert spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0])) is None


def test_e05_r2_scalar():
    from dwc26.evaluation.metrics import r2_scalar

    t = np.array([1.0, 2.0, 3.0])
    assert r2_scalar(t, t) == 1.0
    assert r2_scalar(t, np.array([1.0, 2.0, 4.0])) == pytest.approx(1 - 1 / 2, abs=1e-12)
    assert r2_scalar(np.ones(3), t) is None


def test_e09_local_set_and_errors():
    from dwc26.evaluation.metrics import local_errors, local_set

    O_sim = np.array([5.0, 1.0, 3.0, 1.0, 9.0])
    case_id = np.array([3, 4, 5, 6, 7])
    loc = local_set(O_sim, case_id, 2)
    assert case_id[loc].tolist() == [4, 6]
    assert case_id[local_set(O_sim, case_id, 3)].tolist() == [4, 5, 6]
    yt = np.zeros((5, 4, 3))
    yp = np.zeros((5, 4, 3))
    yp[1, 1, 0], yp[1, 2, 0] = 3.0, 4.0                       # case 4, inside K
    yp[3, 1, 0] = 1.0                                          # case 6, inside K
    yp[3, 3, 0] = 50.0                                         # case 6, outside K: ignored
    yp[0, 1, 0] = 99.0                                         # case 3: not local
    k_idx = np.array([1, 2])
    O_prx = np.array([7.0, 2.0, 3.0, 4.0, 9.0])
    r, dO = local_errors(yt, yp, k_idx, loc, O_sim, O_prx)
    assert r[0] == pytest.approx(np.sqrt((9 + 16 + 1 + 0) / 4), abs=1e-12)
    assert r[2] == 0.0
    assert dO == pytest.approx(np.median([2.0 - 1.0, 4.0 - 1.0]), abs=1e-12)
    with pytest.raises(ContractError):
        local_set(O_sim, case_id, 6)
