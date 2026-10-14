"""E10, E14: dwc26.evaluation.plausibility (spec task3 section 2.7; D-020, D-021, D-022)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from dwc26.data.check import ContractError
from dwc26.evaluation.plausibility import (continuation_checks, gor_check, gor_range, max_epoch_flags,
                                           model_continuation)
from synth_eval import FakeProxy

T, N_FC, N = 20, 16, 15
CFG = {"p2_mode": "reject", "p2_ratio": 1.05, "p2_case_share": 0.5, "p3_hist_quarters": 8, "p3_fc_quarters": [9, 16],
       "p3_ratio_min": 0.5, "p3_min_decline": 0.002, "gor_rel_tol": 0.005, "gor_oil_frac": 0.01}


def _decline(n=N):
    k = np.arange(T + N_FC, dtype=float)
    oil = 1000.0 * np.exp(-0.03 * k)
    R = np.zeros((n, T + N_FC, 3))
    R[..., 0] = oil
    R[..., 1] = 0.36 * oil
    R[..., 2] = 50.0
    return R


def test_e10_p1_counts_the_continuation_only():
    raw = _decline()
    raw[3, T + 5, 2] = -1.0                                       # one negative raw value in the continuation
    m = FakeProxy(raw, T)
    R, p1 = model_continuation(m, np.full((N, 4), 0.5), T, N_FC, np.ones(3))
    assert p1 == 1 and continuation_checks(R, T, p1, CFG)["p1_pass"] is False
    raw = _decline()
    raw[3, 5, 2] = -1.0                                           # a negative in the history only
    m = FakeProxy(raw, T)
    R, p1 = model_continuation(m, np.full((N, 4), 0.5), T, N_FC, np.ones(3))
    assert p1 == 0 and continuation_checks(R, T, p1, CFG)["p1_pass"] is True


def _rising(n_rise: int, ratio: float = 1.2):
    R = _decline()
    for i in range(n_rise):
        R[i, T + 3, 0] = ratio * R[i, T - 1, 0]
    return R


def test_e10_p2_case_share():
    out = continuation_checks(_rising(8), T, 0, CFG)
    assert out["p2_n_rising"] == 8 and out["p2_pass"] is False                       # 8 > 7.5
    assert out["p2_max_ratio"] == pytest.approx(1.2)
    out = continuation_checks(_rising(7), T, 0, CFG)
    assert out["p2_n_rising"] == 7 and out["p2_pass"] is True
    out = continuation_checks(_rising(15, 1.04), T, 0, CFG)
    assert out["p2_n_rising"] == 0 and out["p2_pass"] is True                        # a ratio of 1.04 does not count
    R = _decline()
    R[0, :T, 0] = 0.0                                                                 # q_hat = 0, later oil > 0
    out = continuation_checks(R, T, 0, CFG)
    assert out["p2_n_rising"] == 1


def test_e10_p3_ratio_and_exclusions():
    k = np.arange(T + N_FC, dtype=float)
    oil = np.where(k <= T - 1, np.exp(-0.03 * k), np.exp(-0.03 * (T - 1)) * np.exp(-0.01 * (k - T + 1)))
    R = np.zeros((N, T + N_FC, 3))
    R[..., 0] = 1000.0 * oil
    R[..., 2] = 10.0
    Y_true = R[:, :T].copy()
    out = continuation_checks(R, T, 0, CFG, Y_true)
    assert out["p3_n_used"] == N
    assert out["p3_R"] == pytest.approx(1 / 3, abs=1e-12)
    assert out["p3_flag"] is True
    assert out["p3_D_hist_true"] == pytest.approx(0.03, abs=1e-12)
    flat = R.copy()
    flat[0, :, 0] = 500.0                                                             # D_hist = 0 < 0.002: excluded
    out = continuation_checks(flat, T, 0, CFG)
    assert out["p3_n_used"] == N - 1 and out["p3_R"] == pytest.approx(1 / 3, abs=1e-12)
    flat[:, :, 0] = 500.0
    out = continuation_checks(flat, T, 0, CFG)
    assert out["p3_n_used"] == 0 and out["p3_R"] is None and out["p3_flag"] is None
    zero = R.copy()
    zero[1, T + 15, 0] = 0.0                                                          # a rate <= 0: excluded
    assert continuation_checks(zero, T, 0, CFG)["p3_n_used"] == N - 1
    with pytest.raises(ContractError):
        continuation_checks(R[:, :T + 10], T, 0, CFG)                                # n_fc < b


def test_e14_gor_flag_tolerance():
    Y = np.zeros((1, 3, 3))
    Y[0, :, 0] = 100.0
    Y[0, :, 1] = [36.0, 36.2, 36.4]
    g = gor_range(Y)
    assert g == (pytest.approx(0.36), pytest.approx(0.364))
    P = Y.copy()
    P[0, 1, 1] = 100.0 * 0.364 * 1.004
    assert gor_check(P, g, 0.01, 0.005)["flag"] is False
    P[0, 1, 1] = 100.0 * 0.364 * 1.006
    assert gor_check(P, g, 0.01, 0.005)["flag"] is True
    low = Y.copy()
    low[0, 2, 0] = 0.5                                                                # oil below 1% of its maximum
    low[0, 2, 1] = 10.0
    assert gor_check(low, g, 0.01, 0.005)["flag"] is False
    assert gor_check(np.zeros((1, 3, 3)), g, 0.01, 0.005)["flag"] is None


def _neural_dir(path, n_epochs: str):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"training": {"max_epochs": 1000}}), encoding="utf-8")
    (path / "trials.csv").write_text("trial,variant,status,error,lr,j_val,fit_s,best_epoch,n_epochs\n"
                                     f"0,A,ok,,0.001,0.1,1.0,500;480,{n_epochs}\n"
                                     "1,A,ok,,0.003,0.2,1.0,10;12,60;62\n", encoding="utf-8")
    (path / "A").mkdir(exist_ok=True)
    (path / "A" / "meta.json").write_text(json.dumps({"variant": "A", "members": [
        {"seed": 0, "n_epochs": 1000}, {"seed": 1, "n_epochs": 530}]}), encoding="utf-8")
    return path


def test_e14_max_epoch_flags(tmp_path):
    f = max_epoch_flags(_neural_dir(tmp_path / "a", "1000;530"), {"A": 0})
    assert f["flagged_trials"] == [0] and f["n_trials_max_epochs"] == 1
    assert f["selected_flagged"] == {"A": True}
    assert f["n_members_max_epochs"] == {"A": 1}
    assert f["trials"][0]["lr"] == pytest.approx(0.001)
    f = max_epoch_flags(_neural_dir(tmp_path / "b", "999;530"), {"A": 1})
    assert f["flagged_trials"] == [] and f["selected_flagged"] == {"A": False}
