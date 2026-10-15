"""M18 to M23 and M42 to M44 (amendment A2): dwc26.models.dmd."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.evaluation.metrics import nrmse
from dwc26.models.base import TrialFailed
from dwc26.models.dmd import (
    DMDProxy,
    clean_eigs,
    energy_rank,
    exact_dmd_eigs,
    qr_basis,
    real_basis,
    varpro_real_eigs,
)
from our_work.proxy_model_sijie.tests.synth_models import (
    EXP_SCALE,
    SLOW_PAIR,
    exp_true_eigs,
    fit_settings,
    make_curves,
    make_exponentials,
    make_growth,
    make_real_exponentials,
    make_slow_pair,
    unit_X,
)

DMD_CFG = {"unstable": "clip", "stab_tol": 1.0e-9, "imag_tol": 1.0e-10, "sigma_rel_tol": 1.0e-10,
           "dup_tol": 1.0e-8, "r_max": None}


def _settings(tmp_path, **kw):
    s = fit_settings("dmd", tmp_path, rank=kw.pop("rank", 5), gp_target=kw.pop("gp_target", "coef"),
                     kernel=kw.pop("kernel", "matern25"))
    for k, v in kw.items():
        if k in s["dmd"]:
            s["dmd"][k] = v
        elif k in s["gp"]:
            s["gp"][k] = v
        else:
            s[k] = v
    return s


def _sort(lam):
    lam = np.asarray(lam, dtype=complex)
    return lam[np.lexsort((lam.imag, lam.real))]


# --- M18: sum of exponentials ------------------------------------------------------

@pytest.fixture(scope="module")
def exp_fit(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m18")
    X, Xv = unit_X(30, 21), unit_X(8, 22)
    T = 25
    s = _settings(tmp, basis="exact", rank=5, n_restarts_optimizer=5)
    m = DMDProxy().fit(X, make_exponentials(X, T), s)
    return m, X, Xv, T


def test_m18_eigenvalues(exp_fit):
    m, X, Xv, T = exp_fit
    d = m.diagnostics
    assert d["q"] == 5 and d["r"] == 5 and d["r_eff"] == 5
    assert d["n_stabilised"] == 0
    full = np.concatenate([m.reps, m.reps[m.kinds == 1].conj()])
    np.testing.assert_allclose(_sort(full), _sort(exp_true_eigs()), rtol=0, atol=1e-8)
    used = np.array([complex(a, b) for a, b in d["lam_used"]])
    np.testing.assert_array_equal(used, m.reps)


def test_m18_reconstruction_and_continuation(exp_fit):
    m, X, Xv, T = exp_fit
    Y = make_exponentials(X, T)
    rec = m.rollout_from_coef(m.coef_train, T)
    assert np.abs(rec - Y).max() < 1e-9
    long = m.rollout_from_coef(m.coef_train, T + 80)
    truth = make_exponentials(X, T + 80)
    np.testing.assert_allclose(long, truth, rtol=1e-8, atol=0)


def test_m18_end_to_end(exp_fit):
    m, X, Xv, T = exp_fit
    P = m.predict(Xv)
    assert (nrmse(make_exponentials(Xv, T), P) < 0.02).all()
    F = m.forecast(Xv, 80)
    truth = make_exponentials(Xv, T + 80)[:, T:]
    assert (nrmse(truth, F) < 0.05).all()


# --- M19: energy rank ---------------------------------------------------------------

def test_m19_energy_rank():
    s = np.array([3.0, 2.0, 1.0, 0.1])
    assert energy_rank(s, 0.99, 1e-10) == (3, 4)
    assert energy_rank(s, 0.9995, 1e-10) == (4, 4)
    assert energy_rank(s, 1.0, 1e-10) == (4, 4)
    assert energy_rank(s, 1.0, 1e-10, r_max=2) == (2, 4)
    assert energy_rank(np.array([1.0, 1e-12]), 0.99, 1e-10) == (1, 1)
    assert energy_rank(np.array([1.0, 1e-12]), 1.0, 1e-10) == (1, 1)


# --- M20: real basis -----------------------------------------------------------------

def test_m20_real_basis_equivalence():
    rng = np.random.default_rng(5)
    real = rng.uniform(-0.9, 1.0, 2)
    pairs = rng.uniform(0.5, 1.0, 2) * np.exp(1j * rng.uniform(0.1, 3.0, 2))
    c_real = rng.normal(size=2)
    c_pair = rng.normal(size=2) + 1j * rng.normal(size=2)
    lam_all = np.concatenate([real, pairs, pairs.conj()])
    c_all = np.concatenate([c_real, c_pair, c_pair.conj()])
    reps = np.concatenate([real.astype(complex), pairs])
    kinds = np.array([0, 0, 1, 1], dtype=np.int8)
    a = np.concatenate([c_real, [2 * c_pair[0].real, -2 * c_pair[0].imag, 2 * c_pair[1].real, -2 * c_pair[1].imag]])
    tau = np.arange(51)
    B = real_basis(reps, kinds, tau)
    assert B.shape == (51, 6) and B.dtype == np.float64
    lhs = np.real((c_all[None, :] * lam_all[None, :] ** tau[:, None]).sum(axis=1))
    np.testing.assert_allclose(B @ a, lhs, rtol=0, atol=1e-12)
    assert real_basis(np.array([0.0 + 0j]), np.array([0], dtype=np.int8), np.arange(3)).ravel().tolist() == [1, 0, 0]


# --- M21: eigenvalue handling ---------------------------------------------------------

def test_m21_clip(tmp_path):
    X = unit_X(20, 31)
    T = 15
    m = DMDProxy().fit(X, make_growth(X, T), _settings(tmp_path, basis="exact", rank=3))
    d = m.diagnostics
    assert d["q"] == 3 and d["n_stabilised"] == 1 and d["n_merged"] == 1 and d["r_eff"] == 2
    assert d["max_abs_lam_raw"] == pytest.approx(1.03, abs=1e-8)
    assert (np.abs(m.reps) <= 1 + 1e-12).all()
    assert np.isfinite(m.rollout(unit_X(4, 32), T + 80)).all()
    assert m.reps.dtype == np.complex128 and m.kinds.dtype == np.int8
    assert d["r_eff"] == d["n_real"] + 2 * d["n_pair"]


def test_m21_keep(tmp_path):
    X = unit_X(20, 31)
    T = 15
    m = DMDProxy().fit(X, make_growth(X, T), _settings(tmp_path, basis="exact", rank=3, unstable="keep"))
    long = m.rollout_from_coef(m.coef_train, T + 80)
    np.testing.assert_allclose(long, make_growth(X, T + 80), rtol=1e-6, atol=0)
    assert (long[:, -1, :] / long[:, T - 1, :] > 2).all()
    B = real_basis(m.reps, m.kinds, np.arange(5))
    assert B.dtype == np.float64 and B.shape[1] == m.diagnostics["r_eff"]
    assert m.diagnostics["r_eff"] == m.diagnostics["n_real"] + 2 * m.diagnostics["n_pair"]


def test_clean_eigs_rules():
    lam = np.array([1.05, 0.5 + 1e-12j, 0.9 * np.exp(0.3j), 0.9 * np.exp(-0.3j), 1.2 * np.exp(0.5j),
                    1.2 * np.exp(-0.5j), -1.1])
    reps, kinds, info = clean_eigs(lam, DMD_CFG)
    assert info["n_stabilised"] == 4 and info["max_abs_lam_raw"] == pytest.approx(1.2)
    assert (np.abs(reps) <= 1 + 1e-12).all()
    assert list(np.abs(reps)) == sorted(np.abs(reps), reverse=True)
    assert kinds.tolist().count(1) == 2 and (reps[kinds == 1].imag > 0).all()
    assert (reps[kinds == 0].imag == 0).all()
    with pytest.raises(Exception):
        clean_eigs(np.array([0.5 + 0.3j]), DMD_CFG)                    # no conjugate partner
    keep = dict(DMD_CFG, unstable="keep")
    reps_k, _, info_k = clean_eigs(lam, keep)
    assert info_k["n_stabilised"] == 0 and np.abs(reps_k).max() == pytest.approx(1.2)


def test_exact_dmd_eigs_linear_system():
    rng = np.random.default_rng(0)
    lam_true = np.array([0.95, 0.8, 0.5])
    x0 = rng.normal(size=(6, 3))
    k = np.arange(10)
    traj = x0[:, None, :] * lam_true[None, None, :] ** k[None, :, None]     # (6 trajectories, 10 steps, 3)
    D = traj.transpose(0, 2, 1).reshape(18, 10)
    lam = exact_dmd_eigs(D, 3)
    assert lam.dtype == np.complex128 and lam.shape == (3,)
    np.testing.assert_allclose(np.sort(lam.real), [0.5, 0.8, 0.95], atol=1e-10)
    np.testing.assert_allclose(lam.imag, 0.0, atol=1e-10)


# --- M22: proj target -------------------------------------------------------------------

def test_m22_proj(tmp_path):
    X = unit_X(30, 21)
    T = 25
    Y = make_exponentials(X, T)
    m = DMDProxy().fit(X, Y, _settings(tmp_path, basis="exact", rank=5, gp_target="proj"))
    Q, R = qr_basis(m.B_fit)
    Ymat = Y.transpose(1, 0, 2).reshape(T, -1)
    coef = np.linalg.solve(R, Q.T @ Ymat).T.reshape(30, 3, -1)
    np.testing.assert_allclose(coef, m.coef_train, rtol=1e-10, atol=1e-10 * np.abs(m.coef_train).max())
    np.testing.assert_allclose(m.targets_train, (Q.T @ Ymat).T.reshape(30, 3, -1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.basis_map @ R, np.eye(R.shape[0]), atol=1e-10)
    b = np.arange(1.0, 6.0)
    with pytest.raises(TrialFailed):
        qr_basis(np.column_stack([b, b, np.ones(5)]))


# --- M23: predict_std --------------------------------------------------------------------

@pytest.mark.parametrize("basis", ["exact", "real"])
@pytest.mark.parametrize("target", ["coef", "proj"])
@pytest.mark.parametrize("with_resid", [True, False])
def test_m23_predict_std(tmp_path, basis, target, with_resid):
    X, Xv = unit_X(20, 41), unit_X(3, 42)
    T, k0 = 13, 2
    m = DMDProxy().fit(X, make_curves(X, T, k0),
                       _settings(tmp_path, basis=basis, rank=3, gp_target=target,
                                 std_includes_basis_residual=with_resid))
    S = m.predict_std(Xv)
    assert S.shape == (3, T, 3) and np.all(S[:, :k0] == 0.0) and (S >= 0).all()
    x = Xv[1:2]
    P = real_basis(m.reps, m.kinds, np.arange(T - k0)) @ m.basis_map
    for f in range(3):
        sig = np.array([m.gps[f][j].predict(x, return_std=True)[1][0] for j in range(P.shape[1])])
        s2 = (P ** 2 * sig[None, :] ** 2).sum(axis=1)
        if with_resid:
            s2 = s2 + m.resid_var[:, f]
        np.testing.assert_allclose(S[1, k0:, f], np.sqrt(s2), rtol=1e-12, atol=1e-12)


def test_dmd_scale_of_fluids(tmp_path):
    X = unit_X(30, 21)
    m = DMDProxy().fit(X, make_exponentials(X, 25), _settings(tmp_path, basis="exact", rank=5))
    assert np.allclose(m.coef_train[:, 1, :], m.coef_train[:, 0, :] * EXP_SCALE[1], atol=1e-8)


# --- M42 to M44: amendment A2, real eigenvalues by variable projection ----------------------

DMD_REAL = {"basis": "real", "real_kappa_max": 3.0, "real_kappa_sep": 0.005, "energy_report": 0.99,
            "unstable": "clip", "stab_tol": 1.0e-9, "imag_tol": 1.0e-10, "sigma_rel_tol": 1.0e-10,
            "dup_tol": 1.0e-8, "r_max": None}


def test_m42_real_exponentials(tmp_path):
    X = unit_X(30, 71)
    T = 25
    Y = make_real_exponentials(X, T)
    m = DMDProxy().fit(X, Y, _settings(tmp_path, basis="real", rank=3))
    d = m.diagnostics
    lam = m.reps.real
    np.testing.assert_allclose(lam, [1.0, 0.9, 0.7], rtol=0, atol=1e-6)
    assert (m.reps.imag == 0).all() and (m.kinds == 0).all()
    assert d["n_merged"] == 0 and d["n_stabilised"] == 0 and d["basis"] == "real" and d["r_eff"] == 3
    assert np.abs(m.rollout_from_coef(m.coef_train, T) - Y).max() < 1e-6
    np.testing.assert_allclose(m.rollout_from_coef(m.coef_train, T + 80), make_real_exponentials(X, T + 80),
                               rtol=1e-5, atol=0)
    vp = d["varpro"]
    assert len(vp["start_costs"]) == 4 and vp["cost"] == min(vp["start_costs"])
    assert vp["best_start"] == int(np.argmin(vp["start_costs"]))


def _check_real_constraints(m, kmax=3.0, sep=0.005):
    lam = m.reps
    assert (lam.imag == 0).all() and (m.kinds == 0).all()
    lam = lam.real
    assert lam[0] <= 1.0 and (lam > 0).all() and (np.diff(lam) < 0).all()
    ratio = lam[1:] / lam[:-1]
    assert (ratio >= np.exp(-kmax) - 1e-12).all() and (ratio <= np.exp(-sep) + 1e-12).all()
    d = m.diagnostics
    assert d["r_eff"] == len(lam) == d["r"]
    assert len(d["lam_exact"]) == d["r"]


@pytest.mark.parametrize("case", ["exponentials", "curves"])
def test_m43_constraints_and_determinism(tmp_path, case):
    if case == "exponentials":
        X, Xv, T, rank = unit_X(30, 21), unit_X(8, 22), 25, 5
        Y = make_exponentials(X, T)
    else:
        X, Xv, T, rank = unit_X(12, 3), unit_X(4, 103), 13, 3
        Y = make_curves(X, T, 2)
    s = _settings(tmp_path, basis="real", rank=rank, gp_target="proj")
    a = DMDProxy().fit(X, Y, s)
    b = DMDProxy().fit(X, Y, s)
    _check_real_constraints(a)
    assert np.array_equal(a.reps, b.reps)
    assert np.array_equal(a.predict(Xv), b.predict(Xv))
    assert np.array_equal(a.forecast(Xv, 80), b.forecast(Xv, 80))


def test_varpro_real_eigs_direct():
    X = unit_X(30, 71)
    Y = make_real_exponentials(X, 25)
    D = (np.sqrt([0.25, 0.25, 0.5])[:, None, None] * Y.transpose(2, 0, 1)).reshape(90, 25)
    lam, info = varpro_real_eigs(D, 3, DMD_REAL)
    assert lam.dtype == np.float64 and lam.shape == (3,)
    np.testing.assert_allclose(lam, [1.0, 0.9, 0.7], atol=1e-6)
    assert info["lam_exact"].dtype == np.complex128 and len(info["lam_exact"]) == 3
    assert set(info["varpro"]) >= {"cost", "start_costs", "best_start", "nfev", "status", "d_at_bound"}
    lam1, _ = varpro_real_eigs(D, 1, DMD_REAL)                       # m = 1: one start value per start
    assert lam1.shape == (1,) and 0 < lam1[0] <= 1


def _sign_changes(y: np.ndarray, tol: float) -> int:
    y = y[np.abs(y) > tol]
    return int(np.count_nonzero(np.diff(np.sign(y)) != 0))


def test_m44_oscillation_removed(tmp_path, capsys):
    X = unit_X(20, 81)
    T, n_tot = 37, 37 + 80
    Y = make_slow_pair(X, T)
    rho, theta = SLOW_PAIR
    ex = DMDProxy().fit(X, Y, _settings(tmp_path, basis="exact", rank=2))
    lam = np.sort_complex(np.concatenate([ex.reps, ex.reps[ex.kinds == 1].conj()]))
    np.testing.assert_allclose(lam, np.sort_complex(rho * np.exp(np.array([-1j, 1j]) * theta)), rtol=0, atol=1e-8)
    long_ex = ex.rollout_from_coef(ex.coef_train, n_tot)
    assert (long_ex.min(axis=(1, 2)) < 0).all()                        # reproduces the review-1 failure
    re = DMDProxy().fit(X, Y, _settings(tmp_path, basis="real", rank=2))
    _check_real_constraints(re)
    long_re = re.rollout_from_coef(re.coef_train, n_tot)
    tol = 1e-12 * np.abs(long_re).max()
    for i in range(len(X)):
        for f in range(3):
            y = long_re[i, :, f]
            assert _sign_changes(y, tol) <= 1, (i, f)
            assert _sign_changes(np.diff(y), tol) <= 1, (i, f)     # at most one local extremum
    n_neg = int((long_re[:, -1, :] < 0).sum())
    with capsys.disabled():
        print(f"\nM44: real basis, curves negative at tau = {n_tot - 1}: {n_neg} of {long_re.shape[0] * 3}")
