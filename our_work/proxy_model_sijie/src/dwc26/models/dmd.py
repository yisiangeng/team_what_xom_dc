"""DMD proxy, strategy S4 (spec task2 sections 2.9 and 2.10; PRD section 6; D-013, D-016).

1. Snapshot matrix D (3 n_fit rows, T_p = T_fit - k0 columns), rows scaled by sqrt(w_f).
2. Pairs X1 = D[:, :-1], X2 = D[:, 1:]; one step = one quarter.
3. Rank r = min(rank, q, r_max), the tuned rank (A2, D-019); the 99%-energy rank
   r_99 (PRD section 6) is reported only.
4. basis real (A2, section 2.9a; default): real eigenvalues in (0, 1] by variable
   projection, started from exact DMD (S0) and three log-spaced starts.
   basis exact: exact DMD eigenvalues (Tu et al. 2014, Algorithm 2), then
5. cleaning: real test, |lambda| > 1 moved to the unit circle (unstable: clip),
   representatives, ordered by |lambda| then |arg|, duplicates merged.
6. Real basis: lambda^tau, rho^tau cos(tau theta), rho^tau sin(tau theta).
7. Least-squares coefficients of every training curve.
8. GP target: the coefficients (coef) or Q^T y (proj).
9. One GP for each fluid and target column (Article 4, p. 1106, Eq. 17).
10. y(x, k) = b(k - k0)^T M g(x); the same formula continues after T_fit.
11. predict_std^2 = sum_j ((b^T M)_j sigma_j(x))^2 + the basis residual variance.
"""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.models.base import ModelBase, TrialFailed

PARTNER_TOL = 1e-8          # conjugate partner tolerance, relative to max(1, |lambda|) (section 2.9 step 5)
R_RATIO_MIN = 1e-12         # proj: min |R_jj| / max |R_jj| (section 2.9 step 8)
NU = {"matern25": 2.5, "matern15": 1.5}


def energy_rank(sigma: np.ndarray, energy: float, sigma_rel_tol: float,
                r_max: int | None = None) -> tuple[int, int]:
    """(r, q): q = #{sigma_i > tol * sigma_1}; r = smallest rho <= q whose share of
    sum sigma^2 (over i <= q) is >= energy; energy = 1 gives r = q; r <= r_max."""
    s = np.asarray(sigma, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all() or s[0] <= 0:
        raise TrialFailed(f"energy_rank: no positive singular value ({s[:3].tolist()})")
    q = int(np.count_nonzero(s > sigma_rel_tol * s[0]))
    s2 = s[:q] ** 2
    cum = np.cumsum(s2) / s2.sum()
    cum[-1] = 1.0
    hit = np.nonzero(cum >= energy)[0]
    r = int(hit[0]) + 1 if hit.size else q
    if r_max is not None:
        r = min(r, int(r_max))
    return r, q


def exact_dmd_eigs(D: np.ndarray, r: int) -> np.ndarray:
    """(r,) complex128 eigenvalues of A~ = U_r^T X2 V_r Sigma_r^-1."""
    X1, X2 = D[:, :-1], D[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    Ur, sr, Vr = U[:, :r], s[:r], Vt[:r].T
    A = (Ur.T @ X2 @ Vr) / sr[None, :]
    return np.linalg.eigvals(A).astype(np.complex128)


def clean_eigs(lam: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Step 5. cfg = the `dmd` block (unstable, stab_tol, imag_tol, dup_tol).
    Returns (reps (m,) complex128, kinds (m,) int8 [0 real, 1 pair], info)."""
    lam_raw = np.asarray(lam, dtype=np.complex128).copy()
    lam_c = lam_raw.copy()
    mod = np.abs(lam_c)
    real = np.abs(lam_c.imag) <= cfg["imag_tol"] * np.maximum(1.0, mod)       # 1. real or complex
    lam_c[real] = lam_c[real].real
    mod = np.abs(lam_c)
    unstable = mod > 1.0 + cfg["stab_tol"]
    n_stab = 0
    if cfg["unstable"] == "clip":                                               # 2. stabilise
        lam_c[unstable] = lam_c[unstable] / mod[unstable]
        lam_c[real] = lam_c[real].real
        n_stab = int(unstable.sum())
    reps, kinds = [], []                                                        # 3. representatives
    for i, z in enumerate(lam_c):
        if real[i]:
            reps.append(complex(z.real, 0.0))
            kinds.append(0)
        elif z.imag > 0:
            others = np.delete(lam_c, i)
            others = others[others.imag < 0]
            if not (np.abs(np.conj(z) - others) <= PARTNER_TOL * max(1.0, abs(z))).any():
                raise ContractError(f"clean_eigs: complex eigenvalue {z} has no conjugate partner")
            reps.append(complex(z))
            kinds.append(1)
    reps = np.array(reps, dtype=np.complex128)
    kinds = np.array(kinds, dtype=np.int8)
    order = np.lexsort((np.abs(np.angle(reps)), -np.abs(reps)))                  # 5. order
    reps, kinds = reps[order], kinds[order]
    keep = []                                                                   # 4. merge duplicates
    for i in range(len(reps)):
        if any(abs(reps[i] - reps[j]) <= cfg["dup_tol"] for j in keep):
            continue
        keep.append(i)
    n_merged = len(reps) - len(keep)
    reps, kinds = reps[keep], kinds[keep]
    info = {
        "lam_raw": lam_raw, "lam_clean": lam_c, "unstable": cfg["unstable"],
        "n_unstable_raw": int(unstable.sum()), "n_stabilised": n_stab,
        "max_abs_lam_raw": float(np.abs(lam_raw).max()) if lam_raw.size else 0.0,
        "n_merged": int(n_merged), "n_real": int((kinds == 0).sum()), "n_pair": int((kinds == 1).sum()),
    }
    return reps, kinds, info


def real_basis(reps: np.ndarray, kinds: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """(len(tau), r_eff): lambda^tau for real lambda; rho^tau cos(tau theta) and
    rho^tau sin(tau theta) for each pair. 0^0 = 1; tau are integer steps."""
    tau = np.asarray(tau)
    tf = tau.astype(np.float64)
    cols = []
    for z, kind in zip(np.asarray(reps, dtype=np.complex128), kinds):
        if kind == 0:
            cols.append(np.power(z.real, tf))
        else:
            rt = np.power(abs(z), tf)
            th = float(np.angle(z))
            cols.append(rt * np.cos(tf * th))
            cols.append(rt * np.sin(tf * th))
    if not cols:
        return np.zeros((len(tf), 0))
    return np.column_stack(cols)


VARPRO_STARTS = ((1e-3, 0.3), (1e-2, 1.0), (3e-3, 3.0))     # S1 to S3: log-spaced decay rates (section 2.9a)
COND_B_MAX = 1e12                                            # real basis, coef target: singular basis (section 2.9a)


def _real_vandermonde(lam: np.ndarray, Tp: int) -> np.ndarray:
    """(Tp, m): B[tau, j] = lam_j^tau, tau = 0..Tp-1 (0^0 = 1)."""
    return np.power(lam[None, :], np.arange(Tp, dtype=np.float64)[:, None])


def varpro_real_eigs(D: np.ndarray, m: int, cfg: dict) -> tuple[np.ndarray, dict]:
    """A2, section 2.9a: optimized DMD with real eigenvalues 1 >= lam_1 > ... > lam_m > 0,
    fitted to all snapshots Z = D^T by variable projection. lam_j = exp(-(d_1 + ... + d_j)),
    d_1 in [0, kappa_max], d_j in [kappa_sep, kappa_max] for j >= 2. cfg = the `dmd` block.
    Returns (lam (m,) float64 descending, info with lam_exact and varpro)."""
    from scipy.optimize import least_squares

    kmax, sep = float(cfg["real_kappa_max"]), float(cfg["real_kappa_sep"])
    Z = np.asarray(D, dtype=np.float64).T                                       # (Tp, 3 n_fit)
    Tp = Z.shape[0]
    lo = np.array([0.0] + [sep] * (m - 1))
    hi = np.full(m, kmax)

    def fun(d: np.ndarray) -> np.ndarray:
        B = _real_vandermonde(np.exp(-np.cumsum(d)), Tp)
        C = np.linalg.lstsq(B, Z, rcond=None)[0]
        return (B @ C - Z).ravel()

    lam_exact = exact_dmd_eigs(D, m)                                            # S0 (PRD section 6 step 2)
    starts = [np.sort(-np.log(np.clip(np.abs(lam_exact), np.exp(-kmax), 1.0)))]
    for a, b in VARPRO_STARTS:
        starts.append(np.array([a]) if m == 1 else a * (b / a) ** (np.arange(m) / (m - 1)))
    results, costs = [], []
    for kap in starts:
        kap = np.asarray(kap, dtype=np.float64).copy()
        kap[0] = max(kap[0], 0.0)
        for j in range(1, m):
            kap[j] = max(kap[j], kap[j - 1] + sep)
        d0 = np.clip(np.concatenate([kap[:1], np.diff(kap)]), lo, hi)
        try:
            res = least_squares(fun, d0, bounds=(lo, hi), method="trf", jac="2-point", x_scale="jac",
                                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
        except Exception as e:  # noqa: BLE001 - any optimiser failure fails the trial, not the run
            raise TrialFailed(f"varpro: least_squares failed: {type(e).__name__}: {e}") from e
        ok = np.isfinite(res.cost) and np.isfinite(res.x).all()
        results.append(res)
        costs.append(float(res.cost) if ok else math.inf)
    if not np.isfinite(costs).any():
        raise TrialFailed("varpro: non-finite cost in every start")
    best = int(np.argmin(costs))                                                # first start wins a tie
    res = results[best]
    lam = np.exp(-np.cumsum(res.x))
    if not (np.isfinite(lam).all() and (lam > 0).all() and (lam <= 1).all()):
        raise TrialFailed(f"varpro: eigenvalues outside (0, 1]: {lam}")
    at_bound = ["lo" if abs(x - l) <= 1e-12 else "hi" if abs(x - h) <= 1e-12 else ""
                for x, l, h in zip(res.x, lo, hi)]
    info = {"lam_exact": lam_exact,
            "varpro": {"cost": costs[best], "start_costs": costs, "best_start": best, "nfev": int(res.nfev),
                       "status": int(res.status), "d": res.x.tolist(), "d_at_bound": at_bound,
                       "nfev_per_start": [int(r.nfev) for r in results],
                       "status_per_start": [int(r.status) for r in results]}}
    return lam, info


def qr_basis(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduced QR of the basis. TrialFailed if min |R_jj| / max |R_jj| < 1e-12."""
    Q, R = np.linalg.qr(B, mode="reduced")
    d = np.abs(np.diag(R))
    if d.size == 0 or d.max() == 0 or d.min() / d.max() < R_RATIO_MIN:
        raise TrialFailed(f"proj: the basis is singular (min |R_jj| / max |R_jj| = "
                          f"{(d.min() / d.max()) if d.size and d.max() > 0 else 0.0:.3g})")
    return Q, R


def make_gp(cfg: dict, p: int):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

    g = cfg["gp"]
    ls_bounds = tuple(g["length_scale_bounds"])
    if cfg["kernel"] == "rbf":
        base = RBF(length_scale=np.ones(p), length_scale_bounds=ls_bounds)
    else:
        base = Matern(length_scale=np.ones(p), length_scale_bounds=ls_bounds, nu=NU[cfg["kernel"]])
    kernel = (ConstantKernel(1.0, tuple(g["constant_bounds"])) * base
              + WhiteKernel(g["noise_init"], tuple(g["noise_bounds"])))
    return GaussianProcessRegressor(kernel=kernel, alpha=g["alpha"], normalize_y=g["normalize_y"],
                                    n_restarts_optimizer=g["n_restarts_optimizer"], random_state=cfg["seed"])


def _at_bound(v: float, bounds) -> bool:
    lo, hi = bounds
    return bool(v <= lo * 1.01 or v >= hi / 1.01)


def gp_hyper(gp, cfg: dict, f: int, j: int) -> dict:
    """Fitted hyperparameters of one GP, with the at-bound flags."""
    g = cfg["gp"]
    k = gp.kernel_
    c = float(k.k1.k1.constant_value)
    ls = np.atleast_1d(k.k1.k2.length_scale).astype(float).tolist()
    noise = float(k.k2.noise_level)
    at = []
    if _at_bound(c, g["constant_bounds"]):
        at.append("constant")
    at += [f"length_scale[{d}]" for d, v in enumerate(ls) if _at_bound(v, g["length_scale_bounds"])]
    if _at_bound(noise, g["noise_bounds"]):
        at.append("noise")
    return {"fluid": FLUIDS[f], "j": j, "constant": c, "length_scale": ls, "noise": noise,
            "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_), "at_bound": at}


def _cplx(a) -> list:
    return [[float(z.real), float(z.imag)] for z in np.asarray(a, dtype=np.complex128)]


class DMDProxy(ModelBase):
    """Shared dynamics (S4): DMD time basis and GPs from X to the curve coefficients."""

    name = "dmd"
    extrapolates_time = True

    def _fit(self, X, Y, cfg, val):
        from sklearn.exceptions import ConvergenceWarning

        k0, T_fit, n, p = self.k0, self.T_fit, X.shape[0], X.shape[1]
        Tp = T_fit - k0
        dcfg = cfg["dmd"]
        w = np.asarray(cfg["fluid_weights"], dtype=np.float64)
        Yp = Y[:, k0:, :]                                                       # (n, Tp, 3)
        D = (np.sqrt(w)[:, None, None] * Yp.transpose(2, 0, 1)).reshape(len(FLUIDS) * n, Tp)   # 1.
        sigma = np.linalg.svd(D[:, :-1], compute_uv=False)                       # 2., 3.
        r_99, q = energy_rank(sigma, dcfg["energy_report"], dcfg["sigma_rel_tol"])   # PRD section 6, report only
        r = min(int(cfg["rank"]), q, dcfg["r_max"] if dcfg["r_max"] is not None else q)   # A2: tuned rank
        basis = dcfg["basis"]
        varpro = None
        if basis == "real":                                                     # A2, section 2.9a
            lam, vinfo = varpro_real_eigs(D, r, dcfg)
            reps, kinds = lam.astype(np.complex128), np.zeros(len(lam), dtype=np.int8)
            varpro = vinfo["varpro"]
            info = {"lam_raw": vinfo["lam_exact"], "lam_clean": reps.copy(), "unstable": None,
                    "n_unstable_raw": int((np.abs(vinfo["lam_exact"]) > 1.0 + dcfg["stab_tol"]).sum()),
                    "n_stabilised": 0, "n_merged": 0,
                    "max_abs_lam_raw": float(np.abs(vinfo["lam_exact"]).max()),
                    "n_real": len(lam), "n_pair": 0}
        elif basis == "exact":                                                  # steps 4 and 5
            lam = exact_dmd_eigs(D, r)
            reps, kinds, info = clean_eigs(lam, dcfg)
        else:
            raise ContractError(f"dmd: unknown basis {basis!r}")
        tau = np.arange(Tp)
        B = real_basis(reps, kinds, tau)                                        # 6.
        r_eff = B.shape[1]
        if r_eff == 0 or not np.isfinite(B).all():
            raise TrialFailed("dmd: empty or non-finite basis")
        Ymat = Yp.transpose(1, 0, 2).reshape(Tp, n * len(FLUIDS))               # column i*3 + f
        coef, _, rank_B, _ = np.linalg.lstsq(B, Ymat, rcond=None)              # 7.
        A = coef.T.reshape(n, len(FLUIDS), r_eff)
        resid = (B @ coef - Ymat).reshape(Tp, n, len(FLUIDS))
        resid_var = (resid ** 2).mean(axis=1)                                   # (Tp, 3)
        cond_B = float(np.linalg.cond(B))
        if basis == "real" and cfg["gp_target"] == "coef" and not cond_B <= COND_B_MAX:
            raise TrialFailed(f"dmd: the real basis is numerically singular (cond(B) = {cond_B:.3g})")
        if cfg["gp_target"] == "coef":                                          # 8.
            G, M, R = A, np.eye(r_eff), np.zeros((0, 0))
        elif cfg["gp_target"] == "proj":
            Q, R = qr_basis(B)
            G = (Q.T @ Ymat).T.reshape(n, len(FLUIDS), r_eff)
            M = solve_triangular(R, np.eye(r_eff))
        else:
            raise ContractError(f"dmd: unknown gp_target {cfg['gp_target']!r}")
        gps, hyper, n_warn = [], [], 0                                           # 9.
        for f in range(len(FLUIDS)):
            row = []
            for j in range(r_eff):
                gp = make_gp(cfg, p)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(X, G[:, f, j])
                n_warn += sum(issubclass(c.category, ConvergenceWarning) for c in caught)
                row.append(gp)
                hyper.append(gp_hyper(gp, cfg, f, j))
            gps.append(row)
        self.lam_raw, self.reps, self.kinds = info["lam_raw"], reps, kinds
        self.B_fit, self.R, self.basis_map = B, R, M
        self.coef_train, self.targets_train, self.resid_var = A, G, resid_var
        self.sigma, self.X_fit, self.gps = sigma[:q].copy(), X.copy(), gps
        s2 = sigma[:q] ** 2
        self.diagnostics = {
            "q": q, "r": r, "r_eff": r_eff, "n_real": info["n_real"], "n_pair": info["n_pair"],
            "n_stabilised": info["n_stabilised"], "n_unstable_raw": info["n_unstable_raw"],
            "n_merged": info["n_merged"], "max_abs_lam_raw": info["max_abs_lam_raw"], "cond_B": cond_B,
            "rank_B": int(rank_B), "unstable": dcfg["unstable"] if basis == "exact" else None,
            "basis": basis, "rank": int(cfg["rank"]), "r_99": int(r_99), "energy_report": dcfg["energy_report"],
            "kernel": cfg["kernel"], "gp_target": cfg["gp_target"],
            "fluid_weights_name": cfg.get("fluid_weights_name"),
            "sigma": sigma.tolist(), "energy_cum": (np.cumsum(s2) / s2.sum()).tolist(),
            "lam_raw": _cplx(info["lam_raw"]), "lam_clean": _cplx(info["lam_clean"]),
            "lam_exact": _cplx(info["lam_raw"]), "lam_used": _cplx(reps), "kinds": kinds.tolist(),
            "varpro": varpro, "plausible_selected": None,
            "gp_hyper": hyper, "n_convergence_warnings": int(n_warn),
        }

    # --- prediction -------------------------------------------------------------------
    def basis_rows(self, n_total: int) -> np.ndarray:
        """(n_total - k0, r_eff): rows b(tau)^T M for tau = 0..n_total-k0-1."""
        return real_basis(self.reps, self.kinds, np.arange(n_total - self.k0)) @ self.basis_map

    def _gp(self, X: np.ndarray, std: bool) -> tuple[np.ndarray, np.ndarray | None]:
        F, r_eff = len(FLUIDS), self.basis_map.shape[0]
        mean = np.empty((X.shape[0], F, r_eff))
        sd = np.empty((X.shape[0], F, r_eff)) if std else None
        for f in range(F):
            for j in range(r_eff):
                if std:
                    mean[:, f, j], sd[:, f, j] = self.gps[f][j].predict(X, return_std=True)
                else:
                    mean[:, f, j] = self.gps[f][j].predict(X)
        return mean, sd

    def _raw_rollout(self, X, n_total):
        P = self.basis_rows(n_total)                                            # (L, r_eff)
        G, _ = self._gp(X, std=False)                                           # (n, 3, r_eff)
        return np.ascontiguousarray((G @ P.T).transpose(0, 2, 1))

    def _raw_std(self, X):
        P = self.basis_rows(self.T_fit)
        _, sd = self._gp(X, std=True)
        s2 = (sd ** 2) @ (P ** 2).T                                             # (n, 3, Tp)
        if self.settings["gp"]["std_includes_basis_residual"]:
            s2 = s2 + self.resid_var.T[None, :, :]
        return np.ascontiguousarray(np.sqrt(s2).transpose(0, 2, 1))

    def rollout_from_coef(self, A: np.ndarray, n_total: int) -> np.ndarray:
        """(n, 3, r_eff) least-squares coefficients -> (n, n_total, 3) raw, unclipped."""
        A = np.asarray(A, dtype=np.float64)
        out = np.zeros((A.shape[0], n_total, len(FLUIDS)))
        if n_total > self.k0:
            Bt = real_basis(self.reps, self.kinds, np.arange(n_total - self.k0))
            out[:, self.k0:, :] = (A @ Bt.T).transpose(0, 2, 1)
        return out

    # --- save and load -----------------------------------------------------------------
    def _meta_extra(self) -> dict:
        return {"diagnostics": self.diagnostics}

    def _save_artefacts(self, path: Path) -> list[str]:
        with open(path / "dmd.npz", "wb") as fh:
            np.savez(fh, lam_raw=self.lam_raw, reps=self.reps, kinds=self.kinds, B_fit=self.B_fit, R=self.R,
                     coef_train=self.coef_train, resid_var=self.resid_var, sigma=self.sigma, X_fit=self.X_fit)
        (path / "gps.pkl").write_bytes(pickle.dumps(self.gps, protocol=5))
        return ["dmd.npz", "gps.pkl"]

    def _load_artefacts(self, path: Path, meta: dict) -> None:
        with np.load(path / "dmd.npz", allow_pickle=False) as z:
            for k in ("lam_raw", "reps", "kinds", "B_fit", "R", "coef_train", "resid_var", "sigma", "X_fit"):
                setattr(self, k, z[k])
        r_eff = self.B_fit.shape[1]
        if self.settings["gp_target"] == "proj":
            self.basis_map = solve_triangular(self.R, np.eye(r_eff))
        else:
            self.basis_map = np.eye(r_eff)
        # Trusted local artefact of this repository; read only after the sha256 check (spec 1.5).
        self.gps = pickle.loads((path / "gps.pkl").read_bytes())  # noqa: S301
        self.diagnostics = meta.get("diagnostics", {})
        self.targets_train = None
