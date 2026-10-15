"""Synthetic data and configs for the Task 2 tests (spec task2 section 4.0).

- make_curves: plateau-then-decline oil, gas = 0.99 oil, delayed water; 0 before k0.
- make_exponentials: a sum of exponentials with known eigenvalues (k0 = 0).
- make_growth: the M21 data with one growing mode (1.03).
- synthetic_task1_db: a Task 1 database written with plain INSERTs.
- tiny_cfg: a valid model config for that database, all paths in tmp_path.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

FLUIDS = ("oil", "gas", "water")
Y_PHYS_SCALE = (15000.0, 5450.0, 9000.0)
N_CASES, T_DB, K0_DB = 30, 13, 2
TRAIN_IDS, VAL_IDS, TEST_IDS = range(1, 19), range(19, 25), range(25, 31)
X_LO = np.array([0.05, 0.8, 0.5, 50.0])
X_HI = np.array([0.15, 1.5, 10.0, 200.0])
X_NAMES = ("fault_trans", "poro_mult", "perm_mult", "aquifer_pv")
EXP_LAM = (0.9, 0.7, 0.95, 0.4)          # lambda_1, lambda_2, rho, theta
EXP_SCALE = np.array([1.0, 0.98, 0.6])


def make_curves(X: np.ndarray, T: int, k0: int) -> np.ndarray:
    """Y (n, T, 3) in [0, 1], smooth in x, exactly 0 for k < k0."""
    X = np.asarray(X, dtype=np.float64)
    tau = np.arange(T) - k0
    tp = np.maximum(tau, 0)[None, :].astype(np.float64)
    x0, x1, x2, x3 = (X[:, j:j + 1] for j in range(4))
    oil = np.exp(-(0.08 + 0.12 * x1) * np.maximum(0.0, tp - (2.0 + 3.0 * x0)))
    gas = 0.99 * oil
    water = 0.9 * (1.0 - np.exp(-(0.15 + 0.25 * x3) * np.maximum(0.0, tp - (1.0 + 4.0 * x2))))
    Y = np.stack([oil, gas, water], axis=-1)
    Y[:, tau < 0, :] = 0.0
    return Y


def exp_coefficients(X: np.ndarray) -> np.ndarray:
    """(n, 5): a0..a4 of make_exponentials."""
    X = np.asarray(X, dtype=np.float64)
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.stack([2 + 0.5 * x0, 0.3 + 0.1 * x1, -0.3 + 0.1 * x2, 0.2 + 0.1 * x3, 0.1 + 0.1 * x0], axis=1)


def make_exponentials(X: np.ndarray, T: int, lam: tuple = EXP_LAM) -> np.ndarray:
    """y_f(tau) = s_f [a0 + a1 l1^tau + a2 l2^tau + rho^tau (a3 cos(theta tau) + a4 sin(theta tau))], k0 = 0."""
    l1, l2, rho, theta = lam
    a = exp_coefficients(X)
    tau = np.arange(T, dtype=np.float64)[None, :]
    g = (a[:, 0:1] + a[:, 1:2] * l1 ** tau + a[:, 2:3] * l2 ** tau
         + rho ** tau * (a[:, 3:4] * np.cos(theta * tau) + a[:, 4:5] * np.sin(theta * tau)))
    return g[..., None] * EXP_SCALE


def make_real_exponentials(X: np.ndarray, T: int) -> np.ndarray:
    """M42 (A2): make_exponentials without the complex pair, y_f = s_f [a0 + a1 0.9^tau + a2 0.7^tau], k0 = 0."""
    a = exp_coefficients(X)
    tau = np.arange(T, dtype=np.float64)[None, :]
    g = a[:, 0:1] + a[:, 1:2] * 0.9 ** tau + a[:, 2:3] * 0.7 ** tau
    return g[..., None] * EXP_SCALE


SLOW_PAIR = (0.9936, 0.0157)          # review 1: rho, theta of the slow exact-DMD pair


def make_slow_pair(X: np.ndarray, T: int) -> np.ndarray:
    """M44 (A2): y_f = s_f rho^tau [a(x) cos(theta tau) + b(x) sin(theta tau)], a = 1 + 0.1 x0,
    b = -(1 + 0.1 x1). Rank 2; > 0 for tau <= 36, crosses 0 at tau of about 47 to 53. k0 = 0."""
    X = np.asarray(X, dtype=np.float64)
    rho, theta = SLOW_PAIR
    tau = np.arange(T, dtype=np.float64)[None, :]
    a, b = 1 + 0.1 * X[:, 0:1], -(1 + 0.1 * X[:, 1:2])
    g = rho ** tau * (a * np.cos(theta * tau) + b * np.sin(theta * tau))
    return g[..., None] * EXP_SCALE


def exp_true_eigs(lam: tuple = EXP_LAM) -> np.ndarray:
    l1, l2, rho, theta = lam
    return np.array([1.0, l1, l2, rho * np.exp(1j * theta), rho * np.exp(-1j * theta)])


def make_growth(X: np.ndarray, T: int) -> np.ndarray:
    """M21: y_f = s_f [(1 + 0.5 x0) + (0.2 + 0.1 x1) 1.03^tau + (0.3 + 0.1 x2) 0.8^tau], k0 = 0."""
    X = np.asarray(X, dtype=np.float64)
    tau = np.arange(T, dtype=np.float64)[None, :]
    g = ((1 + 0.5 * X[:, 0:1]) + (0.2 + 0.1 * X[:, 1:2]) * 1.03 ** tau
         + (0.3 + 0.1 * X[:, 2:3]) * 0.8 ** tau)
    return g[..., None] * EXP_SCALE


def unit_X(n: int, seed: int, p: int = 4) -> np.ndarray:
    """Random X in [0, 1) (strictly inside the box)."""
    return np.random.default_rng(seed).random((n, p))


# --- synthetic Task 1 database ------------------------------------------------

def db_dates() -> np.ndarray:
    return np.arange(np.datetime64("2000-01"), np.datetime64("2000-01") + 3 * T_DB,
                     np.timedelta64(3, "M")).astype("datetime64[D]")


def db_X() -> np.ndarray:
    raw = np.random.default_rng(7).random((N_CASES, 4))
    return (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0))


def synthetic_task1_db(tmp_path: Path) -> dict:
    """Task 1 schema (init_schema) and plain INSERTs. Train 1-18, val 19-24, test 25-30."""
    from dwc26.data.db import connect, init_schema, read_task1_arrays
    from dwc26.data.prepare import arrays_sha256

    path = Path(tmp_path) / "dwc26.sqlite"
    con = connect(path)
    init_schema(con)
    run_id = "prepare-20000101T000000000000Z"
    con.execute(
        "INSERT INTO runs (run_id, task, command, model, config, config_path, config_sha256, seed, git_commit, "
        "git_dirty, start, duration_s, status, outputs, python, packages) VALUES "
        "(?, 'task1', 'prepare', NULL, '{}', 'synthetic', '0', 0, 'synthetic', 0, '2000-01-01T00:00:00.000000Z', "
        "1.0, 'ok', ?, '3.13', '{}')", (run_id, json.dumps({"rate_definition": "R1"})))
    X = db_X()
    X_phys = X_LO + X * (X_HI - X_LO)
    dates = db_dates()
    t_days = (dates - dates[0]).astype(np.float64)
    Y_phys = make_curves(X, T_DB, K0_DB) * np.array(Y_PHYS_SCALE)
    dt = np.diff(t_days, prepend=t_days[0])
    cum = np.cumsum(Y_phys * dt[None, :, None], axis=1)
    split = ["train" if i in TRAIN_IDS else "val" if i in VAL_IDS else "test" for i in range(1, N_CASES + 1)]
    con.executemany(
        f"INSERT INTO parameters (case_id, split, {', '.join(X_NAMES)}, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(i + 1, split[i], *map(float, X_phys[i]), run_id) for i in range(N_CASES)])
    ds = [str(d) for d in dates]
    con.executemany(
        "INSERT INTO production_grid (case_id, date_idx, date, t_days, oil_rate, gas_rate, water_rate, "
        "oil_cum, gas_cum, water_cum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i + 1, k, ds[k], float(t_days[k]), *map(float, Y_phys[i, k]), *map(float, cum[i, k]))
         for i in range(N_CASES) for k in range(T_DB)])
    con.executemany(
        "INSERT INTO history (date_idx, date, t_days, oil_rate, gas_rate, water_rate, oil_cum, gas_cum, water_cum, "
        "oil_rate_reported, gas_rate_reported, water_rate_reported, water_inj_rate_reported) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        [(k, ds[k], float(t_days[k]), *map(float, Y_phys[0, k]), *map(float, cum[0, k]), *map(float, Y_phys[0, k]))
         for k in range(T_DB)])
    train = np.array([s == "train" for s in split])
    factor = Y_phys[train].max(axis=(0, 1))
    fitted = json.dumps([i + 1 for i in range(N_CASES) if train[i]])
    rows = [(f"x_{n}", "x_minmax", n, None, None, float(X_phys[:, j].min()), float(X_phys[:, j].max()), None,
             "design: synthetic", json.dumps(list(range(1, N_CASES + 1))), run_id) for j, n in enumerate(X_NAMES)]
    rows += [(f"y_{f}", "y_max", f, f, float(factor[j]), None, None, "STB/d", "max over train cases, all dates",
              fitted, run_id) for j, f in enumerate(FLUIDS)]
    con.executemany(
        "INSERT INTO scalers (name, kind, variable, fluid, factor, lower, upper, unit, source, fitted_on, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    prod = Y_phys[..., 0] > 0
    gor = Y_phys[..., 1][prod] / Y_phys[..., 0][prod]
    con.execute("INSERT INTO data_checks (run_id, check_id, status, action, detail) VALUES (?, 'C7', 'report', "
                "'continue', ?)", (run_id, json.dumps({"gor_min": {"value": float(gor.min())},
                                                       "gor_max": {"value": float(gor.max())}})))
    con.commit()
    sha = arrays_sha256(read_task1_arrays(con))
    con.execute("UPDATE runs SET outputs = ? WHERE run_id = ?",
                (json.dumps({"rate_definition": "R1", "arrays_sha256": sha}), run_id))
    con.commit()
    con.close()
    return {"db": path, "run_id": run_id, "X": X, "X_phys": X_phys, "Y_phys": Y_phys, "dates": dates,
            "factor": factor, "arrays_sha256": sha}


# --- configs ------------------------------------------------------------------

REPORT = {"plateau_frac": 0.98, "breakthrough_frac": 0.01, "forecast_steps": 80,
          "coverage_z": 1.959963984540054}
SPEED_TINY = {"n_batch_rows": 500, "n_calls": 5, "call_rows": 32, "mcmc_calls": 5000, "limit_s": 600}
WEIGHTS = {"d012": [0.25, 0.25, 0.5], "equal": [1 / 3, 1 / 3, 1 / 3]}


def tiny_cfg_dict(tmp_path: Path, model: str, db: Path | None = None) -> dict:
    tmp_path = Path(tmp_path)
    cfg = {
        "model": model,
        "seed": 0,
        "paths": {"db": str(db or tmp_path / "dwc26.sqlite"), "models": str(tmp_path / "models"),
                  "logs": str(tmp_path / "logs")},
        "data": {"t_cut": None},
        "objective": {"fluids": ["oil", "water"]},
    }
    if model in ("rnn", "lstm"):
        space = {
            "num_layers": {"type": "choice", "values": [1, 2]},
            "hidden_size": {"type": "choice", "values": [16]},
            "lr": {"type": "loguniform", "low": 3.0e-3, "high": 1.0e-2},
            "batch_size": {"type": "choice", "values": [4, 8]},
            "fluid_weights": {"type": "choice", "values": ["d012", "equal"]},
        }
        if model == "rnn":
            space["cell"] = {"type": "choice", "values": ["rnn", "gru"]}
        cfg.update({
            "fluid_weight_sets": copy.deepcopy(WEIGHTS),
            "network": {"input_options": ["A", "B", "C"], "x_affine": True},
            "training": {"seeds": [0, 1], "tune_seeds": 1, "max_epochs": 5, "patience": 3, "min_delta": 0.0,
                         "grad_clip_norm": 1.0,
                         "adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8, "weight_decay": 0.0},
                         "dtype": "float32", "torch_threads": 1, "n_jobs": 1, "predict_batch": 2048},
            "tuning": {"n_trials": 3, "sampler": "lhs", "space": space},
        })
    elif model == "dmd":
        cfg.update({
            "fluid_weight_sets": copy.deepcopy(WEIGHTS),
            "dmd": {"basis": "real", "real_kappa_max": 3.0, "real_kappa_sep": 0.005, "energy_report": 0.99,
                    "unstable": "clip", "stab_tol": 1.0e-9, "imag_tol": 1.0e-10, "sigma_rel_tol": 1.0e-10,
                    "dup_tol": 1.0e-8, "r_max": None},
            "gp": {"alpha": 1.0e-10, "normalize_y": True, "n_restarts_optimizer": 0,
                   "constant_bounds": [1.0e-3, 1.0e3], "length_scale_bounds": [1.0e-2, 1.0e2],
                   "noise_init": 1.0e-4, "noise_bounds": [1.0e-10, 1.0], "std_includes_basis_residual": True},
            "tuning": {"n_trials": 2, "sampler": "grid", "space": {
                "rank": {"type": "choice", "values": [3]},
                "kernel": {"type": "choice", "values": ["matern25"]},
                "fluid_weights": {"type": "choice", "values": ["d012"]},
                "gp_target": {"type": "choice", "values": ["coef", "proj"]},
            }},
        })
    elif model == "b1":
        cfg["knn"] = {"n_neighbors": 3, "weights": "distance", "algorithm": "brute"}
    cfg["report"] = dict(REPORT)
    if model in ("rnn", "lstm", "dmd"):
        cfg["speed"] = dict(SPEED_TINY)
    return cfg


def write_cfg(cfg: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def tiny_cfg(tmp_path: Path, model: str, db: Path | None = None, **overrides) -> Path:
    cfg = tiny_cfg_dict(tmp_path, model, db)
    for dotted, value in overrides.items():
        d = cfg
        keys = dotted.split("__")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return write_cfg(cfg, Path(tmp_path) / f"{model}.yaml")


def fit_settings(model: str, tmp_path: Path, variant: str = "A", **tuned) -> dict:
    """Flat fit settings for a direct fit: the tiny config, first value of each dimension."""
    from dwc26.config import load_model_config
    from dwc26.models.tuning import resolve_settings

    cfg = load_model_config(tiny_cfg(tmp_path, model))
    chosen = {}
    for dim, spec in cfg.get("tuning", {}).get("space", {}).items():
        chosen[dim] = spec["values"][0] if spec["type"] == "choice" else spec["low"]
    chosen.update(tuned)
    if model in ("b0", "b1", "dmd"):
        variant = "main"
    return resolve_settings(cfg, variant, chosen)


ALL_MODELS = ("rnn", "lstm", "dmd", "b0", "b1")
PROXY_MODELS = ("rnn", "lstm", "dmd")
NEURAL_MODELS = ("rnn", "lstm")


def fit_model(name: str, tmp_path: Path, X, Y, val, *, variant: str = "A", seeds=None, **tuned):
    """Fits one model directly (no pipeline) with the tiny settings."""
    from dwc26.models.registry import get_class

    s = fit_settings(name, tmp_path, variant, **tuned)
    if seeds is not None:
        s["seeds"] = list(seeds)
    return get_class(name)().fit(X, Y, s, val=val)


def curves_data(n_fit: int = 12, n_val: int = 4, T: int = 13, k0: int = 2, seed: int = 3):
    X = unit_X(n_fit, seed)
    Xv = unit_X(n_val, seed + 100)
    return X, make_curves(X, T, k0), Xv, make_curves(Xv, T, k0)
