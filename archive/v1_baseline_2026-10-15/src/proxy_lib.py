"""Proxy model library for DWC26 (Elton's model, shared by notebooks 2 and 3).

The model code is copied from ``Elton Model/proxy_model.py`` and
``Elton Model/forecast_pipeline.py`` (grid mapping, cumulative twin). Only the paths
and the I/O are changed, so the two notebooks import one definition of the model:

* ``notebook2_proxy_model.ipynb`` tunes, trains and evaluates the proxy, then exports
  it to ``hist_match_input/``.
* ``notebook3_hist_match.ipynb`` loads it back with ``CumulativeProxy``.

Model: a small MLP (PyTorch, CPU). Inputs are the 4 uncertainty parameters scaled to
[0, 1] with the train-case min/max, optionally plus 8 grid features (``p4g8``). The
network learns the standardized residual from the train-case mean curve, for all
41 dates x 3 phases at once. The final model is a 5-seed ensemble trained on cases 1-70.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_after_engineering"

PHASES = ["oil", "gas", "water"]                    # model output order of the phases
PARAMS = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
GRID_FEATURES = ["total_pv", "aquifer_pv_grid", "reservoir_pv", "mean_poro",
                 "mean_permx", "mean_permx_zone", "mean_tranx", "mean_trany"]
N_CASES, N_DATES = 100, 41
SEED = 0

# Zone sizes measured in notebook 1 section 5.1 (active mask identical in all 100 cases).
N_ACTIVE, N_ZONE_PERM, N_ZONE_AQ = 42_512, 11_575, 2_286
FZ_PERM = N_ZONE_PERM / N_ACTIVE


# --------------------------------------------------------------------------- data

def load_data(target="rate"):
    """Everything the proxy needs. ``target`` is "rate" or "cumulative".

    Y has shape (100, 41, 3) in PHASES order. ``y_hist`` (41, 3) is the observed
    history in the same units as Y.
    """
    cases = pd.read_csv(DATA / "cases.csv")
    quarterly = pd.read_csv(DATA / "curves_quarterly.csv")
    quarterly["date"] = pd.to_datetime(quarterly["date"])
    dates = quarterly.loc[quarterly.case_id == 1, "date"].to_numpy()
    t_days = quarterly.loc[quarterly.case_id == 1, "t_days"].to_numpy(dtype=float)

    grid = pd.read_csv(DATA / "grid_case_features.csv").set_index("case_id")
    grid = grid.loc[cases.case_id.to_numpy(), GRID_FEATURES].reset_index(drop=True)

    z = np.load(DATA / "curves_quarterly_cum.npz")
    # the npz stores its third axis as [gas, oil, water] (notebook 1); reorder to PHASES
    assert list(z["quantities"]) == ["gas_cum", "oil_cum", "water_cum"]
    order = [list(z["quantities"]).index(f"{p}_cum") for p in PHASES]
    if target == "rate":
        rate_cols = [f"{p}_rate" for p in PHASES]
        Y = np.stack([quarterly.loc[quarterly.case_id == c, rate_cols].to_numpy()
                      for c in range(1, N_CASES + 1)])
        # reported history rates; they follow a different convention than the cases'
        # reported rates (notebook 1), so they are shown for context only, never matched
        y_hist = pd.read_csv(DATA / "history_observed.csv")[rate_cols].to_numpy()
    elif target == "cumulative":
        Y = z["Y"][:, :, order].astype(np.float64)
        y_hist = z["y_hist"][:, order].astype(np.float64)
    else:
        raise ValueError(target)
    return {"cases": cases, "Y": Y, "y_hist": y_hist, "dates": dates, "t_days": t_days,
            "grid": grid, "target": target}


def build_inputs(data, feature_set, fit_indices=None):
    """Model input matrix ('p4' = 4 params, 'p4g8' = 4 params + 8 grid features).

    Scalers use fit_indices, or all training cases when fit_indices is omitted.
    """
    cases = data["cases"]
    if fit_indices is None:
        fit_indices = cases.index[cases["split"] == "train"].to_numpy()
    raw_parameters = cases[PARAMS].to_numpy(dtype=np.float64)
    parameter_min = raw_parameters[fit_indices].min(axis=0)
    parameter_range = raw_parameters[fit_indices].max(axis=0) - parameter_min
    parameter_range = np.where(parameter_range > 0, parameter_range, 1.0)
    parts = [(raw_parameters - parameter_min) / parameter_range]
    if feature_set == "p4g8":
        g = data["grid"][GRID_FEATURES].to_numpy(dtype=np.float64)
        mu, sd = g[fit_indices].mean(axis=0), g[fit_indices].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        parts.append((g - mu) / sd)
    elif feature_set != "p4":
        raise ValueError(feature_set)
    return np.hstack(parts)


class PhaseScaler:
    """Standardize phase-wise residuals from a training-only mean curve."""

    def __init__(self, use_log: bool):
        self.use_log = use_log

    def fit(self, Y_train: np.ndarray):                      # (n, T, 3)
        self.base_curve_ = Y_train.mean(axis=0)
        residual = Y_train - self.base_curve_
        Z = np.sign(residual) * np.log1p(np.abs(residual)) if self.use_log else residual
        flat = Z.reshape(-1, 3)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-12
        return self

    def transform(self, Y):                                  # -> (n, T*3) float32
        residual = Y - self.base_curve_
        Z = np.sign(residual) * np.log1p(np.abs(residual)) if self.use_log else residual
        return ((Z - self.mean_) / self.std_).reshape(len(Y), -1).astype(np.float32)

    def inverse(self, Z):                                    # (n, T*3) -> (n, T, 3)
        residual = np.asarray(Z).reshape(-1, N_DATES, 3) * self.std_ + self.mean_
        if self.use_log:
            residual = np.sign(residual) * np.expm1(np.abs(residual))
        return residual + self.base_curve_


# --------------------------------------------------------------------------- model

@dataclass
class Config:
    feature_set: str = "p4"          # "p4" or "p4g8"
    use_log: bool = False            # signed log1p on mean-curve residuals
    hidden: tuple = (64, 64)         # hidden layer widths
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 3000
    patience: int = 300              # early-stopping patience (epochs without val loss gain)


class RateProxy(nn.Module):
    """MLP: case inputs -> 41 dates x 3 phases (standardized)."""

    def __init__(self, in_dim: int, out_dim: int, hidden=(64, 64), dropout=0.05):
        super().__init__()
        blocks, d = [], in_dim
        for h in hidden:
            blocks += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            d = h
        blocks.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def _train_one(Xtr, Ztr, Xva, Zva, cfg: Config, seed: int):
    """Train one net (full batch). Returns (best_state_dict, n_epochs_used, best_val_loss)."""
    torch.manual_seed(seed)
    model = RateProxy(Xtr.shape[1], Ztr.shape[1], cfg.hidden, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs,
                                                       eta_min=cfg.lr * 0.01)
    Xtr_t, Ztr_t = torch.from_numpy(Xtr).float(), torch.from_numpy(Ztr).float()
    Xva_t, Zva_t = torch.from_numpy(Xva).float(), torch.from_numpy(Zva).float()
    best, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        opt.zero_grad()
        F.mse_loss(model(Xtr_t), Ztr_t).backward()
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(F.mse_loss(model(Xva_t), Zva_t))
        if val < best - 1e-10:
            best, best_ep, wait = val, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    return best_state, best_ep, best


def predict(state_dict, X, in_dim, cfg: Config) -> np.ndarray:
    model = RateProxy(in_dim, N_DATES * 3, cfg.hidden, cfg.dropout)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32))).numpy()


# ------------------------------------------------------------------- CV and tuning

def _fold_indices(cases):
    """Yield (fold, train_idx, val_idx) per CV fold, over the 70 train cases."""
    tr = cases.index[cases["split"] == "train"].to_numpy()
    folds = sorted(cases.loc[tr, "cv_fold"].unique())
    for f in folds:
        va = tr[cases.loc[tr, "cv_fold"].to_numpy() == f]
        yield f, np.setdiff1d(tr, va), va


def run_config(data, cfg: Config, folds, seeds=(0,)):
    """Evaluate one config: per fold, train with early stopping, score on the fold's cases."""
    cases = data["cases"]
    fold_rmse, epochs_used = [], []
    for f, itr, iva in [(f, i, v) for f, i, v in _fold_indices(cases) if f in folds]:
        X = build_inputs(data, cfg.feature_set, fit_indices=itr)
        scaler = PhaseScaler(cfg.use_log).fit(data["Y"][itr])   # fold-train cases only
        Ztr = scaler.transform(data["Y"][itr])
        Zva = scaler.transform(data["Y"][iva])
        preds, epochs = [], []
        for s in seeds:
            state, used, _ = _train_one(X[itr], Ztr, X[iva], Zva, cfg, s)
            preds.append(scaler.inverse(predict(state, X[iva], X.shape[1], cfg)))
            epochs.append(used)
        pred = np.mean(preds, axis=0)
        fold_rmse.append(np.sqrt(np.mean((pred - data["Y"][iva]) ** 2)))
        epochs_used.append(int(np.mean(epochs)))
    return {"config": cfg, "folds": sorted(folds), "val_rmse": float(np.mean(fold_rmse)),
            "val_rmse_per_fold": [float(r) for r in fold_rmse], "epochs_used": epochs_used}


def tuning_grid():
    """Elton's search grid: 2 feature sets x 2 target transforms x 3 widths x 3 weight decays = 36."""
    grid = []
    for feature_set in ["p4", "p4g8"]:
        for use_log in [False, True]:
            for hidden in [(32, 32), (64, 64), (128, 128)]:
                for wd in [1e-4, 1e-3, 1e-2]:
                    grid.append(Config(feature_set=feature_set, use_log=use_log,
                                       hidden=hidden, weight_decay=wd))
    return grid


def tune(data, stage1_folds=(0,), stage2_top=3, seeds=(0,), verbose=True):
    """Two-stage search: all 36 configs on fold 0, then the top 3 on all 5 folds."""
    grid = tuning_grid()
    rows, t0 = [], time.time()
    say = print if verbose else (lambda *a, **k: None)
    say(f"stage 1: {len(grid)} configs on fold(s) {stage1_folds}")
    for i, cfg in enumerate(grid):
        r = run_config(data, cfg, stage1_folds, seeds)
        rows.append({**cfg_row(cfg), **{f"rmse_fold{f}": v for f, v in
                                        zip(r["folds"], r["val_rmse_per_fold"])},
                     "mean_val_rmse": r["val_rmse"]})
        say(f"  [{i + 1:>2}/{len(grid)}] {cfg.feature_set} log={int(cfg.use_log)} "
            f"hidden={cfg.hidden} wd={cfg.weight_decay:g} -> {r['val_rmse']:.1f}")
    top = sorted(rows, key=lambda r: r["mean_val_rmse"])[:stage2_top]
    say(f"stage 2: top {stage2_top} on all 5 folds")
    for r in top:
        cfg = row_cfg(r)
        res = run_config(data, cfg, folds={0, 1, 2, 3, 4}, seeds=seeds)
        for f, v in zip(res["folds"], res["val_rmse_per_fold"]):
            r[f"rmse_fold{f}"] = v
        r["mean_val_rmse"] = res["val_rmse"]
        r["epochs_used"] = res["epochs_used"]
        say(f"  {cfg.feature_set} log={int(cfg.use_log)} hidden={cfg.hidden} "
            f"wd={cfg.weight_decay:g} -> {res['val_rmse']:.1f}  epochs={res['epochs_used']}")
    table = pd.DataFrame(rows)
    best_row = min(rows, key=lambda r: r["mean_val_rmse"])
    say(f"tuning done in {time.time() - t0:.0f} s")
    return row_cfg(best_row), table


def cfg_row(cfg: Config):
    return {"feature_set": cfg.feature_set, "use_log": cfg.use_log,
            "hidden": "x".join(map(str, cfg.hidden)), "dropout": cfg.dropout,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay}


def row_cfg(r) -> Config:
    return Config(feature_set=r["feature_set"], use_log=bool(r["use_log"]),
                  hidden=tuple(int(x) for x in str(r["hidden"]).split("x")),
                  dropout=float(r["dropout"]), lr=float(r["lr"]),
                  weight_decay=float(r["weight_decay"]))


def config_from_json(path) -> Config:
    d = json.loads(Path(path).read_text())
    d["hidden"] = tuple(d["hidden"])
    return Config(**d)


# ------------------------------------------------------------------ final fit

def train_final(data, cfg: Config, n_seeds=5, epochs=None, verbose=True):
    """Train the final ensemble on all 70 train cases.

    epochs: fixed budget. If None, the median of the early-stopping epochs of the
    5 CV folds of this config is used. Returns (states, scaler, epochs_used).
    """
    cases = data["cases"]
    tr = cases.index[cases["split"] == "train"].to_numpy()
    X = build_inputs(data, cfg.feature_set, fit_indices=tr)
    scaler = PhaseScaler(cfg.use_log).fit(data["Y"][tr])
    if epochs is None:
        epochs = int(np.median(run_config(data, cfg, folds={0, 1, 2, 3, 4})["epochs_used"]))
        if verbose:
            print(f"final epochs from CV median: {epochs}")
    Ztr = scaler.transform(data["Y"][tr])
    states = []
    for k in range(n_seeds):
        state, used = _train_fixed(X[tr], Ztr, cfg, SEED + 100 + k, epochs)
        states.append(state)
        if verbose:
            print(f"  ensemble member {k + 1}/{n_seeds} trained ({used} epochs)")
    return states, scaler, epochs


def _train_fixed(Xtr, Ztr, cfg: Config, seed: int, epochs: int):
    """Train one net for a fixed number of epochs (no early stopping)."""
    torch.manual_seed(seed)
    model = RateProxy(Xtr.shape[1], Ztr.shape[1], cfg.hidden, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.lr * 0.01)
    Xtr_t, Ztr_t = torch.from_numpy(Xtr).float(), torch.from_numpy(Ztr).float()
    for _ in range(epochs):
        opt.zero_grad()
        F.mse_loss(model(Xtr_t), Ztr_t).backward()
        opt.step()
        sched.step()
    return model.state_dict(), epochs


def predict_ensemble(states, scaler, X, cfg: Config) -> np.ndarray:
    """Average the ensemble members and map back to original units (clipped at 0)."""
    preds = [scaler.inverse(predict(s, X, X.shape[1], cfg)) for s in states]
    return np.clip(np.mean(preds, axis=0), 0.0, None)


# ------------------------------------------------------------------- baselines

def baseline_mean_curve(data):
    """Predict the train-case mean curve for every case."""
    tr = data["cases"]["split"].to_numpy() == "train"
    return np.repeat(data["Y"][tr].mean(axis=0)[None], N_CASES, axis=0)


def baseline_ridge(data, cfg: Config, verbose=True):
    """Linear ridge baseline on the same inputs (alpha by 5-fold CV on the train cases)."""
    from sklearn.linear_model import Ridge

    cases, Y = data["cases"], data["Y"]
    tr = cases.index[cases["split"] == "train"].to_numpy()
    Yflat = Y.reshape(N_CASES, -1)
    best_a, best_rmse = None, float("inf")
    for a in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        errs = []
        for f, itr, iva in _fold_indices(cases):
            X_fold = build_inputs(data, cfg.feature_set, fit_indices=itr)
            m = Ridge(alpha=a).fit(X_fold[itr], Yflat[itr])
            errs.append(np.sqrt(np.mean((m.predict(X_fold[iva]) - Yflat[iva]) ** 2)))
        if np.mean(errs) < best_rmse:
            best_a, best_rmse = a, np.mean(errs)
    X = build_inputs(data, cfg.feature_set, fit_indices=tr)
    model = Ridge(alpha=best_a).fit(X[tr], Yflat[tr])
    if verbose:
        print(f"ridge: alpha={best_a:g} (CV RMSE {best_rmse:,.1f})")
    return np.clip(model.predict(X).reshape(Y.shape), 0.0, None)


# -------------------------------------------------------------------- metrics

def metrics_table(Y_true, Y_pred, split_mask, split_name, model_name):
    """Per-phase and pooled RMSE / MAE / R2 / nRMSE for one split and model."""
    rows = []
    yt, yp = Y_true[split_mask], Y_pred[split_mask]
    for j, phase in enumerate(PHASES):
        t, p = yt[:, :, j], yp[:, :, j]
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        ss = float(((t - t.mean()) ** 2).sum())
        rows.append({"split": split_name, "model": model_name, "phase": phase,
                     "rmse": rmse, "mae": float(np.mean(np.abs(t - p))),
                     "r2": float(1 - ((t - p) ** 2).sum() / ss) if ss > 0 else np.nan,
                     "nrmse": rmse / float(t.std())})
    t, p = yt.ravel(), yp.ravel()
    rows.append({"split": split_name, "model": model_name, "phase": "all",
                 "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
                 "mae": float(np.mean(np.abs(t - p))),
                 "r2": float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()),
                 "nrmse": float(np.sqrt(np.mean((t - p) ** 2)) / t.std())})
    return rows


def worst_cases(Y_true, Y_pred, split_mask, case_ids, k=3):
    """The k cases of a split with the highest RMSE over all phases."""
    rmses = np.sqrt(np.mean((Y_true[split_mask] - Y_pred[split_mask]) ** 2, axis=(1, 2)))
    order = np.argsort(rmses)[::-1]
    return [(int(case_ids[split_mask][i]), float(rmses[i])) for i in order[:k]]


# ------------------------------------------------- params -> grid features mapping

@functools.lru_cache(maxsize=1)
def _case_one():
    # cached: the history-matching samplers call the mapper ~100,000 times
    cases = pd.read_csv(DATA / "cases.csv")
    feats = pd.read_csv(DATA / "grid_case_features.csv")
    p1 = cases.loc[cases.case_id == 1, PARAMS].iloc[0]
    f1 = feats.loc[feats.case_id == 1, GRID_FEATURES].iloc[0]
    return p1, f1


def grid_features_analytic(theta):
    """Analytic grid summaries for any parameter set.

    Every case grid is case 1 scaled by the parameter ratios (notebook 1 section 5.1):
    porosity everywhere, permeability inside its zone, pore volume inside the aquifer
    zone. The eight summaries are closed-form functions of the three ratios.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    p1, f1 = _case_one()
    r_poro = theta[:, 1] / float(p1["poro_mult"])
    r_perm = theta[:, 2] / float(p1["perm_mult"])
    r_aq = theta[:, 3] / float(p1["aquifer_pv"])
    fz = FZ_PERM
    w_tx = fz * float(f1["mean_permx_zone"]) / float(f1["mean_permx"])
    return np.column_stack([
        r_poro * (f1["reservoir_pv"] + r_aq * f1["aquifer_pv_grid"]),  # total_pv
        r_poro * r_aq * f1["aquifer_pv_grid"],                         # aquifer_pv_grid
        r_poro * f1["reservoir_pv"],                                   # reservoir_pv
        r_poro * f1["mean_poro"],                                      # mean_poro
        f1["mean_permx"] + (r_perm - 1.0) * fz * f1["mean_permx_zone"],# mean_permx
        r_perm * f1["mean_permx_zone"],                                # mean_permx_zone
        f1["mean_tranx"] * (1.0 + (r_perm - 1.0) * w_tx),              # mean_tranx
        f1["mean_trany"] * (1.0 + (r_perm - 1.0) * w_tx),              # mean_trany
    ])


class GridMapper:
    """Parameters -> 8 grid features (Elton's rule, validated on all 100 cases).

    Primary method is the analytic scaling above. Acceptance: every feature within
    0.2 sigma of the stored ``grid_case_features.csv`` values (sigma = train-case std).
    Otherwise a linear fallback on ``[1, r_poro, r_perm, r_poro * r_aq]`` is fitted on
    the 70 train cases and must pass the same check on the 30 held-out cases.
    """

    TOL = 0.2

    def __init__(self, data=None, state=None):
        if state is not None:                      # rebuilt from a checkpoint
            self.method = state["method"]
            self.coef_ = None if state["coef"] is None else np.asarray(state["coef"])
            self.report = state.get("report")
            return
        cases = data["cases"]
        theta = cases[PARAMS].to_numpy(dtype=np.float64)
        stored = data["grid"][GRID_FEATURES].to_numpy(dtype=np.float64)
        train = cases["split"].to_numpy() == "train"
        sigma = np.where(stored[train].std(axis=0) > 0, stored[train].std(axis=0), 1.0)
        err_a = np.abs(grid_features_analytic(theta) - stored) / sigma
        self.method = "analytic" if (err_a.max(axis=0) < self.TOL).all() else "fallback"
        self.coef_ = None
        rep = pd.DataFrame({"analytic_max_error_sigma": err_a.max(axis=0)},
                           index=pd.Index(GRID_FEATURES, name="feature"))
        if self.method == "fallback":
            A = self._design(theta)
            self.coef_, *_ = np.linalg.lstsq(A[train], stored[train], rcond=None)
            err_f = np.abs(A @ self.coef_ - stored) / sigma
            rep["fallback_max_error_train_sigma"] = err_f[train].max(axis=0)
            rep["fallback_max_error_holdout_sigma"] = err_f[~train].max(axis=0)
            if err_f[~train].max() >= self.TOL:
                raise RuntimeError("grid feature mapping failed: analytic and fallback both exceed "
                                   f"{self.TOL} sigma (holdout max {err_f[~train].max():.3g})")
        self.report = rep

    @staticmethod
    def _design(theta):
        p1, _ = _case_one()
        r_poro = theta[:, 1] / float(p1["poro_mult"])
        r_perm = theta[:, 2] / float(p1["perm_mult"])
        r_aq = theta[:, 3] / float(p1["aquifer_pv"])
        return np.column_stack([np.ones(len(theta)), r_poro, r_perm, r_poro * r_aq])

    def transform(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if self.method == "analytic":
            return grid_features_analytic(theta)
        return self._design(theta) @ self.coef_

    def state(self):
        return {"method": self.method, "coef": None if self.coef_ is None else self.coef_.tolist()}


# ------------------------------------------------------------------- checkpoints

def save_checkpoint(path, cfg, states, scaler, data, epochs, n_seeds, grid_mapper=None):
    """Same layout as Elton's final_model.pt / cumulative_model.pt, plus the grid mapper."""
    cases = data["cases"]
    splits = cases["split"].to_numpy()
    g = data["grid"][GRID_FEATURES].to_numpy(dtype=np.float64)[splits == "train"]
    torch.save({
        "config": asdict(cfg) | {"hidden": list(cfg.hidden)},
        "seeds": [SEED + 100 + k for k in range(n_seeds)],
        "states": states,
        "scaler": {"use_log": scaler.use_log, "mean": scaler.mean_,
                   "std": scaler.std_, "base_curve": scaler.base_curve_},
        "phases": PHASES, "params": PARAMS,
        "grid_features": GRID_FEATURES if cfg.feature_set == "p4g8" else [],
        "grid_scaler": ({"mean": g.mean(axis=0),
                         "std": np.where(g.std(axis=0) > 0, g.std(axis=0), 1.0)}
                        if cfg.feature_set == "p4g8" else None),
        "dates": np.datetime_as_string(data["dates"], unit="D"),
        "t_days": data["t_days"],
        "parameter_min": cases.loc[splits == "train", PARAMS].min().to_numpy(dtype=float),
        "parameter_max": cases.loc[splits == "train", PARAMS].max().to_numpy(dtype=float),
        "epochs": epochs,
        "target": data["target"],
        "grid_mapper": None if grid_mapper is None else grid_mapper.state(),
        "env": {"torch": torch.__version__, "python": platform.python_version()},
    }, path)


def load_checkpoint(path):
    # weights_only=False: the checkpoint holds numpy arrays and the config dict.
    # Only load checkpoints trained in this repository.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    d = dict(ckpt["config"])
    d["hidden"] = tuple(d["hidden"])
    ckpt["_cfg"] = Config(**d)
    sc = PhaseScaler(ckpt["scaler"]["use_log"])
    sc.mean_ = np.asarray(ckpt["scaler"]["mean"])
    sc.std_ = np.asarray(ckpt["scaler"]["std"])
    sc.base_curve_ = np.asarray(ckpt["scaler"]["base_curve"])
    ckpt["_scaler"] = sc
    gm = ckpt.get("grid_mapper")
    ckpt["_mapper"] = GridMapper(state=gm) if gm is not None else None
    # build the ensemble members once (Elton's predict() rebuilds them on every call,
    # which dominates the cost of the many small batches an MCMC sends)
    in_dim = len(PARAMS) + (len(GRID_FEATURES) if ckpt["_cfg"].feature_set == "p4g8" else 0)
    ckpt["_models"] = []
    for s in ckpt["states"]:
        m = RateProxy(in_dim, N_DATES * 3, ckpt["_cfg"].hidden, ckpt["_cfg"].dropout)
        m.load_state_dict(s)
        ckpt["_models"].append(m.eval())
    return ckpt


def predict_from_params(ckpt, theta, grid=None, batch=8192):
    """Curves (n, 41, 3) in PHASES order for raw parameter sets (n, 4).

    For a p4g8 checkpoint the grid features come from the checkpoint's grid mapper
    (Elton's GridMapper) unless ``grid`` is given.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    p_min = np.asarray(ckpt["parameter_min"], dtype=np.float64)
    p_range = np.asarray(ckpt["parameter_max"], dtype=np.float64) - p_min
    parts = [(theta - p_min) / p_range]
    if ckpt["_cfg"].feature_set == "p4g8":
        if grid is None:
            if ckpt.get("_mapper") is None:
                raise ValueError("p4g8 checkpoint without a grid mapper: pass grid=")
            grid = ckpt["_mapper"].transform(theta)
        parts.append((grid - np.asarray(ckpt["grid_scaler"]["mean"])) / np.asarray(ckpt["grid_scaler"]["std"]))
    X = np.hstack(parts).astype(np.float32)
    sc = ckpt["_scaler"]
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch])
            # same as predict_ensemble: inverse-transform each member, average, clip at 0
            preds = [sc.inverse(m(xb).numpy()) for m in ckpt["_models"]]
            out.append(np.clip(np.mean(preds, axis=0), 0.0, None))
    return np.concatenate(out, axis=0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------- history-matching proxy

class CumulativeProxy:
    """The proxy as the history-matching notebook uses it.

    ``proxy(theta)`` takes raw parameters (m, 4) in PARAMS order and returns the
    cumulative curves (m, Tw, Q) on the matching window, for the quantities in
    ``quantities`` (names like "oil_cum"). ``proxy.full(theta)`` returns all 41 dates
    and all three phases as a dict of (m, 41) arrays.

    Reads ``cumulative_model.pt`` and ``proxy_meta.json`` from ``input_dir`` and stops
    if the model file does not match the hash recorded at export time.
    """

    def __init__(self, input_dir, window_dates, quantities=("oil_cum", "water_cum")):
        input_dir = Path(input_dir)
        self.meta = json.loads((input_dir / "proxy_meta.json").read_text())
        model_path = input_dir / self.meta["model_file"]
        if sha256(model_path) != self.meta["model_sha256"]:
            raise RuntimeError(f"{model_path} does not match the hash in proxy_meta.json; "
                               "re-export it from notebook2_proxy_model.ipynb")
        self.ckpt = load_checkpoint(model_path)
        self.dates = np.asarray(self.ckpt["dates"]).astype("datetime64[D]")
        window_dates = np.asarray(window_dates).astype("datetime64[D]")
        self.idx = np.searchsorted(self.dates, window_dates)
        if not np.array_equal(self.dates[self.idx], window_dates):
            raise ValueError("window dates are not on the proxy's 41 quarterly dates")
        self.quantities = list(quantities)
        self.cols = [PHASES.index(q.replace("_cum", "")) for q in self.quantities]
        # proxy error per matched quantity (share of the final observed cumulative)
        err = self.meta["sigma_proxy"]
        self.sigma_proxy = np.array([err[q] for q in self.quantities])
        self.lo = np.array([self.meta["train_min"][p] for p in PARAMS])
        self.hi = np.array([self.meta["train_max"][p] for p in PARAMS])
        self.n_calls = 0

    def full(self, theta):
        pred = predict_from_params(self.ckpt, theta)
        return {f"{p}_cum": pred[:, :, j] for j, p in enumerate(PHASES)}

    def __call__(self, theta):
        self.n_calls += 1
        pred = predict_from_params(self.ckpt, theta)
        return pred[:, self.idx][:, :, self.cols]
