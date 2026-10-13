"""Proxy model for the DWC26 reservoir simulator: 4 uncertainties -> production rates.

Maps the four uncertain parameters of file ``01 Introduction.xlsx`` (tab "Uncertainty
Values": Fault Transmissibility, Porosity Multiplier, Permeability Multiplier,
Aquifer Pore Volume) to the oil / gas / water production **rates** of a case on each
recorded date. The rates come from file ``02 Field Production.xlsx`` (the per-case
columns of the "* production rate" tabs), cut to the 41 quarterly dates that every
simulated case and the observed history share (``data_after_engineering/curves_quarterly.csv``).
The native simulator dates cannot be used as model output because they differ from
case to case (notebook 1, section 3).

Model: a small deep MLP (PyTorch, CPU). Inputs are the 4 parameters scaled to [0, 1]
using train-case min/max, optionally extended with the per-case grid features of
``grid_case_features.csv``. The network learns residuals from the train-case mean
production curve. Hyper-parameters are selected by 5-fold cross-validation over
the 70 train cases (folds from ``cases.csv``); the final model is a 5-seed ensemble
trained on all 70 train cases. Cases 71-85 (validation) and 86-100 (test) are kept
out of fitting.

Optional extra inputs (files ``03``-``06``): the grid of each case is fully determined
by the 4 parameters (notebook 1, section 5.1), so the per-case grid features carry
little extra information; they are nevertheless offered as ``feature_set="p4g8"``
and compared against the 4 parameters alone (``feature_set="p4"``).

Run as a script (writes to ``cod/proxy_outputs/``):

    python proxy_model.py --tune      # hyper-parameter search (config_search.csv)
    python proxy_model.py --train     # final ensemble + evaluation + figures
    python proxy_model.py --all       # both
"""
from __future__ import annotations

import argparse
import copy
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parent
DATA = CODE / "data_after_engineering"
OUT = CODE / "proxy_outputs"

PHASES = ["oil", "gas", "water"]                    # user-facing order of the rates
PARAMS = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
GRID_FEATURES = ["total_pv", "aquifer_pv_grid", "reservoir_pv", "mean_poro",
             "mean_permx", "mean_permx_zone", "mean_tranx", "mean_trany"]
N_CASES, N_DATES = 100, 41
SEED = 0


# --------------------------------------------------------------------------- data

def load_data():
    """Load everything the proxy needs. Returns a dict of arrays / tables."""
    cases = pd.read_csv(DATA / "cases.csv")
    quarterly = pd.read_csv(DATA / "curves_quarterly.csv")
    quarterly["date"] = pd.to_datetime(quarterly["date"])

    # targets: (100, 41, 3) reported rates, phase order = PHASES
    rate_cols = [f"{p}_rate" for p in PHASES]
    Y = np.stack([quarterly.loc[quarterly.case_id == c, rate_cols].to_numpy()
                  for c in range(1, N_CASES + 1)])
    dates = quarterly.loc[quarterly.case_id == 1, "date"].to_numpy()
    t_days = quarterly.loc[quarterly.case_id == 1, "t_days"].to_numpy(dtype=float)

    grid = pd.read_csv(DATA / "grid_case_features.csv").set_index("case_id")
    grid = grid.loc[cases.case_id.to_numpy(), GRID_FEATURES].reset_index(drop=True)

    z = np.load(DATA / "curves_quarterly_cum.npz")
    assert list(z["quantities"]) == ["gas_cum", "oil_cum", "water_cum"]  # notebook 1 order

    return {"cases": cases, "Y": Y, "dates": dates, "t_days": t_days, "grid": grid}


def build_inputs(data, feature_set, fit_indices=None):
    """Model input matrix for a feature set ('p4' = 4 params, 'p4g8' = 4 params + 8 grid features).

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
    """Standardize phase-wise residuals from a training-only mean rate curve."""

    def __init__(self, use_log: bool):
        self.use_log = use_log

    def fit(self, Y_train: np.ndarray):                      # (n, T, 3)
        self.base_curve_ = Y_train.mean(axis=0)
        residual = Y_train - self.base_curve_
        Z = (
            np.sign(residual) * np.log1p(np.abs(residual))
            if self.use_log
            else residual
        )
        flat = Z.reshape(-1, 3)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-12
        return self

    def transform(self, Y):                                  # -> (n, T*3) float32
        residual = Y - self.base_curve_
        Z = (
            np.sign(residual) * np.log1p(np.abs(residual))
            if self.use_log
            else residual
        )
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
    """MLP: case parameters -> 41 dates x 3 phase rates (standardized)."""

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
    """Yield (train_idx, val_idx) per CV fold, over the 70 train cases."""
    tr = cases.index[cases["split"] == "train"].to_numpy()
    folds = sorted(cases.loc[tr, "cv_fold"].unique())
    for f in folds:
        va = tr[cases.loc[tr, "cv_fold"].to_numpy() == f]
        yield f, np.setdiff1d(tr, va), va


def run_config(data, cfg: Config, folds, seeds=(0,)):
    """Evaluate one config: per fold, train with early stopping, score on the fold's cases.

    Returns a dict with the config, per-fold val RMSE (original units) and epochs used.
    Seeds are averaged inside a fold (a mini-ensemble) to reduce seed noise while tuning.
    """
    cases = data["cases"]
    fold_rmse, epochs_used = [], []
    for f, itr, iva in [(f, i, v) for f, i, v in _fold_indices(cases) if f in folds]:
        X = build_inputs(data, cfg.feature_set, fit_indices=itr)
        # Fit target transforms on this fold's training cases only.
        scaler = PhaseScaler(cfg.use_log).fit(data["Y"][itr])
        Ztr = scaler.transform(data["Y"][itr])
        Zva = scaler.transform(data["Y"][iva])
        preds, epochs = [], []
        for s in seeds:
            state, used, _ = _train_one(X[itr], Ztr, X[iva], Zva, cfg, s)
            preds.append(scaler.inverse(predict(state, X[iva], X.shape[1], cfg)))
            epochs.append(used)
        pred = np.mean(preds, axis=0)
        rm = np.sqrt(np.mean((pred - data["Y"][iva]) ** 2))
        fold_rmse.append(rm)
        epochs_used.append(int(np.mean(epochs)))
    return {"config": cfg, "folds": list(folds), "val_rmse": float(np.mean(fold_rmse)),
            "val_rmse_per_fold": [float(r) for r in fold_rmse],
            "epochs_used": epochs_used}


def tune(data, stage1_folds=(0,), stage2_top=3, seeds=(0,)):
    """Two-stage search over a small grid; writes config_search.csv and returns the best Config."""
    grid = []
    for feature_set in ["p4", "p4g8"]:
        for use_log in [False, True]:
            for hidden in [(32, 32), (64, 64), (128, 128)]:
                for wd in [1e-4, 1e-3, 1e-2]:
                    grid.append(Config(feature_set=feature_set, use_log=use_log,
                                       hidden=hidden, weight_decay=wd))
    rows, t0 = [], time.time()
    print(f"stage 1: {len(grid)} configs on fold(s) {stage1_folds}")
    for i, cfg in enumerate(grid):
        r = run_config(data, cfg, stage1_folds, seeds)
        rows.append({**_cfg_row(cfg), **{f"rmse_fold{f}": v for f, v in
                                         zip(r["folds"], r["val_rmse_per_fold"])},
                     "mean_val_rmse": r["val_rmse"]})
        print(f"  [{i + 1:>2}/{len(grid)}] {cfg.feature_set} log={int(cfg.use_log)} "
              f"hidden={cfg.hidden} wd={cfg.weight_decay:g} -> {r['val_rmse']:.1f}")
    top = sorted(rows, key=lambda r: r["mean_val_rmse"])[:stage2_top]
    print(f"stage 2: top {stage2_top} on all 5 folds")
    for r in top:
        cfg = _row_cfg(r)
        res = run_config(data, cfg, folds={0, 1, 2, 3, 4}, seeds=seeds)
        for f, v in zip(res["folds"], res["val_rmse_per_fold"]):
            r[f"rmse_fold{f}"] = v
        r["mean_val_rmse"] = res["val_rmse"]
        r["epochs_used"] = res["epochs_used"]
        print(f"  {cfg.feature_set} log={int(cfg.use_log)} hidden={cfg.hidden} "
              f"wd={cfg.weight_decay:g} -> {res['val_rmse']:.1f}  epochs={res['epochs_used']}")
    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "config_search.csv", index=False)
    best_row = min(rows, key=lambda r: r["mean_val_rmse"])
    print(f"best: {best_row}  ({time.time() - t0:.0f} s)")
    best_cfg = _row_cfg(best_row)
    import json
    (OUT / "best_config.json").write_text(json.dumps(asdict(best_cfg) | {"hidden": list(best_cfg.hidden)}, indent=2))
    return best_cfg, table


def _cfg_row(cfg: Config):
    return {"feature_set": cfg.feature_set, "use_log": cfg.use_log,
            "hidden": "x".join(map(str, cfg.hidden)), "dropout": cfg.dropout,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay}


def _row_cfg(r) -> Config:
    return Config(feature_set=r["feature_set"], use_log=bool(r["use_log"]),
                  hidden=tuple(int(x) for x in r["hidden"].split("x")),
                  dropout=r["dropout"], lr=r["lr"], weight_decay=r["weight_decay"])


# ------------------------------------------------------------------ final fit

def train_final(data, cfg: Config, n_seeds=5, epochs=None):
    """Train the final ensemble on all 70 train cases.

    epochs: fixed budget. If None, the median of the epochs used by the CV folds of
    this config is used (recomputed with a single fold set covering all 5 folds).
    Returns (states, scaler, epochs_used).
    """
    cases = data["cases"]
    tr = cases.index[cases["split"] == "train"].to_numpy()
    X = build_inputs(data, cfg.feature_set, fit_indices=tr)
    scaler = PhaseScaler(cfg.use_log).fit(data["Y"][tr])
    if epochs is None:
        epochs = int(np.median(run_config(data, cfg, folds={0, 1, 2, 3, 4})["epochs_used"]))
        print(f"final epochs from CV median: {epochs}")
    Ztr = scaler.transform(data["Y"][tr])
    states = []
    for k in range(n_seeds):
        state, used = _train_fixed(X[tr], Ztr, cfg, SEED + 100 + k, epochs)
        states.append(state)
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
    """Predict the train-case mean rate curve for every case."""
    cases = data["cases"]
    tr = cases["split"].to_numpy() == "train"
    mean_curve = data["Y"][tr].mean(axis=0)
    return np.repeat(mean_curve[None, :, :], N_CASES, axis=0)


def baseline_ridge(data, cfg: Config):
    """Linear ridge baseline on the same inputs (alpha by 5-fold CV on the train cases)."""
    from sklearn.linear_model import Ridge

    cases = data["cases"]
    Y = data["Y"]
    tr = cases.index[cases["split"] == "train"].to_numpy()
    Yflat = Y.reshape(N_CASES, -1)
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_a, best_rmse = alphas[0], float("inf")
    for a in alphas:
        errs = []
        for f, itr, iva in _fold_indices(cases):
            X_fold = build_inputs(data, cfg.feature_set, fit_indices=itr)
            m = Ridge(alpha=a).fit(X_fold[itr], Yflat[itr])
            errs.append(np.sqrt(np.mean((m.predict(X_fold[iva]) - Yflat[iva]) ** 2)))
        if np.mean(errs) < best_rmse:
            best_a, best_rmse = a, np.mean(errs)
    X = build_inputs(data, cfg.feature_set, fit_indices=tr)
    model = Ridge(alpha=best_a).fit(X[tr], Yflat[tr])
    pred = np.clip(model.predict(X).reshape(Y.shape), 0.0, None)
    print(f"ridge: alpha={best_a:g} (CV RMSE {best_rmse:.1f})")
    return pred


# -------------------------------------------------------------------- metrics

def metrics_table(Y_true, Y_pred, split_mask, split_name, model_name):
    """Per-phase and pooled RMSE / MAE / R2 / nRMSE for one split and model."""
    rows = []
    yt, yp = Y_true[split_mask], Y_pred[split_mask]
    for j, phase in enumerate(PHASES):
        t, p = yt[:, :, j], yp[:, :, j]
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        ss = float(((t - t.mean()) ** 2).sum())
        rows.append({
            "split": split_name, "model": model_name, "phase": phase,
            "rmse": rmse, "mae": float(np.mean(np.abs(t - p))),
            "r2": float(1 - ((t - p) ** 2).sum() / ss) if ss > 0 else np.nan,
            "nrmse": rmse / float(t.std()),
        })
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


# --------------------------------------------------------------------- driver

def run_tune():
    data = load_data()
    cfg, table = tune(data)
    print("best config:", cfg)
    return cfg


def run_train(cfg: Config, n_seeds=5, epochs=None):
    """Train the final ensemble, evaluate everywhere, save all artifacts to OUT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = load_data()
    cases = data["cases"]
    splits = cases["split"].to_numpy()
    case_ids = cases["case_id"].to_numpy()
    Y = data["Y"]

    states, scaler, epochs = train_final(data, cfg, n_seeds=n_seeds, epochs=epochs)
    X = build_inputs(data, cfg.feature_set)
    pred = predict_ensemble(states, scaler, X, cfg)

    # baselines with the same inputs
    pred_ridge = baseline_ridge(data, cfg)
    pred_mean = baseline_mean_curve(data)

    rows = []
    for sp in ["train", "validation", "test"]:
        m = splits == sp
        rows += metrics_table(Y, pred, m, sp, "dnn_ensemble")
        rows += metrics_table(Y, pred_ridge, m, sp, "ridge")
        rows += metrics_table(Y, pred_mean, m, sp, "mean_curve")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print(metrics.to_string(index=False,
                            float_format=lambda v: f"{v:,.3f}"))

    # long-format predictions of the proxy for every case
    rec = []
    for i, cid in enumerate(case_ids):
        for t_i, (d, td) in enumerate(zip(data["dates"], data["t_days"])):
            for j, ph in enumerate(PHASES):
                rec.append((cid, splits[i], d, td, ph, Y[i, t_i, j], pred[i, t_i, j]))
    pd.DataFrame(rec, columns=["case_id", "split", "date", "t_days", "phase",
                               "rate_true", "rate_pred"]).to_csv(
        OUT / "predictions.csv", index=False)
    wide = []
    for i, cid in enumerate(case_ids):
        for t_i, (date, elapsed_days) in enumerate(zip(data["dates"], data["t_days"])):
            row = {
                "case_id": cid,
                "split": splits[i],
                "date": pd.Timestamp(date).date().isoformat(),
                "t_days": elapsed_days,
            }
            for j, phase in enumerate(PHASES):
                row[f"{phase}_rate_true"] = Y[i, t_i, j]
                row[f"{phase}_rate_pred"] = pred[i, t_i, j]
            wide.append(row)
    pd.DataFrame(wide).to_csv(OUT / "predictions_wide.csv", index=False)

    # worst validation / test cases (for the curve pictures)
    wv = worst_cases(Y, pred, splits == "validation", case_ids)
    wt = worst_cases(Y, pred, splits == "test", case_ids)
    print("worst validation cases:", wv)
    print("worst test cases:", wt)

    # ---- figures
    colors = {"oil": "#1f77b4", "gas": "#ff7f0e", "water": "#2ca02c"}
    units = {"oil": "Oil rate (STB/d)", "gas": "Gas rate (MSCF/d)", "water": "Water rate (STB/d)"}
    dates = pd.DatetimeIndex(data["dates"])

    # 1: pred vs truth scatter, validation + test
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharex="col")
    for r, sp in enumerate(["validation", "test"]):
        m = splits == sp
        for j, ph in enumerate(PHASES):
            t, p = Y[m][:, :, j].ravel(), pred[m][:, :, j].ravel()
            ax[r, j].scatter(t, p, s=14, alpha=0.6, color=colors[ph])
            lim = [0, max(t.max(), p.max()) * 1.05]
            ax[r, j].plot(lim, lim, "k--", lw=1)
            rmse = float(np.sqrt(np.mean((t - p) ** 2)))
            ax[r, j].set_title(f"{sp} - {ph} (RMSE {rmse:,.0f})")
            ax[r, j].set_xlabel("simulator")
            ax[r, j].set_ylabel("proxy")
            ax[r, j].grid(alpha=0.3)
    fig.suptitle("Proxy vs simulator rates, 41 quarterly dates")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pred_vs_truth.png", dpi=150)
    plt.close(fig)

    # 2/3: curves for the best / median / worst cases of validation and test
    for sp, worst in [("validation", wv), ("test", wt)]:
        m_ids = case_ids[splits == sp]
        rmses = np.sqrt(np.mean((Y - pred) ** 2, axis=(1, 2)))
        sp_rmse = {cid: rmses[cid - 1] for cid in m_ids}
        ranked = sorted(sp_rmse, key=sp_rmse.get)
        show = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
        fig, axes = plt.subplots(len(show), 3, figsize=(13, 3 * len(show)), sharex=True)
        for r, cid in enumerate(show):
            i = cid - 1
            for j, ph in enumerate(PHASES):
                a = axes[r, j]
                a.plot(dates, Y[i, :, j], color=colors[ph], lw=2, label="simulator")
                a.plot(dates, pred[i, :, j], color="black", lw=1.4, ls="--", label="proxy")
                a.set_title(f"Case {cid} ({sp}) - {ph}" if j == 1 else f"Case {cid}")
                a.set_ylabel(units[ph])
                a.grid(alpha=0.3)
                if r == 0 and j == 0:
                    a.legend()
        fig.suptitle(f"Proxy vs simulator rate curves ({sp}: best / median / worst case)")
        fig.tight_layout()
        fig.savefig(OUT / f"fig2_curves_{sp}.png", dpi=150)
        plt.close(fig)

    # 4: RMSE by date, per phase, validation + test
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for sp, ls in [("validation", "-"), ("test", "--")]:
        m = splits == sp
        for j, ph in enumerate(PHASES):
            rm = np.sqrt(np.mean((Y[m][:, :, j] - pred[m][:, :, j]) ** 2, axis=0))
            ax.plot(dates, rm, ls, color=colors[ph], label=f"{sp} - {ph}")
    ax.set_ylabel("RMSE (rate units)")
    ax.set_title("Proxy error by date (41 quarterly dates)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig4_rmse_by_date.png", dpi=150)
    plt.close(fig)

    # ---- checkpoint
    torch.save({
        "config": asdict(cfg) | {"hidden": list(cfg.hidden)},
        "seeds": [SEED + 100 + k for k in range(n_seeds)],
        "states": states,
        "scaler": {
            "use_log": scaler.use_log,
            "mean": scaler.mean_,
            "std": scaler.std_,
            "base_curve": scaler.base_curve_,
        },
        "phases": PHASES, "params": PARAMS,
        "grid_features": GRID_FEATURES if cfg.feature_set == "p4g8" else [],
        "grid_scaler": ({"mean": data["grid"][GRID_FEATURES].to_numpy(dtype=np.float64)[
                            splits == "train"].mean(axis=0),
                         "std": np.where(
                             data["grid"][GRID_FEATURES].to_numpy(dtype=np.float64)[
                                 splits == "train"].std(axis=0) > 0,
                             data["grid"][GRID_FEATURES].to_numpy(dtype=np.float64)[
                                 splits == "train"].std(axis=0),
                             1.0,
                         )}
                        if cfg.feature_set == "p4g8" else None),
        "dates": np.datetime_as_string(data["dates"], unit="D"),
        "t_days": data["t_days"],
        "parameter_min": cases.loc[splits == "train", PARAMS].min().to_numpy(dtype=float),
        "parameter_max": cases.loc[splits == "train", PARAMS].max().to_numpy(dtype=float),
        "epochs": epochs,
        "env": {"torch": torch.__version__, "python": platform.python_version()},
    }, OUT / "final_model.pt")

    print(f"\nartifacts written to {OUT}")
    return metrics


def predict_case(model_path, parameters, grid_features=None):
    """Predict the three rates for each recorded date from raw uncertainty values.

    For a ``p4g8`` checkpoint, supply the eight values in ``GRID_FEATURES`` order.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg_data = checkpoint["config"].copy()
    cfg_data["hidden"] = tuple(cfg_data["hidden"])
    cfg = Config(**cfg_data)

    parameter_values = np.asarray(parameters, dtype=np.float64)
    if parameter_values.shape != (len(PARAMS),) or not np.isfinite(parameter_values).all():
        raise ValueError(f"parameters must contain {len(PARAMS)} finite values in {PARAMS} order.")
    p_min = np.asarray(checkpoint["parameter_min"], dtype=np.float64)
    p_max = np.asarray(checkpoint["parameter_max"], dtype=np.float64)
    p_range = p_max - p_min
    if np.any(p_range <= 0):
        raise ValueError("Saved training data has a zero-width uncertainty range.")
    parts = [((parameter_values - p_min) / p_range)[None, :]]

    if cfg.feature_set == "p4g8":
        grid_values = np.asarray(grid_features, dtype=np.float64)
        if grid_values.shape != (len(GRID_FEATURES),) or not np.isfinite(grid_values).all():
            raise ValueError(
                f"grid_features must contain {len(GRID_FEATURES)} finite values in {GRID_FEATURES} order."
            )
        grid_scaler = checkpoint["grid_scaler"]
        g_mean = np.asarray(grid_scaler["mean"], dtype=np.float64)
        g_std = np.asarray(grid_scaler["std"], dtype=np.float64)
        parts.append(((grid_values - g_mean) / g_std)[None, :])
    elif grid_features is not None:
        raise ValueError("grid_features should only be supplied for a p4g8 model.")

    X = np.hstack(parts)
    scaler = PhaseScaler(checkpoint["scaler"]["use_log"])
    scaler.mean_ = np.asarray(checkpoint["scaler"]["mean"])
    scaler.std_ = np.asarray(checkpoint["scaler"]["std"])
    scaler.base_curve_ = np.asarray(checkpoint["scaler"]["base_curve"])
    prediction = predict_ensemble(checkpoint["states"], scaler, X, cfg)[0]
    return pd.DataFrame(
        prediction,
        columns=[f"{phase}_rate" for phase in PHASES],
    ).assign(date=pd.to_datetime(checkpoint["dates"]), t_days=checkpoint["t_days"])[
        ["date", "t_days", "oil_rate", "gas_rate", "water_rate"]
    ]


def load_best_config() -> Config:
    """Best config from the last --tune run (best_config.json), else the defaults."""
    import json

    p = OUT / "best_config.json"
    if not p.exists():
        print(f"warning: {p} not found - using the default config (run --tune first)")
        return Config()
    d = json.loads(p.read_text())
    d["hidden"] = tuple(d["hidden"])
    print(f"loaded best config from {p}: {d}")
    return Config(**d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tune", action="store_true", help="run the hyper-parameter search")
    ap.add_argument("--train", action="store_true", help="train the final ensemble")
    ap.add_argument("--seeds", type=int, default=5, help="ensemble size for --train")
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="fixed training epochs; omit to derive the epoch count from 5-fold CV",
    )
    ap.add_argument("--feature-set", choices=["p4", "p4g8"], help="input set for --train")
    ap.add_argument("--hidden", nargs="+", type=int, help="hidden layer widths, e.g. --hidden 32 32")
    args = ap.parse_args()

    cfg = None
    if args.tune:
        cfg = run_tune()
    if args.train:
        if cfg is None:
            cfg = load_best_config()
        if args.feature_set is not None:
            cfg.feature_set = args.feature_set
        if args.hidden is not None:
            if not args.hidden or any(width < 1 for width in args.hidden):
                ap.error("--hidden widths must all be positive integers")
            cfg.hidden = tuple(args.hidden)
        if args.epochs is not None and args.epochs < 1:
            ap.error("--epochs must be at least 1")
        if args.seeds < 1:
            ap.error("--seeds must be at least 1")
        run_train(cfg, n_seeds=args.seeds, epochs=args.epochs)
    if not (args.tune or args.train):
        ap.print_help()


if __name__ == "__main__":
    main()
