"""Proxy variants for the improvement tests (temporary, improve_proxy/).

Builds on ``src/proxy_lib.py`` (Elton's model) without changing it. Every variant is a
cumulative proxy: raw parameters (m, 4) -> cumulative curves. What a variant can change:

* ``target``: "full" = 41 dates x (oil, gas, water), as Elton; "ow_window" = oil and water
  on the 37 producing dates only (gas is rebuilt as GOR x oil, the 4 pre-production dates are 0).
* ``output``: "residual" = standardized difference from the mean train curve (Elton);
  "monotone" = non-negative quarterly increments, so the cumulative cannot decrease.
* training: width, depth, dropout, learning rate, weight decay, batch size, number of seeds.
* ``family``: "mlp", "gp" (PCA + Gaussian process per score, Zi Yang), "mlp+gp" (average).

``VariantProxy`` has the same interface as ``proxy_lib.CumulativeProxy``, so the
history-matching notebook can use either.
"""
from __future__ import annotations

import copy
import json
import pickle
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import proxy_lib as pl

PHASES = pl.PHASES                      # oil, gas, water
OIL, GAS, WATER = 0, 1, 2


@dataclass
class VConfig:
    name: str = "V0 baseline"
    family: str = "mlp"                 # mlp | gp | mlp+gp
    target: str = "full"                # full | ow_window
    output: str = "residual"            # residual | monotone
    feature_set: str = "p4g8"
    hidden: tuple = (128, 128)
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 0                 # 0 = full batch (Elton)
    max_epochs: int = 3000
    patience: int = 300
    n_seeds: int = 5
    gp_n_pc: int = 8

    def short(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# ------------------------------------------------------------------ targets and scalers

def window_index(y_hist):
    return np.where(y_hist[:, OIL] > 0)[0]


class Target:
    """Which part of the (n, 41, 3) cumulative array a variant learns, and how to rebuild it."""

    def __init__(self, kind, y_hist, Y_train):
        self.kind = kind
        self.K = window_index(y_hist)
        if kind == "full":
            self.dates, self.phases = np.arange(Y_train.shape[1]), [OIL, GAS, WATER]
        elif kind == "ow_window":
            self.dates, self.phases = self.K, [OIL, WATER]
        else:
            raise ValueError(kind)
        # gas / oil ratio by date (train cases), used to rebuild gas for "ow_window"
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # pre-production dates: 0 / 0
            r = Y_train[:, :, GAS] / Y_train[:, :, OIL]
            self.gor = np.nan_to_num(np.nanmean(np.where(Y_train[:, :, OIL] > 0, r, np.nan), axis=0))

    def cut(self, Y):
        return Y[:, self.dates][:, :, self.phases]

    def rebuild(self, Ys, T=41):
        out = np.zeros((len(Ys), T, 3))
        for j, p in enumerate(self.phases):
            out[:, self.dates, p] = Ys[:, :, j]
        if GAS not in self.phases:
            out[:, :, GAS] = out[:, :, OIL] * self.gor[None]
        return out


class GenScaler:
    """Elton's PhaseScaler for any (dates, phases) shape: residual from the mean curve, per-phase z-score."""

    def fit(self, Y):
        self.base_, self.T, self.P = Y.mean(axis=0), Y.shape[1], Y.shape[2]
        R = (Y - self.base_).reshape(-1, self.P)
        self.mean_, self.std_ = R.mean(axis=0), R.std(axis=0) + 1e-12
        return self

    def transform(self, Y):
        return (((Y - self.base_) - self.mean_) / self.std_).reshape(len(Y), -1).astype(np.float32)

    def inverse(self, Z):
        return np.asarray(Z).reshape(-1, self.T, self.P) * self.std_ + self.mean_ + self.base_


class MonoScaler:
    """For the monotone output: per-phase scale and the mean train increment per date and phase."""

    def fit(self, Y):
        self.T, self.P = Y.shape[1], Y.shape[2]
        self.scale_ = Y.reshape(-1, self.P).std(axis=0) + 1e-12
        inc = np.diff(np.concatenate([np.zeros((len(Y), 1, self.P)), Y], axis=1), axis=1)
        self.base_inc_ = np.maximum(inc.mean(axis=0), 0.0) / self.scale_        # (T, P) scaled
        return self

    def transform(self, Y):
        return (Y / self.scale_).reshape(len(Y), -1).astype(np.float32)

    def inverse(self, Z):
        return np.asarray(Z).reshape(-1, self.T, self.P) * self.scale_


SOFTPLUS_ONE = float(np.log(np.e - 1.0))       # softplus(SOFTPLUS_ONE) = 1


class Net(nn.Module):
    """Elton's MLP; with ``base_inc`` it outputs a cumulative built from non-negative increments."""

    def __init__(self, in_dim, T, P, hidden, dropout, base_inc=None):
        super().__init__()
        blocks, d = [], in_dim
        for h in hidden:
            blocks += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            d = h
        blocks.append(nn.Linear(d, T * P))
        self.net = nn.Sequential(*blocks)
        self.T, self.P = T, P
        self.register_buffer("base_inc", None if base_inc is None else torch.as_tensor(base_inc, dtype=torch.float32))

    def forward(self, x):
        z = self.net(x)
        if self.base_inc is None:
            return z
        inc = self.base_inc[None] * F.softplus(z.view(-1, self.T, self.P) + SOFTPLUS_ONE)
        return torch.cumsum(inc, dim=1).reshape(len(x), -1)


# ------------------------------------------------------------------ inputs

class Inputs:
    """Scaled inputs, fitted on the given training cases (as Elton's build_inputs)."""

    def __init__(self, cfg, theta_tr, grid_tr):
        self.cfg = cfg
        self.p_min = theta_tr.min(axis=0)
        self.p_rng = np.where(theta_tr.max(axis=0) - self.p_min > 0, theta_tr.max(axis=0) - self.p_min, 1.0)
        self.g_mu = grid_tr.mean(axis=0)
        self.g_sd = np.where(grid_tr.std(axis=0) > 0, grid_tr.std(axis=0), 1.0)

    def __call__(self, theta, grid):
        parts = [(theta - self.p_min) / self.p_rng]
        if self.cfg.feature_set == "p4g8":
            parts.append((grid - self.g_mu) / self.g_sd)
        return np.hstack(parts).astype(np.float32)


# ------------------------------------------------------------------ training

def _batches(n, bs, gen):
    if bs <= 0 or bs >= n:
        yield slice(None)
        return
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def train_net(cfg, X, Z, seed, epochs=None, Xva=None, Zva=None, base_inc=None, T=41, P=3):
    """One network. Early stopping on (Xva, Zva) when given, else a fixed number of epochs.
    Returns (state_dict, epochs_used)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Net(X.shape[1], T, P, cfg.hidden, cfg.dropout, base_inc)
    n_ep = cfg.max_epochs if epochs is None else epochs
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=cfg.lr * 0.01)
    Xt, Zt = torch.from_numpy(X), torch.from_numpy(Z)
    if Xva is not None:
        Xv, Zv = torch.from_numpy(Xva), torch.from_numpy(Zva)
    best, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(1, n_ep + 1):
        model.train()
        for b in _batches(len(Xt), cfg.batch_size, gen):
            opt.zero_grad()
            F.mse_loss(model(Xt[b]), Zt[b]).backward()
            opt.step()
        sched.step()
        if Xva is None:
            continue
        model.eval()
        with torch.no_grad():
            val = float(F.mse_loss(model(Xv), Zv))
        if val < best - 1e-10:
            best, best_ep, wait = val, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if Xva is None:
        return model.state_dict(), n_ep
    return best_state, best_ep


@dataclass
class FittedMLP:
    cfg: VConfig
    target: Target
    scaler: object
    inputs: Inputs
    states: list
    epochs: int
    models: list = field(default_factory=list, repr=False)

    def build(self):
        T, P = len(self.target.dates), len(self.target.phases)
        base = self.scaler.base_inc_ if self.cfg.output == "monotone" else None
        self.models = []
        for s in self.states:
            m = Net(self.inputs(np.zeros((1, 4)), np.zeros((1, 8))).shape[1], T, P, self.cfg.hidden, self.cfg.dropout, base)
            m.load_state_dict(s)
            self.models.append(m.eval())
        return self

    def predict(self, theta, grid):
        if not self.models:
            self.build()
        X = torch.from_numpy(self.inputs(theta, grid))
        with torch.no_grad():
            preds = [self.scaler.inverse(m(X).numpy()) for m in self.models]
        return np.clip(self.target.rebuild(np.mean(preds, axis=0)), 0.0, None)

    def member_predictions(self, theta, grid):
        if not self.models:
            self.build()
        X = torch.from_numpy(self.inputs(theta, grid))
        with torch.no_grad():
            return np.stack([np.clip(self.target.rebuild(self.scaler.inverse(m(X).numpy())), 0, None) for m in self.models])


def _prep(cfg, Y_tr, y_hist):
    tgt = Target(cfg.target, y_hist, Y_tr)
    Ys = tgt.cut(Y_tr)
    sc = (MonoScaler() if cfg.output == "monotone" else GenScaler()).fit(Ys)
    return tgt, sc, Ys


def fit_mlp(cfg, theta, grid, Y, y_hist, idx, epochs, seeds=None):
    """Final ensemble on cases ``idx`` with a fixed number of epochs (Elton's protocol)."""
    tgt, sc, Ys = _prep(cfg, Y[idx], y_hist)
    inp = Inputs(cfg, theta[idx], grid[idx])
    X, Z = inp(theta[idx], grid[idx]), sc.transform(Ys)
    base = sc.base_inc_ if cfg.output == "monotone" else None
    seeds = seeds if seeds is not None else [pl.SEED + 100 + k for k in range(cfg.n_seeds)]
    states = [train_net(cfg, X, Z, s, epochs=epochs, base_inc=base, T=Ys.shape[1], P=Ys.shape[2])[0] for s in seeds]
    return FittedMLP(cfg, tgt, sc, inp, states, epochs).build()


def cv_mlp(cfg, theta, grid, Y, y_hist, folds, seed=0):
    """Elton's CV: per fold, train on the other folds with early stopping on this fold.
    folds: list of (train_idx, holdout_idx). Returns out-of-fold predictions (n, 41, 3) and epochs."""
    oof = np.full(Y.shape, np.nan)
    eps = []
    for itr, iva in folds:
        tgt, sc, Ys = _prep(cfg, Y[itr], y_hist)
        inp = Inputs(cfg, theta[itr], grid[itr])
        base = sc.base_inc_ if cfg.output == "monotone" else None
        state, ep = train_net(cfg, inp(theta[itr], grid[itr]), sc.transform(Ys), seed,
                              Xva=inp(theta[iva], grid[iva]), Zva=sc.transform(tgt.cut(Y[iva])),
                              base_inc=base, T=Ys.shape[1], P=Ys.shape[2])
        oof[iva] = FittedMLP(cfg, tgt, sc, inp, [state], ep).predict(theta[iva], grid[iva])
        eps.append(ep)
    return oof, eps


# ------------------------------------------------------------------ GP on PCA scores (Zi Yang)

@dataclass
class FittedGP:
    cfg: VConfig
    target: Target
    scale: np.ndarray
    inputs: Inputs
    pca: object
    zs: np.ndarray
    gps: list

    def predict(self, theta, grid):
        X = self.inputs(theta, grid)[:, :4]
        Zn = np.column_stack([g.predict(X) for g in self.gps])
        Yf = self.pca.inverse_transform(Zn * self.zs)
        Ys = Yf.reshape(len(X), len(self.target.dates), len(self.target.phases)) * self.scale
        return np.clip(self.target.rebuild(Ys), 0.0, None)


def fit_gp(cfg, theta, grid, Y, y_hist, idx):
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

    tgt = Target(cfg.target, y_hist, Y[idx])
    scale = y_hist.max(axis=0)[tgt.phases]
    Ys = tgt.cut(Y[idx]) / scale
    Yf = Ys.reshape(len(idx), -1)
    inp = Inputs(VConfig(feature_set="p4"), theta[idx], grid[idx])
    X = inp(theta[idx], grid[idx])
    pca = PCA(n_components=cfg.gp_n_pc).fit(Yf)
    Z = pca.transform(Yf)
    zs = Z.std(0) + 1e-12
    gps = []
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)     # length scales of unimportant inputs run to the bound
        for k in range(cfg.gp_n_pc):
            kern = C(1.0) * RBF(np.ones(4), (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-9, 1e-1))   # Zi Yang's bounds
            gps.append(GaussianProcessRegressor(kern, n_restarts_optimizer=2, random_state=pl.SEED).fit(X, Z[:, k] / zs[k]))
    return FittedGP(cfg, tgt, scale, inp, pca, zs, gps)


@dataclass
class FittedAverage:
    parts: list

    def predict(self, theta, grid):
        return np.mean([p.predict(theta, grid) for p in self.parts], axis=0)


# ------------------------------------------------------------------ metrics (history-matching units)

def hm_errors(pred, Y, y_hist, idx):
    """RMS and max error as a share of the final observed cumulative, window dates, oil and water."""
    K = window_index(y_hist)
    fin = y_hist[-1]
    R = (pred[idx][:, K] - Y[idx][:, K]) / fin
    out = {}
    for j, ph in [(OIL, "oil"), (WATER, "water")]:
        out[f"{ph}_rms_%"] = float(np.sqrt((R[:, :, j] ** 2).mean()) * 100)
        out[f"{ph}_max_%"] = float(np.abs(R[:, :, j]).max() * 100)
    out["J_%"] = float(np.sqrt((out["oil_rms_%"] ** 2 + out["water_rms_%"] ** 2) / 2))
    return out


def hm_score(C, y_hist):
    K = window_index(y_hist)
    R = (C[:, K][:, :, [OIL, WATER]] - y_hist[K][:, [OIL, WATER]]) / y_hist[-1][[OIL, WATER]]
    return np.sqrt((R ** 2).mean(axis=1).mean(axis=1))


# ------------------------------------------------------------------ export and loading

def save_variant(path, fitted, mapper, theta_tr, meta_extra=None):
    """Pickle a fitted variant (MLP states are plain tensors; GP via sklearn)."""
    obj = {"fitted": fitted, "mapper_state": mapper.state(),
           "train_min": theta_tr.min(axis=0).tolist(), "train_max": theta_tr.max(axis=0).tolist()}
    if meta_extra:
        obj.update(meta_extra)
    if isinstance(fitted, FittedMLP):
        fitted.models = []                     # rebuilt on load
    if isinstance(fitted, FittedAverage):
        for p in fitted.parts:
            if isinstance(p, FittedMLP):
                p.models = []
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class VariantProxy:
    """Same interface as proxy_lib.CumulativeProxy (see its docstring)."""

    def __init__(self, input_dir, window_dates, quantities=("oil_cum", "water_cum")):
        input_dir = Path(input_dir)
        self.meta = json.loads((input_dir / "proxy_meta.json").read_text())
        model_path = input_dir / self.meta["model_file"]
        if pl.sha256(model_path) != self.meta["model_sha256"]:
            raise RuntimeError(f"{model_path} does not match the hash in proxy_meta.json")
        # only load pickles written by this repository (improve_proxy)
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        self.fitted = obj["fitted"]
        self.mapper = pl.GridMapper(state=obj["mapper_state"])
        self.dates = np.array(self.meta["dates"]).astype("datetime64[D]")
        window_dates = np.asarray(window_dates).astype("datetime64[D]")
        self.idx = np.searchsorted(self.dates, window_dates)
        if not np.array_equal(self.dates[self.idx], window_dates):
            raise ValueError("window dates are not on the proxy's 41 quarterly dates")
        self.quantities = list(quantities)
        self.cols = [PHASES.index(q.replace("_cum", "")) for q in self.quantities]
        self.sigma_proxy = np.array([self.meta["sigma_proxy"][q] for q in self.quantities])
        self.lo = np.array([self.meta["train_min"][p] for p in pl.PARAMS])
        self.hi = np.array([self.meta["train_max"][p] for p in pl.PARAMS])
        self.n_calls = 0

    def _predict(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        return self.fitted.predict(theta, self.mapper.transform(theta))

    def full(self, theta):
        pred = self._predict(theta)
        return {f"{p}_cum": pred[:, :, j] for j, p in enumerate(PHASES)}

    def __call__(self, theta):
        self.n_calls += 1
        return self._predict(theta)[:, self.idx][:, :, self.cols]
