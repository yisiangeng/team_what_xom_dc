"""Conditioned sequence networks, strategy S3 (spec task2 sections 2.3 to 2.8; PRD section 6).

- Input options (D-014): A = [u, t_k] at each step; B = [t_k] with the initial
  state set from u; C = [u] at each step. u = 2x - 1 (Article 2, p. 989).
- Time: t_k = k / (T_fit - 1) (PRD section 6), k = k0..K-1.
- Output: one linear layer with 3 units (PRD section 6).
- Loss: fluid-weighted MSE on the raw scaled outputs, dates k0..T_fit-1 (D-012).
- Members: Adam, minibatches in a seeded order, gradient-norm clipping, early
  stopping on the validation J (D-015). The ensemble predicts the clipped mean
  and its std with ddof = 1 (section 2.8).
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.evaluation.metrics import objective_j
from our_work.proxy_model_sijie.src.dwc26.models.base import ModelBase, TrialFailed, clip_nonneg

DTYPES = {"float32": torch.float32, "float64": torch.float64}
CELLS = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
OPTIONS = ("A", "B", "C")


def time_grid(k_start: int, k_end: int, T_fit: int) -> np.ndarray:
    """t_k = k / (T_fit - 1) for k = k_start..k_end-1 (float64)."""
    return np.arange(k_start, k_end, dtype=np.float64) / (T_fit - 1)


def param_input(X: np.ndarray, x_affine: bool) -> np.ndarray:
    """u = 2x - 1 in [-1, 1] (x_affine), else x."""
    return 2.0 * X - 1.0 if x_affine else X


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """sum_f w_f * mean over cases and steps of (pred - target)^2 (section 2.4)."""
    d = pred - target
    return ((d * d).mean(dim=tuple(range(d.dim() - 1))) * w).sum()


class SeqNet(nn.Module):
    """Recurrent cell (LSTM, GRU or tanh RNN) and a linear output layer of 3 units."""

    def __init__(self, cell: str, option: str, p: int, num_layers: int, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if cell not in CELLS:
            raise ContractError(f"SeqNet: unknown cell {cell!r}")
        if option not in OPTIONS:
            raise ContractError(f"SeqNet: unknown input option {option!r}")
        self.cell, self.option, self.p = cell, option, p
        self.num_layers, self.hidden_size = num_layers, hidden_size
        size = {"A": p + 1, "B": 1, "C": p}[option]
        kw = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.rnn = CELLS[cell](size, hidden_size, num_layers, batch_first=True, dropout=0.0, bias=True,
                               dtype=dtype, **kw)
        if option == "B":
            self.h0 = nn.Linear(p, num_layers * hidden_size, dtype=dtype)
            if cell == "lstm":
                self.c0 = nn.Linear(p, num_layers * hidden_size, dtype=dtype)
        self.out = nn.Linear(hidden_size, len(FLUIDS), dtype=dtype)

    def _layers(self, a: torch.Tensor) -> torch.Tensor:
        """(n, num_layers * H) -> (num_layers, n, H)."""
        return a.reshape(a.shape[0], self.num_layers, self.hidden_size).transpose(0, 1).contiguous()

    def initial_state(self, u: torch.Tensor):
        """None (zeros) for options A and C. Option B: h0 = tanh(W_h u + b_h) and, for LSTM,
        c0 = W_c u + b_c, each (num_layers, n, H)."""
        if self.option != "B":
            return None
        h0 = self._layers(torch.tanh(self.h0(u)))
        if self.cell == "lstm":
            return h0, self._layers(self.c0(u))
        return h0

    def inputs(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(n, L, input size) step inputs z_k."""
        n, L = u.shape[0], t.shape[0]
        tt = t.reshape(1, L, 1).expand(n, L, 1)
        if self.option == "A":
            return torch.cat([u.unsqueeze(1).expand(n, L, self.p), tt], dim=2)
        if self.option == "B":
            return tt
        return u.unsqueeze(1).expand(n, L, self.p)

    def forward(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u: (n, p) in [-1, 1]; t: (L,) time values; returns (n, L, 3) raw."""
        z = self.inputs(u, t)
        s0 = self.initial_state(u)
        h, _ = self.rnn(z) if s0 is None else self.rnn(z, s0)
        return self.out(h)


def build_net(settings: dict, p: int) -> SeqNet:
    return SeqNet(settings["cell"], settings["input_option"], p, int(settings["num_layers"]),
                  int(settings["hidden_size"]), dtype=DTYPES[settings["dtype"]])


def _tensor(a: np.ndarray, settings: dict) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=DTYPES[settings["dtype"]])


def fluid_index(settings: dict) -> tuple[int, ...]:
    return tuple(FLUIDS.index(f) for f in settings["objective_fluids"])


@dataclass
class MemberResult:
    seed: int
    state: dict                    # best-epoch state_dict, CPU tensors
    best_epoch: int
    n_epochs: int
    j_val: float                   # J of this member at best_epoch
    history: np.ndarray            # (n_epochs, 3): epoch, train loss, J_val
    fit_s: float


def _net_raw(net: SeqNet, u: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return net(u, t).to(torch.float64).numpy()


def _val_j(raw: np.ndarray, Y_val: np.ndarray, k0: int, fidx: tuple[int, ...]) -> float:
    if not np.isfinite(raw).all():
        raise TrialFailed("non-finite network output on the validation cases")
    full = np.zeros(Y_val.shape)
    full[:, k0:] = clip_nonneg(raw)[0]
    return objective_j(Y_val, full, fidx)


def train_member(settings: dict, seed: int, X: np.ndarray, Y: np.ndarray,
                 X_val: np.ndarray, Y_val: np.ndarray, k0: int) -> MemberResult:
    """Trains one ensemble member (section 2.7). Module-level, so that worker
    processes can run it. Raises TrialFailed on a non-finite loss."""
    t_start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, T_fit, p = X.shape[0], Y.shape[1], X.shape[1]
    net = build_net(settings, p)
    a = settings["adam"]
    opt = torch.optim.Adam(net.parameters(), lr=float(settings["lr"]), betas=(a["beta1"], a["beta2"]),
                           eps=a["eps"], weight_decay=a["weight_decay"])
    t = _tensor(time_grid(k0, T_fit, T_fit), settings)
    u = _tensor(param_input(X, settings["x_affine"]), settings)
    y = _tensor(Y[:, k0:, :], settings)
    w = _tensor(np.asarray(settings["fluid_weights"], dtype=np.float64), settings)
    u_val = _tensor(param_input(X_val, settings["x_affine"]), settings)
    fidx = fluid_index(settings)
    bs, clip = int(settings["batch_size"]), settings["grad_clip_norm"]
    patience, min_delta = int(settings["patience"]), float(settings["min_delta"])

    history = []
    j_best, best_epoch, best_state = math.inf, 0, None
    opt.zero_grad(set_to_none=True)
    for e in range(1, int(settings["max_epochs"]) + 1):
        net.train()
        order = rng.permutation(n)
        losses = []
        for s in range(0, n, bs):
            idx = torch.as_tensor(order[s:s + bs])
            loss = weighted_mse(net(u[idx], t), y[idx], w)
            if not torch.isfinite(loss):
                raise TrialFailed(f"non-finite training loss at epoch {e} (seed {seed})")
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(net.parameters(), float(clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        net.eval()
        j_e = _val_j(_net_raw(net, u_val, t), Y_val, k0, fidx)
        history.append((float(e), float(np.mean(losses)), j_e))
        if j_e < j_best - min_delta:
            j_best, best_epoch = j_e, e
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if e - best_epoch >= patience:
            break
    net.load_state_dict(best_state)
    return MemberResult(seed=int(seed), state=best_state, best_epoch=best_epoch, n_epochs=len(history),
                        j_val=float(j_best), history=np.array(history, dtype=np.float64),
                        fit_s=time.perf_counter() - t_start)


class SequenceProxy(ModelBase):
    """Seed ensemble of SeqNet members (section 2.8). RNNProxy and LSTMProxy subclass it."""

    extrapolates_time = True
    cells: tuple[str, ...] = ()

    def _check_cell(self, settings: dict) -> None:
        if settings.get("cell") not in self.cells:
            raise ContractError(f"{self.name}: cell {settings.get('cell')!r} is not one of {list(self.cells)}")
        if settings.get("input_option") not in OPTIONS:
            raise ContractError(f"{self.name}: input_option {settings.get('input_option')!r} is not A, B or C")

    def _fit(self, X, Y, cfg, val):
        if val is None:
            raise ContractError(f"{self.name}.fit needs val=(X_val, Y_val) for early stopping (section 1.3)")
        self._check_cell(cfg)
        members = [train_member(cfg, s, X, Y, val[0], val[1], self.k0) for s in cfg["seeds"]]
        self._set_members(members)

    @classmethod
    def from_members(cls, variant: str, settings: dict, members: list[MemberResult],
                     k0: int, T_fit: int, p: int) -> "SequenceProxy":
        m = cls()
        m.settings = dict(settings)
        m.k0, m.T_fit, m.p = int(k0), int(T_fit), int(p)
        if settings.get("input_option") != variant:
            raise ContractError(f"from_members: variant {variant!r} differs from input_option "
                                f"{settings.get('input_option')!r}")
        m._check_cell(m.settings)
        m._set_members(members)
        return m

    def _set_members(self, members: list[MemberResult]) -> None:
        if not members:
            raise ContractError(f"{self.name}: no ensemble member")
        self.variant = self.settings["input_option"]
        self.members_info = [{"seed": r.seed, "best_epoch": r.best_epoch, "n_epochs": r.n_epochs,
                              "j_val": r.j_val, "fit_s": r.fit_s} for r in members]
        self._set_states([r.seed for r in members], [r.state for r in members])

    def _set_states(self, seeds: list[int], states: list[dict]) -> None:
        self.seeds = [int(s) for s in seeds]
        self.states = states
        self.nets = []
        for st in states:
            net = build_net(self.settings, self.p)
            net.load_state_dict(st)
            net.eval()
            self.nets.append(net)

    def _chunks(self, X: np.ndarray, n_total: int):
        """Yields (row slice, (M, rows, n_total - k0, 3) float64 raw member outputs)."""
        t = _tensor(time_grid(self.k0, n_total, self.T_fit), self.settings)
        u_all = param_input(X, self.settings["x_affine"])
        step = int(self.settings["predict_batch"])
        for s in range(0, X.shape[0], step):
            u = _tensor(u_all[s:s + step], self.settings)
            yield slice(s, s + step), np.stack([_net_raw(net, u, t) for net in self.nets])

    def _raw_rollout(self, X, n_total):
        out = np.empty((X.shape[0], n_total - self.k0, len(FLUIDS)))
        for sl, G in self._chunks(X, n_total):
            out[sl] = G.mean(axis=0)
        return out

    def _raw_std(self, X):
        if len(self.nets) < 2:
            raise ContractError(f"{self.name}: predict_std needs at least 2 members, got {len(self.nets)}")
        out = np.empty((X.shape[0], self.T_fit - self.k0, len(FLUIDS)))
        for sl, G in self._chunks(X, self.T_fit):
            out[sl] = G.std(axis=0, ddof=1)
        return out

    def _meta_extra(self) -> dict:
        return {"members": getattr(self, "members_info", None)}

    def _save_artefacts(self, path: Path) -> list[str]:
        buf = io.BytesIO()
        torch.save({"seeds": self.seeds, "states": self.states}, buf)
        (path / "weights.pt").write_bytes(buf.getvalue())
        return ["weights.pt"]

    def _load_artefacts(self, path: Path, meta: dict) -> None:
        self._check_cell(self.settings)
        blob = torch.load(path / "weights.pt", weights_only=True)
        self.members_info = meta.get("members")
        self._set_states(blob["seeds"], blob["states"])
