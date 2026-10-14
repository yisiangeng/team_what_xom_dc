"""Task 3 figures (spec task3 section 1.8; D-030). matplotlib with the Agg backend, imported lazily.

All values are in physical units. One PNG for each call; each figure is closed after
saving. Plotting never changes a metric: the panel annotations call the metric functions.
Colours follow the candidate order in matplotlib's tab10 cycle (pass `colors` to keep one
colour for each model across figures).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dwc26.data.load import FLUIDS
from dwc26.evaluation.metrics import r2_pooled, rmse

UNITS = {"oil": "STB/d", "gas": "MSCF/d", "water": "STB/d"}
NCOLS = 5


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def tab10_colors(labels: list[str]) -> dict[str, tuple]:
    """label -> tab10 colour, in the given (candidate) order."""
    plt = _pyplot()
    cmap = plt.get_cmap("tab10")
    return {lab: cmap(i % 10) for i, lab in enumerate(labels)}


def _save(fig, path: Path, dpi: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    _pyplot().close(fig)


def _dt(d: np.ndarray):
    return np.asarray(d).astype("datetime64[D]").astype("datetime64[ms]").astype(object)


def plot_curves(path: Path, dates: np.ndarray, fc_dates: np.ndarray, case_id: np.ndarray, Y_true: np.ndarray,
                hist: np.ndarray, curves: dict[str, np.ndarray], fluid: int, title: str, dpi: int, *,
                colors: dict | None = None) -> None:
    """One panel for each case (5 columns): the simulated curve (black), RAMP_History (grey dotted)
    and each proxy (history solid, continuation dashed after a vertical line at the last date).
    curves: label -> (n, T + n_fc, 3)."""
    plt = _pyplot()
    n, T = Y_true.shape[:2]
    colors = colors or tab10_colors(list(curves))
    nrows = max(1, math.ceil(n / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.2 * NCOLS, 2.4 * nrows), sharex=True, squeeze=False)
    xd, xf = _dt(dates), _dt(fc_dates)
    x_all = np.concatenate([xd, xf])
    for i in range(nrows * NCOLS):
        ax = axes[i // NCOLS][i % NCOLS]
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(xd, Y_true[i, :, fluid], color="black", lw=1.4, label="simulated")
        ax.plot(xd, hist[:, fluid], color="grey", ls=":", lw=1.2, label="RAMP_History")
        for lab, C in curves.items():
            c = colors.get(lab)
            ax.plot(xd, C[i, :T, fluid], color=c, lw=1.0, label=lab)
            if C.shape[1] > T:
                ax.plot(x_all[T - 1:C.shape[1]], C[i, T - 1:, fluid], color=c, lw=1.0, ls="--")
        ax.axvline(xd[-1], color="grey", lw=0.6)
        ax.set_title(f"case {int(case_id[i])}", fontsize=8)
        ax.tick_params(labelsize=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 9), fontsize=7)
    fig.suptitle(f"{title} ({UNITS[FLUIDS[fluid]]})", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save(fig, path, dpi)


def plot_parity(path: Path, Y_true: np.ndarray, preds: dict[str, np.ndarray], k0: int, title: str, dpi: int, *,
                colors: dict | None = None) -> None:
    """Rows = models, columns = fluids: predicted against simulated for k >= k0, the 1:1 line,
    and the RMSE (all dates) and R^2 pooled (k >= k0) in each panel."""
    plt = _pyplot()
    labels = list(preds)
    colors = colors or tab10_colors(labels)
    T = Y_true.shape[1]
    W = np.arange(T) >= k0
    fig, axes = plt.subplots(len(labels), len(FLUIDS), figsize=(3.0 * len(FLUIDS), 2.6 * len(labels)),
                             squeeze=False)
    for r, lab in enumerate(labels):
        P = preds[lab]
        err, r2 = rmse(Y_true, P), r2_pooled(Y_true, P, W)
        for jf, f in enumerate(FLUIDS):
            ax = axes[r][jf]
            yt, yp = Y_true[:, W, jf].ravel(), P[:, W, jf].ravel()
            ax.scatter(yt, yp, s=4, color=colors.get(lab), alpha=0.6)
            hi = float(max(yt.max(), yp.max(), 1e-12))
            ax.plot([0, hi], [0, hi], color="black", lw=0.7)
            ax.set_title(f"{lab}, {f}", fontsize=8)
            ax.text(0.03, 0.97, f"RMSE {err[jf]:.4g} {UNITS[f]}\nR² pooled {r2[jf]:.4g}", transform=ax.transAxes,
                    va="top", fontsize=7)
            ax.tick_params(labelsize=6)
            if r == len(labels) - 1:
                ax.set_xlabel(f"simulated ({UNITS[f]})", fontsize=7)
            if jf == 0:
                ax.set_ylabel("predicted", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, path, dpi)


def plot_backtest(path: Path, rmse_h: dict[str, np.ndarray], b2: np.ndarray, title: str, dpi: int, *,
                  colors: dict | None = None) -> None:
    """3 panels (fluids): RMSE against the horizon for each proxy, B2 dashed black.
    rmse_h: label -> (H, 3); b2 (H, 3)."""
    plt = _pyplot()
    colors = colors or tab10_colors(list(rmse_h))
    H = b2.shape[0]
    h = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, len(FLUIDS), figsize=(4.2 * len(FLUIDS), 3.4), squeeze=False)
    for jf, f in enumerate(FLUIDS):
        ax = axes[0][jf]
        for lab, R in rmse_h.items():
            ax.plot(h, R[:, jf], color=colors.get(lab), marker="o", ms=3, lw=1.0, label=lab)
        ax.plot(h, b2[:, jf], color="black", ls="--", lw=1.4, label="B2 persistence")
        ax.set_xticks(h)
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("horizon h (quarters after the cutoff)", fontsize=8)
        ax.set_ylabel(f"RMSE ({UNITS[f]})", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 9), fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save(fig, path, dpi)
