"""Plausibility checks of the continuation, GOR and max-epoch flags (spec task3 section 2.7;
D-020, D-021, D-022).

- P1: a clipped value in the continuation rejects the proxy (primary and secondary).
- P2: oil above p2_ratio x the proxy's own rate at T - 1, at any continuation step, in more
  than p2_case_share of the cases. Rejects under p2_mode 'reject'; report only under 'report'.
- P3 (report only): R = median of D_fc / D_hist, the mean log decline per quarter in the
  forecast quarters [a, b] over the last L_h history quarters; flagged if R < p3_ratio_min.
- GOR (report only): predicted gas / oil against the training range, with a relative tolerance.
- Max-epoch flags (report only): trials and members that stopped at max_epochs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS

OIL, GAS = FLUIDS.index("oil"), FLUIDS.index("gas")


def model_continuation(model, X: np.ndarray, T: int, n_fc: int, y_factor: np.ndarray) -> tuple[np.ndarray, int]:
    """(R, p1_n_clipped): R = rollout(X, T + n_fc) x y_factor, physical (n, T + n_fc, 3); the P1
    count is `last_n_clipped` right after forecast(X, n_fc), the steps T..T + n_fc - 1 only."""
    if model.T_fit != T:
        raise ContractError(f"model_continuation: the model has T_fit = {model.T_fit}, the split T = {T}")
    R = model.rollout(X, T + n_fc) * y_factor
    model.forecast(X, n_fc)
    return R, int(model.last_n_clipped)


def _decline(q: np.ndarray, k_from: int, k_to: int) -> np.ndarray:
    return (np.log(q[:, k_from]) - np.log(q[:, k_to])) / (k_to - k_from)


def continuation_checks(R: np.ndarray, T: int, p1_n_clipped: int, cfg_plaus: dict,
                        Y_true: np.ndarray | None = None) -> dict:
    """P1, P2 and P3 of one proxy on the cases of a split. R: (n, T + n_fc, 3) physical rollout.
    Y_true: (n, T, 3) simulated curves of the same cases, for p3_D_hist_true."""
    R = np.asarray(R)
    assert_array("continuation_checks R", R, (None, None, len(FLUIDS)), nonneg=True)
    n, total = R.shape[:2]
    n_fc = total - T
    L = int(cfg_plaus["p3_hist_quarters"])
    a, b = (int(x) for x in cfg_plaus["p3_fc_quarters"])
    if n < 1 or n_fc < b or T - 1 - L < 0:
        raise ContractError(f"continuation_checks: n = {n}, n_fc = {n_fc} (needs >= {b}), T - 1 - L_h = {T - 1 - L}")
    out = {"n": n, "p1_n_clipped": int(p1_n_clipped), "p1_pass": int(p1_n_clipped) == 0}

    oil = R[..., OIL]
    q_last = oil[:, T - 1]
    cont = oil[:, T:]
    peak = cont.max(axis=1)
    rising = np.where(q_last > 0, peak > float(cfg_plaus["p2_ratio"]) * q_last, peak > 0)
    pos = q_last > 0
    out["p2_n_rising"] = int(rising.sum())
    out["p2_max_ratio"] = float(np.max(peak[pos] / q_last[pos])) if pos.any() else None
    out["p2_pass"] = out["p2_n_rising"] <= float(cfg_plaus["p2_case_share"]) * n

    h0, h1, f0, f1 = T - 1 - L, T - 1, T - 2 + a, T - 1 + b
    four = oil[:, [h0, h1, f0, f1]]
    ok = (four > 0).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(oil > 0, oil, 1.0)
        d_hist = _decline(q, h0, h1)
        d_fc = _decline(q, f0, f1)
    used = ok & (d_hist >= float(cfg_plaus["p3_min_decline"]))
    out["p3_n_used"] = int(used.sum())
    out["p3_R"] = float(np.median(d_fc[used] / d_hist[used])) if used.any() else None
    out["p3_flag"] = None if out["p3_R"] is None else out["p3_R"] < float(cfg_plaus["p3_ratio_min"])
    out["p3_D_hist_true"] = None
    if Y_true is not None:
        Yt = np.asarray(Y_true)
        assert_array("continuation_checks Y_true", Yt, (n, T, len(FLUIDS)), nonneg=True)
        qt = Yt[..., OIL]
        okt = (qt[:, h0] > 0) & (qt[:, h1] > 0)
        if okt.any():
            out["p3_D_hist_true"] = float(np.median(_decline(qt[okt], h0, h1)))
    return out


def gor_range(Y_phys: np.ndarray) -> tuple[float, float]:
    """[min, max] of q_gas / q_oil over every case and date with q_oil > 0 (train cases, V1)."""
    Y_phys = np.asarray(Y_phys)
    assert_array("gor_range Y_phys", Y_phys, (None, None, len(FLUIDS)), nonneg=True)
    sel = Y_phys[..., OIL] > 0
    if not sel.any():
        raise ContractError("gor_range: no value with oil > 0")
    g = Y_phys[..., GAS][sel] / Y_phys[..., OIL][sel]
    return float(g.min()), float(g.max())


def gor_check(Y_pred: np.ndarray, g_range: tuple[float, float], oil_frac: float, rel_tol: float) -> dict:
    """{min, max, n, flag} of predicted gas / oil where the predicted oil exceeds oil_frac x its
    maximum. flag = min < g_min (1 - tol) or max > g_max (1 + tol); None values if nothing qualifies."""
    Y_pred = np.asarray(Y_pred)
    assert_array("gor_check Y_pred", Y_pred, (None, None, len(FLUIDS)), nonneg=True)
    oil, gas = Y_pred[..., OIL], Y_pred[..., GAS]
    top = oil.max()
    sel = oil > oil_frac * top if top > 0 else np.zeros(oil.shape, dtype=bool)
    if not sel.any():
        return {"min": None, "max": None, "n": 0, "flag": None}
    g = gas[sel] / oil[sel]
    lo, hi = float(g.min()), float(g.max())
    flag = lo < g_range[0] * (1 - rel_tol) or hi > g_range[1] * (1 + rel_tol)
    return {"min": lo, "max": hi, "n": int(sel.sum()), "flag": bool(flag)}


def _epochs(text: str) -> list[int]:
    return [int(x) for x in str(text).split(";") if str(x).strip() != ""]


def max_epoch_flags(run_dir: Path, best_trial: dict[str, int] | None = None) -> dict:
    """Trials and final members of a neural run that reached training.max_epochs (D-021 b).
    Reads trials.csv, config.yaml and each <variant>/meta.json of run_dir."""
    run_dir = Path(run_dir)
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        max_epochs = int(cfg["training"]["max_epochs"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        raise ContractError(f"max_epoch_flags: cannot read training.max_epochs from {run_dir / 'config.yaml'}: {e}") from e
    trials = []
    try:
        with open(run_dir / "trials.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ep = _epochs(row.get("n_epochs", ""))
                lr = row.get("lr")
                trials.append({"trial": int(row["trial"]), "variant": row["variant"],
                               "lr": float(lr) if lr not in (None, "") else None, "n_epochs": ep,
                               "flagged": any(e >= max_epochs for e in ep)})
    except (OSError, KeyError, ValueError) as e:
        raise ContractError(f"max_epoch_flags: cannot read {run_dir / 'trials.csv'}: {e}") from e
    members, n_members = {}, {}
    for meta_path in sorted(run_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        v = meta.get("variant", meta_path.parent.name)
        mem = [{"seed": m.get("seed"), "n_epochs": int(m["n_epochs"]), "flagged": int(m["n_epochs"]) >= max_epochs}
               for m in (meta.get("members") or [])]
        members[v] = mem
        n_members[v] = sum(m["flagged"] for m in mem)
    flagged = [t["trial"] for t in trials if t["flagged"]]
    selected = {}
    for v, b in (best_trial or {}).items():
        selected[v] = None if b is None else (int(b) in flagged)
    return {"max_epochs": max_epochs, "trials": trials, "flagged_trials": flagged,
            "n_trials_max_epochs": len(flagged), "members": members, "n_members_max_epochs": n_members,
            "selected_flagged": selected}
