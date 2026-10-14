"""Backtest of the proxies against persistence B2 (PRD section 9; spec task3 sections 1.5.3 and 2.8; D-026).

- t_cut = the number of history dates <= the cutoff; H = T - t_cut; horizon h = k - t_cut + 1.
- Each candidate proxy is retrained with `train --t-cut t_cut --settings-from <candidate run>`
  (the Task 2 settings, no tuning) and continued with its own `forecast` (V13).
- B2 repeats each case's simulated rate at t_cut - 1.
- V6 (B2): for oil and for water, the RMSE pooled over h in v6_horizons and the cases is
  strictly below B2's. BT = mean over oil and water of RMSE(1..H) / the split range.
"""

from __future__ import annotations

import copy
import json
import logging
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np

from dwc26.config import ConfigError, load_eval_config, load_model_config
from dwc26.data.check import ContractError, assert_array
from dwc26.data.load import FLUIDS
from dwc26.evaluation.store import B2, MetricRow, clean, fluid_rows

LOG = logging.getLogger("dwc26.task3")


def cutoff_index(dates: np.ndarray, cutoff) -> int:
    """t_cut = the number of dates <= cutoff."""
    dates = np.asarray(dates).astype("datetime64[D]")
    return int(np.count_nonzero(dates <= np.datetime64(cutoff, "D")))


def check_window(t_cut: int, T: int, k0: int, v6_horizons) -> int:
    """H = T - t_cut. ConfigError unless k0 + 3 <= t_cut and v6_horizons[1] <= H."""
    H = T - t_cut
    if t_cut < k0 + 3 or H < int(v6_horizons[1]):
        raise ConfigError(f"backtest: t_cut = {t_cut} and H = {H} must satisfy k0 + 3 = {k0 + 3} <= t_cut and "
                          f"v6_horizons[1] = {v6_horizons[1]} <= H (spec task3 section 3.1)")
    return H


def persistence(Y_phys: np.ndarray, t_cut: int, H: int) -> np.ndarray:
    """B2: (n, T, 3) -> (n, H, 3), each case's rate at t_cut - 1 repeated for h = 1..H."""
    Y_phys = np.asarray(Y_phys)
    assert_array("persistence Y_phys", Y_phys, (None, None, len(FLUIDS)), nonneg=True)
    if not 1 <= t_cut <= Y_phys.shape[1] or H < 1:
        raise ContractError(f"persistence: t_cut = {t_cut}, H = {H}, T = {Y_phys.shape[1]}")
    return np.repeat(Y_phys[:, t_cut - 1:t_cut, :], H, axis=1)


def backtest_errors(Y_true_h: np.ndarray, Y_pred_h: np.ndarray, cfg: dict, span: np.ndarray) -> list[MetricRow]:
    """bt_rmse (each h), bt_rmse_v6, bt_rmse_all (each fluid) and bt_score, as rows without
    proxy_id and split (the caller fills them). span (3,) = the split range of each fluid."""
    Y_true_h, Y_pred_h = np.asarray(Y_true_h), np.asarray(Y_pred_h)
    assert_array("backtest_errors Y_true_h", Y_true_h, (None, None, len(FLUIDS)), nonneg=True)
    assert_array("backtest_errors Y_pred_h", Y_pred_h, Y_true_h.shape, nonneg=True)
    span = np.asarray(span, dtype=np.float64)
    assert_array("backtest_errors span", span, (len(FLUIDS),))
    H = Y_true_h.shape[1]
    lo, hi = (int(x) for x in cfg["backtest"]["v6_horizons"])
    if H < hi:
        raise ContractError(f"backtest_errors: H = {H} < v6_horizons[1] = {hi}")
    d2 = (Y_pred_h - Y_true_h) ** 2
    per_h = np.sqrt(d2.mean(axis=0))                               # (H, 3)
    v6 = np.sqrt(d2[:, lo - 1:hi].mean(axis=(0, 1)))
    allh = np.sqrt(d2.mean(axis=(0, 1)))
    fidx = [FLUIDS.index(f) for f in cfg["metrics"]["objective_fluids"]]
    if not (span[fidx] > 0).all():
        raise ContractError("backtest_errors: the split range of an objective fluid is 0")
    score = float(np.mean(allh[fidx] / span[fidx]))
    rows = [MetricRow("", "", f, "bt_rmse", h + 1, float(per_h[h, j])) for h in range(H) for j, f in enumerate(FLUIDS)]
    rows += fluid_rows("", "", "bt_rmse_v6", v6) + fluid_rows("", "", "bt_rmse_all", allh)
    rows.append(MetricRow("", "", "all", "bt_score", -1, score))
    return rows


def backtest_split(bt_models: dict[str, object], arr, t_cut: int, H: int, cfg: dict) -> dict:
    """Backtest rows of the candidates and B2 on the cases of one split (val, or test after the freeze).
    bt_models: candidate proxy ID -> backtest model (T_fit = t_cut). Returns rows, rmse_h, v6, n_clipped."""
    T = arr.T
    if t_cut + H != T:
        raise ContractError(f"backtest_split: t_cut {t_cut} + H {H} != T {T}")
    Y_h = arr.Y_phys[:, t_cut:T]
    span = arr.Y_phys.max(axis=(0, 1)) - arr.Y_phys.min(axis=(0, 1))
    rows, rmse_h, n_clipped, v6_vals = [], {}, {}, {}

    def add(pid, F):
        rr = [replace(r, proxy_id=pid, split=arr.split) for r in backtest_errors(Y_h, F, cfg, span)]
        rows.extend(rr)
        rmse_h[pid] = np.array([[next(r.value for r in rr if r.metric == "bt_rmse" and r.idx == h + 1
                                      and r.fluid == f) for f in FLUIDS] for h in range(H)])
        v6_vals[pid] = {r.fluid: r.value for r in rr if r.metric == "bt_rmse_v6"}

    for pid, m in bt_models.items():
        if m.T_fit != t_cut:
            raise ContractError(f"backtest: the backtest model of {pid} has T_fit = {m.T_fit}, expected t_cut = {t_cut}")
        F = m.forecast(arr.X, H) * arr.y_factor
        n_clipped[pid] = int(m.last_n_clipped)
        add(pid, F)
        rows.append(MetricRow(pid, arr.split, "all", "bt_n_clipped", -1, float(n_clipped[pid])))
    add(B2, persistence(arr.Y_phys, t_cut, H))
    v6 = {}
    for pid in bt_models:
        v6[pid] = {f: bool(v6_vals[pid][f] < v6_vals[B2][f]) for f in cfg["metrics"]["objective_fluids"]}
        rows += [MetricRow(pid, arr.split, f, "bt_v6_pass", -1, clean(v)) for f, v in v6[pid].items()]
    return {"rows": rows, "rmse_h": rmse_h, "v6": v6, "v6_rmse": v6_vals, "n_clipped": n_clipped}


# --- run_backtest (section 1.5.3) -------------------------------------------------------------

def _diff_keys(a, b, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                out += _diff_keys(a[k], b[k], p)
        return out
    return [] if a == b else [prefix or "(root)"]


def check_model_config(path: Path, source_cfg: dict, cfg: dict) -> dict:
    """V7: the backtest model config must equal the source run's config, except data.t_cut.
    It must also use the database and the model directory of the evaluation config."""
    cur = json.loads(json.dumps(load_model_config(path)))
    a, b = copy.deepcopy(cur), copy.deepcopy(source_cfg)
    a.setdefault("data", {})["t_cut"] = None
    b.setdefault("data", {})["t_cut"] = None
    diff = _diff_keys(a, b)
    if diff:
        raise ConfigError(f"backtest: {Path(path).as_posix()} differs from the config of the source run at {diff}; "
                          "the backtest must use the Task 2 settings (V7)")
    for key in ("db", "models"):
        if Path(cur["paths"][key]).resolve() != Path(cfg["paths"][key]).resolve():
            raise ConfigError(f"backtest: paths.{key} of {Path(path).as_posix()} ({cur['paths'][key]}) differs from "
                              f"the evaluation config ({cfg['paths'][key]})")
    return cur


def _check_reuse(con, run_id: str, model: str, t_cut: int, source: str, sha: str) -> list[str]:
    row = con.execute("SELECT task, command, model, status, outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ConfigError(f"backtest.reuse.{model}: no run {run_id}")
    o = json.loads(row[4] or "{}")
    if (row[0], row[1], row[2], row[3]) != ("task2", "train", model, "ok") or o.get("t_cut") != t_cut \
            or o.get("settings_from") != source or o.get("arrays_sha256") != sha:
        raise ConfigError(f"backtest.reuse.{model}: run {run_id} must be an ok task2 train run of {model} with "
                          f"t_cut {t_cut}, settings_from {source} and arrays_sha256 {sha[:12]}...")
    return sorted(o.get("variants", {}))


def run_backtest(config_path: Path) -> str:
    """`python -m dwc26 backtest` (section 1.5.3). Validation cases only. Returns the run ID."""
    from dwc26.evaluation import report
    from dwc26.evaluation.runctx import task3_run
    from dwc26.evaluation.splits import load_split, train_reference
    from dwc26.evaluation.store import resolve_candidates, write_metrics
    from dwc26.models.registry import load_model
    from dwc26.models.train import torch_runtime, run_train

    config_path = Path(config_path)
    cfg = load_eval_config(config_path)
    bcfg = cfg["backtest"]
    with task3_run(cfg, config_path, "backtest") as ctx:
        con = ctx.con
        train = load_split(con, "train")                                          # 1. window
        ref = train_reference(train)
        del train
        val = load_split(con, "val")
        t_cut = cutoff_index(val.dates, bcfg["cutoff_date"])
        H = check_window(t_cut, val.T, ref.k0, bcfg["v6_horizons"])
        LOG.info("backtest window: cutoff %s, t_cut %d, H %d, k0 %d, T %d, n_val %d", bcfg["cutoff_date"], t_cut, H,
                 ref.k0, val.T, val.n)
        cands = resolve_candidates(con, cfg, val.arrays_sha256, models=list(bcfg["models"]))   # 2. source runs
        source_runs = {}
        for c in cands:
            source_runs.setdefault(c.model, c.run_id)
        src_variants = {m: sorted(c.variant for c in cands if c.model == m) for m in source_runs}
        LOG.info("source runs: %s", source_runs)
        bt_runs, to_train = {}, []
        for m in bcfg["models"]:                                                  # 3. config checks first
            src = source_runs[m]
            if m in bcfg["reuse"]:
                got = _check_reuse(con, bcfg["reuse"][m], m, t_cut, src, val.arrays_sha256)
                if got != src_variants[m]:
                    raise ConfigError(f"backtest.reuse.{m}: variants {got} differ from the source run {src_variants[m]}")
                bt_runs[m] = bcfg["reuse"][m]
                LOG.info("%s: reusing backtest run %s", m, bt_runs[m])
            else:
                src_cfg = json.loads(con.execute("SELECT config FROM runs WHERE run_id = ?", (src,)).fetchone()[0])
                check_model_config(Path(bcfg["model_configs"][m]), src_cfg, cfg)
                to_train.append(m)
        for m in to_train:                                                        # 3. train at the cutoff
            LOG.info("%s: train --t-cut %d --settings-from %s (%s)", m, t_cut, source_runs[m], bcfg["model_configs"][m])
            bt_runs[m] = run_train(Path(bcfg["model_configs"][m]), m, t_cut=t_cut, settings_from=source_runs[m])
            LOG.info("%s: backtest train run %s", m, bt_runs[m])
        mapping, bt_models = {}, {}
        models_dir = Path(cfg["paths"]["models"])
        neural = any(c.model in ("rnn", "lstm") for c in cands)
        rt = torch_runtime(cfg["runtime"]["torch_threads"]) if neural else nullcontext()
        with rt:
            for c in cands:                                                       # 4. errors
                bt_pid = f"{bt_runs[c.model]}/{c.variant}"
                mapping[c.proxy_id] = bt_pid
                bt_models[c.proxy_id] = load_model(bt_pid, models_dir, con=con)
            res = backtest_split(bt_models, val, t_cut, H, cfg)                   # 4 to 6
        for c in cands:
            LOG.info("%s (%s): bt_rmse_v6 oil %.2f water %.2f vs B2 oil %.2f water %.2f -> %s; clipped %d", c.label,
                     mapping[c.proxy_id], res["v6_rmse"][c.proxy_id]["oil"], res["v6_rmse"][c.proxy_id]["water"],
                     res["v6_rmse"][B2]["oil"], res["v6_rmse"][B2]["water"], res["v6"][c.proxy_id],
                     res["n_clipped"][c.proxy_id])
        figures = {}
        fig_dir = Path(cfg["paths"]["figures"])
        if cfg["report"]["plots"]:
            figures["backtest_val"] = (fig_dir / "backtest_val.png").as_posix()
        n_rows = write_metrics(con, ctx.run_id, res["rows"], T=val.T, H=H)           # 7. database
        durations = {m: con.execute("SELECT duration_s FROM runs WHERE run_id = ?", (r,)).fetchone()[0]
                     for m, r in bt_runs.items()}
        outputs = {
            "cutoff_date": bcfg["cutoff_date"], "t_cut": t_cut, "H": H, "v6_horizons": list(bcfg["v6_horizons"]),
            "source_runs": source_runs, "bt_runs": bt_runs, "bt_run_duration_s": durations,
            "reused": sorted(bcfg["reuse"]), "mapping": mapping, "v6": res["v6"],
            "arrays_sha256": val.arrays_sha256, "prepare_run_id": val.prepare_run_id, "k0": ref.k0, "T": val.T,
            "n_val": val.n, "candidates": [c.to_json() for c in cands], "n_metric_rows": n_rows,
            "files": {"figures": figures, "model_comparison": (Path(cfg["paths"]["reports"]) /
                                                               "model_comparison.md").as_posix()},
        }
        ctx.finish(outputs)
        if cfg["report"]["plots"]:                                                # R2: ctx.post
            from dwc26.evaluation.plots import plot_backtest, tab10_colors

            labels = {c.proxy_id: c.label for c in cands}
            order = [f"{m}/main" for m in cfg["candidates"]["runs"] if m not in bcfg["models"]] + [c.label for c in cands]
            ctx.post("backtest_val figure", lambda: plot_backtest(
                Path(figures["backtest_val"]), {labels[p]: v for p, v in res["rmse_h"].items() if p != B2},
                res["rmse_h"][B2], f"Backtest on the validation cases (t_cut {t_cut}, H {H})", cfg["report"]["dpi"],
                colors=tab10_colors(order)))
        ctx.post("model_comparison.md", lambda: report.write_model_comparison(con, cfg))
        LOG.info("backtest %s ok: %d metric rows", ctx.run_id, n_rows)
        return ctx.run_id

