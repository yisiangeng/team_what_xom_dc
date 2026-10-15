"""`python -m dwc26 evaluate --split {val,test}` (spec task3 sections 1.5.2, 1.5.5 and 2.2 to 2.7).

Validation: every Task 2 candidate (proxies, B0, B1) on the train and validation
cases, the n_fc-step validation continuation (P1, P2, P3, GOR), the coverage, the
mismatch metrics, the cost and the training flags. The recomputed predictions
must equal the stored Task 2 predictions, and J must equal the tuning objective.

Test: runs once after the freeze, behind the guards G1 to G7. The claim row is
committed before any test row is read (V3).
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.config import ConfigError, check_override_reason, load_eval_config
from our_work.proxy_model_sijie.src.dwc26.data import db
from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.evaluation.events import EVENTS, event_errors
from our_work.proxy_model_sijie.src.dwc26.evaluation.metrics import (coverage, local_errors, local_set, nrmse, objective_j, r2_case, r2_pooled,
                                      r2_scalar, rmse, rmse_by_date, spearman)
from our_work.proxy_model_sijie.src.dwc26.evaluation.plausibility import (continuation_checks, gor_check, max_epoch_flags,
                                           model_continuation)
from our_work.proxy_model_sijie.src.dwc26.evaluation.store import Candidate, MetricRow, clean, fluid_rows
from our_work.proxy_model_sijie.src.dwc26.hm.objective import MismatchSpec, mismatch

LOG = logging.getLogger("dwc26.task3")
NEURAL = ("rnn", "lstm")


@dataclass
class EvalResult:
    rows: list
    preds: dict                          # proxy_id -> (n, T, 3) physical, clipped
    stds: dict                           # proxy_id -> (n, T, 3) physical, or None
    conts: dict                          # proxies: proxy_id -> (n, T + n_fc, 3) physical rollout
    O_sim: np.ndarray | None             # (n,)
    O_prx: dict                          # proxy_id -> (n,)
    summary: dict                        # per proxy_id: j, P1, P2, P3, coverage flags (for runs.outputs)
    local: np.ndarray | None = None      # indices of the local cases


def _flag_text(v: float, target) -> str:
    lo, hi = target
    return "under" if v < lo else "over" if v > hi else "ok"


def evaluate_models(models: dict, cands: list[Candidate], arr, ref, spec: MismatchSpec | None, n_fc: int,
                    cfg: dict) -> EvalResult:
    """Pure computation (no DB, no files). spec is None for the train split: then only RMSE,
    NRMSE, J, the two R^2 and n_clipped_pred are computed."""
    split, T, s = arr.split, arr.T, arr.y_factor
    W = np.arange(T) >= ref.k0
    fidx = tuple(FLUIDS.index(f) for f in cfg["metrics"]["objective_fluids"])
    Ys = arr.Y_scaled
    pl, met = cfg["plausibility"], cfg["metrics"]
    rows, preds, stds, conts, O_prx, summary = [], {}, {}, {}, {}, {}
    O_sim = local = None
    if spec is not None:
        O_sim = mismatch(arr.Y_phys, spec)
        local = local_set(O_sim, arr.case_id, int(cfg["mismatch"]["local_n"]))
    for c in cands:
        m, pid = models[c.proxy_id], c.proxy_id
        P = m.predict(arr.X)
        n_clip = int(m.last_n_clipped)
        Pp = P * s
        S = m.predict_std(arr.X)
        Sp = None if S is None else S * s
        preds[pid], stds[pid] = Pp, Sp
        j = objective_j(Ys, P, fidx)                                       # = the Task 2 tuning objective
        rows += fluid_rows(pid, split, "rmse", rmse(arr.Y_phys, Pp))
        rows += fluid_rows(pid, split, "nrmse", nrmse(arr.Y_phys, Pp))
        rows += fluid_rows(pid, split, "r2_pooled", r2_pooled(arr.Y_phys, Pp, W))
        rows += fluid_rows(pid, split, "r2_case", r2_case(arr.Y_phys, Pp, ref.mean_curve, W))
        rows.append(MetricRow(pid, split, "all", "j", -1, j))
        rows.append(MetricRow(pid, split, "all", "n_clipped_pred", -1, float(n_clip)))
        summ = {"label": c.label, "j": j, "n_clipped_pred": n_clip}
        summary[pid] = summ
        if spec is None:
            continue
        if split == "val":                                                 # sigma_proxy for Task 4
            rbd = rmse_by_date(arr.Y_phys, Pp)
            rows += [MetricRow(pid, split, f, "rmse_by_date", k, float(rbd[k, jf]))
                     for k in range(T) for jf, f in enumerate(FLUIDS)]
        ev = event_errors(arr.Y_phys, Pp, arr.t_days, ref.k0, cfg["events"])
        for e in EVENTS:
            for stat, v in ev[e].items():
                rows.append(MetricRow(pid, split, "all", f"ev_{e}_{stat}", -1, clean(v)))
        Op = mismatch(Pp, spec)
        O_prx[pid] = Op
        lr, dO = local_errors(arr.Y_phys, Pp, spec.k_idx, local, O_sim, Op)
        rows += [MetricRow(pid, split, "all", "rho_O", -1, clean(spearman(Op, O_sim))),
                 MetricRow(pid, split, "all", "r2_O", -1, clean(r2_scalar(O_sim, Op))),
                 MetricRow(pid, split, "all", "local_dO_median", -1, dO)]
        rows += fluid_rows(pid, split, "local_rmse", lr)
        g = gor_check(Pp, ref.gor_range, float(pl["gor_oil_frac"]), float(pl["gor_rel_tol"]))
        rows += [MetricRow(pid, split, "gas", "gor_pred_min", -1, g["min"]),
                 MetricRow(pid, split, "gas", "gor_pred_max", -1, g["max"]),
                 MetricRow(pid, split, "gas", "gor_flag", -1, clean(g["flag"]))]
        summ["gor_flag"] = g["flag"]
        if not c.is_proxy:
            continue
        if Sp is None:
            raise ContractError(f"{pid}: a proxy must return predict_std")
        cov, width = coverage(arr.Y_phys, Pp, Sp, float(met["coverage_z"]), W)
        rows += fluid_rows(pid, split, "coverage", cov) + fluid_rows(pid, split, "interval_width", width)
        summ["coverage"] = {f: float(cov[jf]) for jf, f in enumerate(FLUIDS)}
        summ["coverage_flag"] = {f: _flag_text(cov[jf], met["coverage_target"]) for jf, f in enumerate(FLUIDS)}
        R, p1 = model_continuation(m, arr.X, T, n_fc, s)
        conts[pid] = R
        ch = continuation_checks(R, T, p1, pl, arr.Y_phys)
        rows += [MetricRow(pid, split, "all", "p1_n_clipped", -1, float(ch["p1_n_clipped"])),
                 MetricRow(pid, split, "oil", "p2_n_rising", -1, float(ch["p2_n_rising"])),
                 MetricRow(pid, split, "oil", "p2_max_ratio", -1, clean(ch["p2_max_ratio"])),
                 MetricRow(pid, split, "oil", "p3_R", -1, clean(ch["p3_R"])),
                 MetricRow(pid, split, "oil", "p3_n_used", -1, float(ch["p3_n_used"])),
                 MetricRow(pid, split, "oil", "p3_D_hist_true", -1, clean(ch["p3_D_hist_true"]))]
        gc = gor_check(R[:, T:], ref.gor_range, float(pl["gor_oil_frac"]), float(pl["gor_rel_tol"]))
        rows += [MetricRow(pid, split, "gas", "gor_cont_min", -1, gc["min"]),
                 MetricRow(pid, split, "gas", "gor_cont_max", -1, gc["max"]),
                 MetricRow(pid, split, "gas", "gor_cont_flag", -1, clean(gc["flag"]))]
        summ.update({"p1_n_clipped": ch["p1_n_clipped"], "p1_pass": ch["p1_pass"], "p2_n_rising": ch["p2_n_rising"],
                     "p2_max_ratio": ch["p2_max_ratio"], "p2_pass": ch["p2_pass"], "p3_R": ch["p3_R"],
                     "p3_n_used": ch["p3_n_used"], "p3_flag": ch["p3_flag"], "gor_cont_flag": gc["flag"]})
    return EvalResult(rows=rows, preds=preds, stds=stds, conts=conts, O_sim=O_sim, O_prx=O_prx, summary=summary,
                      local=local)


# --- cost and training flags (sections 2.6 and 2.7) ----------------------------------------------

def _run_outputs(con, run_id: str) -> tuple[dict, float | None]:
    row = con.execute("SELECT outputs, duration_s FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ContractError(f"no run {run_id}")
    return json.loads(row[0] or "{}"), row[1]


def _dmd_fit_s(run_dir: Path, best_trial: int) -> float:
    import csv

    with open(run_dir / "trials.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["trial"]) == int(best_trial):
                return float(row["fit_s"])
    raise ContractError(f"cost: trial {best_trial} is not in {run_dir / 'trials.csv'}")


def cost_and_flags(cand: Candidate, model, models_dir: Path, con, cfg: dict) -> list[MetricRow]:
    """cost_fit_s, cost_run_s, cost_predict_s (split 'none'), and for rnn and lstm the max-epoch counts."""
    from our_work.proxy_model_sijie.src.dwc26.models.base import read_meta

    out, dur = _run_outputs(con, cand.run_id)
    run_dir = Path(models_dir) / cand.run_id
    if cand.model in NEURAL:
        members = read_meta(run_dir / cand.variant).get("members") or []
        if not members:
            raise ContractError(f"cost: {cand.proxy_id} meta.json lists no members")
        fit_s = float(sum(float(m["fit_s"]) for m in members))
    elif cand.model == "dmd":
        fit_s = _dmd_fit_s(run_dir, out["variants"][cand.variant]["best_trial"])
    else:
        fit_s = float(dur)
    n_rows, repeats = int(cfg["cost"]["n_rows"]), int(cfg["cost"]["repeats"])
    X = np.random.default_rng(cfg["seed"]).random((n_rows, model.p))
    model.predict(X)                                                         # warm-up, untimed
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)
    rows = [MetricRow(cand.proxy_id, "none", "all", "cost_fit_s", -1, fit_s),
            MetricRow(cand.proxy_id, "none", "all", "cost_run_s", -1, float(dur)),
            MetricRow(cand.proxy_id, "none", "all", "cost_predict_s", -1, float(statistics.median(times)))]
    if cand.model in NEURAL:
        fl = max_epoch_flags(run_dir)
        rows += [MetricRow(cand.proxy_id, "none", "all", "n_trials_max_epochs", -1, float(fl["n_trials_max_epochs"])),
                 MetricRow(cand.proxy_id, "none", "all", "n_members_max_epochs", -1,
                           float(fl["n_members_max_epochs"].get(cand.variant, 0)))]
    return rows


def training_flags(con, cands: list[Candidate], models_dir: Path) -> dict:
    """{run_id: max_epoch_flags} of the neural candidate runs, with the selected trial of each variant."""
    out = {}
    for run_id in dict.fromkeys(c.run_id for c in cands if c.model in NEURAL):
        o, _ = _run_outputs(con, run_id)
        best = {v: vd.get("best_trial") for v, vd in o.get("variants", {}).items()}
        out[run_id] = max_epoch_flags(Path(models_dir) / run_id, best)
    return out


# --- run_evaluate --------------------------------------------------------------------------------

def _stored_predictions(con, run_id: str, variant: str, split: str, case_id: np.ndarray, T: int) -> np.ndarray | None:
    rows = con.execute("SELECT case_id, date_idx, oil, gas, water FROM predictions WHERE run_id = ? AND variant = ? "
                       "AND split = ? ORDER BY case_id, date_idx", (run_id, variant, split)).fetchall()
    if not rows:
        return None
    got = np.array([(r[0], r[1]) for r in rows], dtype=np.int64)
    want = np.array([(c, k) for c in case_id for k in range(T)], dtype=np.int64)
    if got.shape != want.shape or not np.array_equal(got, want):
        raise ContractError(f"consistency: stored {split} predictions of {run_id}/{variant} do not cover exactly the "
                            f"cases {case_id.tolist()} x {T} dates ({len(rows)} rows)")
    return np.array([r[2:5] for r in rows], dtype=np.float64).reshape(len(case_id), T, len(FLUIDS))


def _consistency(stored: np.ndarray, P_phys: np.ndarray, y_factor: np.ndarray) -> float:
    return float(np.max(np.abs(P_phys - stored) / y_factor))


def _setup(cfg: dict, val_like, ref) -> int:
    """n_fc and the run-time config checks that need the data (section 3.1)."""
    from our_work.proxy_model_sijie.src.dwc26.forecast.dates import quarters_between

    try:
        n_fc = quarters_between(val_like.dates[-1], np.datetime64(cfg["continuation"]["end_date"]))
    except ContractError as e:
        raise ConfigError(f"continuation.end_date: {e}") from e
    pl = cfg["plausibility"]
    if n_fc < int(pl["p3_fc_quarters"][1]):
        raise ConfigError(f"n_fc = {n_fc} < p3_fc_quarters[1] = {pl['p3_fc_quarters'][1]}")
    if not 1 <= int(pl["p3_hist_quarters"]) <= val_like.T - ref.k0 - 1:
        raise ConfigError(f"plausibility.p3_hist_quarters = {pl['p3_hist_quarters']} must lie in [1, T - k0 - 1] = "
                          f"[1, {val_like.T - ref.k0 - 1}]")
    if int(cfg["mismatch"]["local_n"]) > val_like.n:
        raise ConfigError(f"mismatch.local_n = {cfg['mismatch']['local_n']} > n = {val_like.n} cases of the split")
    return n_fc


def _mismatch_info(spec, arr, res: EvalResult, cfg: dict) -> dict:
    return {"alpha": cfg["mismatch"]["alpha"], "beta": cfg["mismatch"]["beta"], "fluids": list(cfg["mismatch"]["fluids"]),
            "K_n": int(spec.k_idx.size), "K_first": str(arr.dates[spec.k_idx[0]]),
            "K_last": str(arr.dates[spec.k_idx[-1]]),
            "sigma_obs": {f: [float(spec.sigma_obs[:, j].min()), float(spec.sigma_obs[:, j].max())]
                          for j, f in enumerate(cfg["mismatch"]["fluids"])},
            "O_sim": {str(int(c)): float(o) for c, o in zip(arr.case_id, res.O_sim)},
            "local_cases": [int(arr.case_id[i]) for i in res.local],
            "local_dO": {pid: [float(Op[i] - res.O_sim[i]) for i in res.local] for pid, Op in res.O_prx.items()}}


def _figures(cfg: dict, arr, res: EvalResult, cands: list[Candidate], n_fc: int, ref, curve_ids: list[str],
             tag: str) -> dict:
    """The curve and parity figures of one split (section 1.8)."""
    from our_work.proxy_model_sijie.src.dwc26.evaluation.plots import plot_curves, plot_parity, tab10_colors
    from our_work.proxy_model_sijie.src.dwc26.forecast.dates import forecast_dates

    fig_dir = Path(cfg["paths"]["figures"])
    dpi = int(cfg["report"]["dpi"])
    labels = {c.proxy_id: c.label for c in cands}
    colors = tab10_colors([c.label for c in cands])                     # the candidate order (section 1.8)
    files = {}
    fc_dates = forecast_dates(arr.dates[-1], n_fc)
    curves = {labels[p]: res.conts[p] for p in curve_ids if p in res.conts}
    for jf, f in enumerate(FLUIDS):
        path = fig_dir / f"curves_{tag}_{f}.png"
        plot_curves(path, arr.dates, fc_dates, arr.case_id, arr.Y_phys, arr.hist_phys, curves, jf,
                    f"{f.capitalize()} rate, {tag} cases: simulated, RAMP_History and proxies", dpi, colors=colors)
        files[f"curves_{tag}_{f}"] = path.as_posix()
    path = fig_dir / f"parity_{tag}.png"
    plot_parity(path, arr.Y_phys, {labels[c.proxy_id]: res.preds[c.proxy_id] for c in cands}, ref.k0,
                f"Parity, {tag} cases (k >= k0)", dpi, colors=colors)
    files[f"parity_{tag}"] = path.as_posix()
    return files


def run_evaluate(config_path: Path, split: str, *, override_reason: str | None = None) -> str:
    """`python -m dwc26 evaluate --split {val,test}`. Returns the run ID."""
    from our_work.proxy_model_sijie.src.dwc26.evaluation.runctx import task3_run

    config_path = Path(config_path)
    cfg = load_eval_config(config_path)
    if split not in ("val", "test"):
        raise ConfigError(f"evaluate: --split must be val or test, got {split!r}")
    if split == "val" and override_reason is not None:
        raise ConfigError("evaluate: --override-reason applies to --split test only")
    override_reason = check_override_reason(override_reason, cfg)
    with task3_run(cfg, config_path, "evaluate") as ctx:
        if split == "val":
            return _evaluate_val(ctx, cfg)
        return _evaluate_test(ctx, cfg, override_reason)


def _torch_ctx(cfg: dict, cands: list[Candidate]):
    if any(c.model in NEURAL for c in cands):
        from our_work.proxy_model_sijie.src.dwc26.models.train import torch_runtime

        return torch_runtime(int(cfg["runtime"]["torch_threads"]))
    return nullcontext()


def _evaluate_val(ctx, cfg: dict) -> str:
    from our_work.proxy_model_sijie.src.dwc26.evaluation import report
    from our_work.proxy_model_sijie.src.dwc26.evaluation.splits import load_split, train_reference
    from our_work.proxy_model_sijie.src.dwc26.evaluation.store import resolve_candidates, write_metrics
    from our_work.proxy_model_sijie.src.dwc26.hm.objective import build_mismatch_spec
    from our_work.proxy_model_sijie.src.dwc26.models.registry import load_model

    con = ctx.con
    train, val = load_split(con, "train"), load_split(con, "val")                   # 1. setup
    if train.arrays_sha256 != val.arrays_sha256:
        raise ContractError("evaluate: the train and validation arrays come from different Task 1 runs")
    ref = train_reference(train)
    n_fc = _setup(cfg, val, ref)
    spec = build_mismatch_spec(val.hist_phys, val.dates, cfg["mismatch"])
    cands = resolve_candidates(con, cfg, val.arrays_sha256)
    LOG.info("evaluate val: n_train %d, n_val %d, T %d, k0 %d, |K| %d, n_fc %d; candidates %s", train.n, val.n,
             val.T, ref.k0, spec.k_idx.size, n_fc, [c.proxy_id for c in cands])
    models_dir = Path(cfg["paths"]["models"])
    tol = float(cfg["consistency_rel_tol"])
    with _torch_ctx(cfg, cands):
        models = {c.proxy_id: load_model(c.proxy_id, models_dir, con=con, proxies_only=False) for c in cands}  # 2.
        res_tr = evaluate_models(models, cands, train, ref, None, n_fc, cfg)       # 4. metrics
        res_va = evaluate_models(models, cands, val, ref, spec, n_fc, cfg)
        consistency, cmax = {}, 0.0                                                # 3. consistency (V8)
        for c in cands:
            consistency[c.proxy_id] = {}
            for arr, res in ((train, res_tr), (val, res_va)):
                stored = _stored_predictions(con, c.run_id, c.variant, arr.split, arr.case_id, arr.T)
                if stored is None:
                    raise ContractError(f"consistency: no stored {arr.split} predictions of {c.proxy_id}")
                d = _consistency(stored, res.preds[c.proxy_id], arr.y_factor)
                consistency[c.proxy_id][arr.split] = d
                cmax = max(cmax, d)
                if d > tol:
                    raise ContractError(f"consistency: recomputed {arr.split} predictions of {c.proxy_id} differ from "
                                        f"the stored Task 2 predictions by {d:.3e} x y_factor > {tol:.1e}")
        LOG.info("consistency with the stored Task 2 predictions: max %.3e (tolerance %.1e)", cmax, tol)
        cost_rows = []
        for c in cands:
            cost_rows += cost_and_flags(c, models[c.proxy_id], models_dir, con, cfg)
    identity = {}                                                                  # 5. identity with Task 2
    for c in cands:
        out, _ = _run_outputs(con, c.run_id)
        vd = out["variants"][c.variant]
        j = res_va.summary[c.proxy_id]["j"]
        dj = abs(j - float(vd["j_val"]))
        rm = {r.fluid: r.value for r in res_va.rows if r.proxy_id == c.proxy_id and r.metric == "rmse"}
        drel = max(abs(rm[f] - float(vd["rmse_val"][f])) / max(abs(float(vd["rmse_val"][f])), 1e-300) for f in FLUIDS)
        identity[c.proxy_id] = {"j_abs_diff": dj, "rmse_rel_diff": drel}
        if dj > 1e-12 or drel > 1e-9:
            raise ContractError(f"identity: {c.proxy_id} J_val {j!r} vs Task 2 {vd['j_val']!r} (diff {dj:.2e}), RMSE "
                                f"relative diff {drel:.2e}; the ranking metric must equal the tuning objective")
    flags = training_flags(con, cands, models_dir)
    rows = res_tr.rows + res_va.rows + cost_rows
    reports = Path(cfg["paths"]["reports"])
    figs = {}
    if cfg["report"]["plots"]:
        fig_dir = Path(cfg["paths"]["figures"])
        figs = {f"curves_val_{f}": (fig_dir / f"curves_val_{f}.png").as_posix() for f in FLUIDS}
        figs["parity_val"] = (fig_dir / "parity_val.png").as_posix()
    n_rows = write_metrics(con, ctx.run_id, rows, T=val.T)                        # 6. database
    for c in cands:
        sm = res_va.summary[c.proxy_id]
        LOG.info("%s: J_val %.6f, P1 %s, P2 %s (%s rising, max ratio %s), P3 R %s flag %s, coverage %s", c.label,
                 sm["j"], sm.get("p1_pass"), sm.get("p2_pass"), sm.get("p2_n_rising"), sm.get("p2_max_ratio"),
                 sm.get("p3_R"), sm.get("p3_flag"), sm.get("coverage"))
    outputs = {
        "split": "val", "candidates": {c.model: c.run_id for c in cands},
        "candidate_list": [c.to_json() for c in cands], "arrays_sha256": val.arrays_sha256,
        "prepare_run_id": val.prepare_run_id, "k0": ref.k0, "T": val.T, "n_fc": n_fc,
        "n_cases": {"train": train.n, "val": val.n}, "case_ids": {"val": val.case_id.tolist()},
        "consistency": {"max": cmax, "tolerance": tol, "by_proxy": consistency}, "identity": identity,
        "summary": res_va.summary, "summary_train": res_tr.summary, "mismatch": _mismatch_info(spec, val, res_va, cfg),
        "gor_train_range": list(ref.gor_range), "training_flags": flags, "n_metric_rows": n_rows,
        "files": {"figures": figs, "model_comparison": (reports / "model_comparison.md").as_posix()},
    }
    ctx.finish(outputs)
    if cfg["report"]["plots"]:                                                     # 7. files (R2: ctx.post)
        ctx.post("validation figures",
                 lambda: _figures(cfg, val, res_va, cands, n_fc, ref, [c.proxy_id for c in cands if c.is_proxy], "val"))
    ctx.post("model_comparison.md", lambda: report.write_model_comparison(con, cfg))
    LOG.info("evaluate val %s ok: %d metric rows", ctx.run_id, n_rows)
    return ctx.run_id


def _evaluate_test(ctx, cfg: dict, override_reason: str | None) -> str:
    """Section 1.5.5: once, after the freeze, behind G1 to G7."""
    from our_work.proxy_model_sijie.src.dwc26.evaluation import report
    from our_work.proxy_model_sijie.src.dwc26.evaluation.guards import (GitOps, active_freeze, claim_test_scoring, finish_test_scoring,
                                         frozen_chain)
    from our_work.proxy_model_sijie.src.dwc26.evaluation.splits import load_split, train_reference
    from our_work.proxy_model_sijie.src.dwc26.evaluation.store import write_metrics
    from our_work.proxy_model_sijie.src.dwc26.forecast.backtest import backtest_split, check_window, cutoff_index
    from our_work.proxy_model_sijie.src.dwc26.hm.objective import build_mismatch_spec
    from our_work.proxy_model_sijie.src.dwc26.models.registry import load_model

    con = ctx.con
    repo = cfg["freeze"]["repo_dir"]
    git = GitOps(None if repo is None else Path(repo))
    ticket = claim_test_scoring(con, ctx.run_id, cfg, git, override_reason)        # 1. claim (G1 to G7)
    ctx.on_fail.append(lambda: finish_test_scoring(con, ticket, "failed"))
    if override_reason is not None:
        LOG.warning("OVERRIDE: test scoring %d with reason: %s", ticket.scoring_id, override_reason)
    LOG.info("test scoring claim %d (freeze %d, tag commit %s)", ticket.scoring_id, ticket.freeze_id, ticket.tag_commit)
    fr = active_freeze(con)                                                        # 2. lineage
    chain = frozen_chain(con, fr)
    S, E, B = chain["S"], chain["E"], chain["B"]
    cands = [Candidate.from_json(d) for d in E["outputs"]["candidate_list"]]
    mapping = B["outputs"]["mapping"]
    test = load_split(con, "test", ticket=ticket)                                  # 3. arrays
    train = load_split(con, "train")
    if not train.arrays_sha256 == test.arrays_sha256 == E["outputs"]["arrays_sha256"]:
        raise ContractError("evaluate test: the Task 1 arrays differ from those of the frozen evaluate run")
    ref = train_reference(train)
    del train
    n_fc = _setup(cfg, test, ref)
    spec = build_mismatch_spec(test.hist_phys, test.dates, cfg["mismatch"])
    t_cut = cutoff_index(test.dates, cfg["backtest"]["cutoff_date"])
    H = check_window(t_cut, test.T, ref.k0, cfg["backtest"]["v6_horizons"])
    if (t_cut, H) != (B["outputs"]["t_cut"], B["outputs"]["H"]):
        raise ContractError(f"evaluate test: t_cut {t_cut} and H {H} differ from the backtest run "
                            f"({B['outputs']['t_cut']}, {B['outputs']['H']})")
    models_dir = Path(cfg["paths"]["models"])
    LOG.info("evaluate test: n_test %d, candidates %s, backtest mapping %s", test.n, [c.proxy_id for c in cands],
             mapping)
    with _torch_ctx(cfg, cands):                                                   # 4. metrics
        models = {c.proxy_id: load_model(c.proxy_id, models_dir, con=con, proxies_only=False) for c in cands}
        res = evaluate_models(models, cands, test, ref, spec, n_fc, cfg)
        bt_models = {pid: load_model(bpid, models_dir, con=con) for pid, bpid in mapping.items()}
        bt = backtest_split(bt_models, test, t_cut, H, cfg)
    rows = res.rows + bt["rows"]
    n_pred, compared, cmax = 0, [], 0.0                                            # 5. test predictions
    tol = float(cfg["consistency_rel_tol"])
    for c in cands:
        stored = _stored_predictions(con, c.run_id, c.variant, "test", test.case_id, test.T)
        if stored is None:
            n_pred += db.write_predictions(con, c.run_id, c.variant, test.case_id, np.array(["test"] * test.n),
                                           test.dates, res.preds[c.proxy_id], res.stds[c.proxy_id], allow_test=True)
        else:
            d = _consistency(stored, res.preds[c.proxy_id], test.y_factor)
            cmax = max(cmax, d)
            compared.append(c.proxy_id)
            if d > tol:
                raise ContractError(f"evaluate test: recomputed test predictions of {c.proxy_id} differ from the "
                                    f"stored ones by {d:.3e} x y_factor")
    n_rows = write_metrics(con, ctx.run_id, rows, T=test.T, H=H)                  # 6. one transaction
    finish_test_scoring(con, ticket, "ok")
    ctx.on_fail.clear()
    reports = Path(cfg["paths"]["reports"])
    outputs = {
        "split": "test", "freeze_id": ticket.freeze_id, "scoring_id": ticket.scoring_id,
        "tag_commit": ticket.tag_commit, "override_reason": override_reason, "selection_run": S["run_id"],
        "evaluate_run": E["run_id"], "backtest_run": B["run_id"], "candidates": E["outputs"]["candidates"],
        "candidate_list": E["outputs"]["candidate_list"], "arrays_sha256": test.arrays_sha256,
        "prepare_run_id": test.prepare_run_id, "k0": ref.k0, "T": test.T, "n_fc": n_fc, "t_cut": t_cut, "H": H,
        "n_cases": {"test": test.n}, "case_ids": {"test": test.case_id.tolist()}, "summary": res.summary,
        "mismatch": _mismatch_info(spec, test, res, cfg), "v6": bt["v6"], "n_metric_rows": n_rows,
        "n_prediction_rows": n_pred, "predictions_compared": compared, "predictions_max_diff": cmax,
        "files": {"model_comparison": (reports / "model_comparison.md").as_posix()},
    }
    figs = {}
    if cfg["report"]["plots"]:
        fig_dir = Path(cfg["paths"]["figures"])
        figs = {f"curves_test_{f}": (fig_dir / f"curves_test_{f}.png").as_posix() for f in FLUIDS}
        figs["parity_test"] = (fig_dir / "parity_test.png").as_posix()
        figs["backtest_test"] = (fig_dir / "backtest_test.png").as_posix()
    outputs["files"]["figures"] = figs
    ctx.finish(outputs)
    if cfg["report"]["plots"]:                                                     # 7. files (R2: ctx.post)
        from our_work.proxy_model_sijie.src.dwc26.evaluation.plots import plot_backtest, tab10_colors
        from our_work.proxy_model_sijie.src.dwc26.forecast.backtest import B2

        chosen = [p for p in (fr["primary_id"], fr["secondary_id"]) if p]
        ctx.post("test curve and parity figures", lambda: _figures(cfg, test, res, cands, n_fc, ref, chosen, "test"))
        labels = {c.proxy_id: c.label for c in cands}
        ctx.post("backtest_test figure", lambda: plot_backtest(
            Path(figs["backtest_test"]), {labels[p]: v for p, v in bt["rmse_h"].items() if p != B2}, bt["rmse_h"][B2],
            f"Backtest on the test cases (t_cut {t_cut}, H {H})", int(cfg["report"]["dpi"]),
            colors=tab10_colors([c.label for c in cands])))
    ctx.post("model_comparison.md", lambda: report.write_model_comparison(con, cfg))
    LOG.info("evaluate test %s ok: %d metric rows, %d prediction rows", ctx.run_id, n_rows, n_pred)
    return ctx.run_id
