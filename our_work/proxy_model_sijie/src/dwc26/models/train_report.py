"""Diagnostics, speed benchmark and training report (spec task2 sections 1.7, 2.10, 2.15).

All errors in the report are in physical units (PRD section 5). The
benchmark comparison is information only; V6 is applied in Task 3.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.evaluation.metrics import nrmse, objective_j, rmse

PERIODS = ("plateau", "post_plateau", "pre_breakthrough", "post_breakthrough")
GOR_REL_TOL = 1e-9          # round-off allowance of the report-only GOR range check


# --- plateau and breakthrough (section 2.10) -----------------------------------------

def plateau_periods(Y_phys: np.ndarray, k0: int, plateau_frac: float,
                    breakthrough_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """(k_end, k_bt), each (n,) int: plateau end = first k >= k0 with q_oil < frac * max q_oil
    (T if none); breakthrough = first k with q_w > frac * max q_w (T if none)."""
    n, T = Y_phys.shape[0], Y_phys.shape[1]
    k_end = np.full(n, T, dtype=np.int64)
    k_bt = np.full(n, T, dtype=np.int64)
    for i in range(n):
        oil, wat = Y_phys[i, :, 0], Y_phys[i, :, 2]
        below = np.nonzero(oil[k0:] < plateau_frac * oil.max())[0]
        if below.size:
            k_end[i] = k0 + int(below[0])
        above = np.nonzero(wat > breakthrough_frac * wat.max())[0]
        if wat.max() > 0 and above.size:
            k_bt[i] = int(above[0])
    return k_end, k_bt


def period_masks(n: int, T: int, k0: int, k_end: np.ndarray, k_bt: np.ndarray) -> dict[str, np.ndarray]:
    k = np.arange(T)[None, :]
    ke, kb = k_end[:, None], k_bt[:, None]
    return {
        "plateau": (k >= k0) & (k < ke),
        "post_plateau": (k >= ke) & (k < T),
        "pre_breakthrough": (k >= k0) & (k < kb),
        "post_breakthrough": (k >= np.maximum(kb, k0)) & (k < T),
    }


def period_errors(Y_true: np.ndarray, Y_pred: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """RMSE (physical) of each fluid over the union of each period across the cases."""
    out = {}
    for name, m in masks.items():
        cnt = int(m.sum())
        if cnt == 0:
            out[name] = {"n": 0, **{f: None for f in FLUIDS}}
            continue
        d = (Y_pred - Y_true)[m]                                    # (cnt, 3)
        out[name] = {"n": cnt, **{f: float(np.sqrt(np.mean(d[:, j] ** 2))) for j, f in enumerate(FLUIDS)}}
    return out


def plateau_report(data, P_fit_phys, P_val_phys, cfg, k0: int, ls_fit_phys=None) -> dict:
    s = data.y_factor
    rep = cfg["report"]
    out = {}
    sets = [("train", data.Y_fit * s, P_fit_phys), ("val", data.Y_val * s, P_val_phys)]
    if ls_fit_phys is not None:
        sets.insert(0, ("ls_reconstruction_train", data.Y_fit * s, ls_fit_phys))
    for name, Yt, Yp in sets:
        k_end, k_bt = plateau_periods(Yt, k0, rep["plateau_frac"], rep["breakthrough_frac"])
        masks = period_masks(Yt.shape[0], Yt.shape[1], k0, k_end, k_bt)
        out[name] = {"errors": period_errors(Yt, Yp, masks), "k_end": k_end.tolist(), "k_bt": k_bt.tolist()}
    return out


def _fluid_dict(a) -> dict:
    return {f: float(a[j]) for j, f in enumerate(FLUIDS)}


# --- diagnostics of one variant --------------------------------------------------------

def diagnostics(model, data, cfg: dict, *, c7_range: tuple | None = None) -> dict:
    """Predictions, clip counts, metrics, coverage preview, continuation, GOR and
    plateau errors of one final model (spec section 1.4 step 6)."""
    s = data.y_factor
    k0, T = model.k0, model.T_fit
    fidx = tuple(FLUIDS.index(f) for f in cfg["objective"]["fluids"])
    rep = cfg["report"]
    P_fit = model.predict(data.X_fit)
    clip_fit = model.last_n_clipped
    P_val = model.predict(data.X_val)
    clip_val = model.last_n_clipped
    S_fit = model.predict_std(data.X_fit)
    S_val = model.predict_std(data.X_val)
    Yf, Yv = data.Y_fit * s, data.Y_val * s
    Pf, Pv = P_fit * s, P_val * s
    d = {
        "P_fit": P_fit, "P_val": P_val, "S_fit": S_fit, "S_val": S_val,
        "n_clipped": {"train": clip_fit, "val": clip_val, "forecast_val": None},
        "rmse_train": _fluid_dict(rmse(Yf, Pf)), "rmse_val": _fluid_dict(rmse(Yv, Pv)),
        "nrmse_train": _fluid_dict(nrmse(Yf, Pf)), "nrmse_val": _fluid_dict(nrmse(Yv, Pv)),
        "j_val": objective_j(data.Y_val, P_val, fidx),
        "n_val_pre_onset_positive": int((data.Y_val[:, :k0] > 0).sum()),
    }
    # coverage preview (k >= k0)
    if S_val is not None:
        z = rep["coverage_z"]
        err = np.abs(Yv - Pv)[:, k0:]
        sd = (S_val * s)[:, k0:]
        r_k = np.sqrt(np.mean(err ** 2, axis=(0, 1)))
        d["coverage_val"] = {f: float(np.mean(err[..., j] <= z * sd[..., j])) for j, f in enumerate(FLUIDS)}
        d["std_over_rmse_val"] = {f: (float(np.mean(sd[..., j]) / r_k[j]) if r_k[j] > 0 else None)
                                  for j, f in enumerate(FLUIDS)}
    else:
        d["coverage_val"] = None
        d["std_over_rmse_val"] = None
    # continuation check on the validation cases
    if model.extrapolates_time:
        steps = rep["forecast_steps"]
        F = model.forecast(data.X_val, steps)
        d["n_clipped"]["forecast_val"] = model.last_n_clipped
        Fp = F * s
        last = Pv[:, T - 1, 0]
        ratio = np.where(last[:, None] > 0, Fp[..., 0] / np.where(last[:, None] > 0, last[:, None], 1.0), np.nan)
        d["continuation"] = {
            "steps": steps,
            "min": _fluid_dict(Fp.min(axis=(0, 1))), "max": _fluid_dict(Fp.max(axis=(0, 1))),
            "n_cases_oil_rising": int((Fp[..., 0] > last[:, None]).any(axis=1).sum()),
            "max_oil_ratio": float(np.nanmax(ratio)) if np.isfinite(ratio).any() else None,
            "n_clipped": model.last_n_clipped,
        }
    else:
        d["continuation"] = None
    # GOR check (D-010)
    oil, gas = Pv[..., 0], Pv[..., 1]
    sel = oil > 0.01 * oil.max() if oil.max() > 0 else np.zeros_like(oil, dtype=bool)
    gor = gas[sel] / oil[sel]
    tf = Yf[..., 0] > 0
    tgor = Yf[..., 1][tf] / Yf[..., 0][tf]
    d["gor"] = {
        "pred_min": float(gor.min()) if gor.size else None, "pred_max": float(gor.max()) if gor.size else None,
        "n": int(gor.size), "train_min": float(tgor.min()) if tgor.size else None,
        "train_max": float(tgor.max()) if tgor.size else None,
        "c7_min": c7_range[0] if c7_range else None, "c7_max": c7_range[1] if c7_range else None,
    }
    ls = None
    if model.name == "dmd":
        ls = model.rollout_from_coef(model.coef_train, T) * s
    d["plateau"] = plateau_report(data, Pf, Pv, cfg, k0, ls)
    return d


# --- speed (section 2.15) ----------------------------------------------------------------

def speed_benchmark(model, cfg: dict) -> dict:
    sp = cfg["speed"]
    p = model.p
    Xb = np.random.default_rng(cfg["seed"]).random((sp["n_batch_rows"], p))
    t0 = time.perf_counter()
    model.predict(Xb)
    t_batch = time.perf_counter() - t0
    Xc = np.random.default_rng(cfg["seed"]).random((sp["call_rows"], p))
    for _ in range(5):
        model.predict(Xc)
    t0 = time.perf_counter()
    for _ in range(sp["n_calls"]):
        model.predict(Xc)
    t_call = (time.perf_counter() - t0) / sp["n_calls"]
    n_sets = sp["mcmc_calls"] * sp["call_rows"]
    batch_factor = n_sets / sp["n_batch_rows"]
    mcmc = sp["mcmc_calls"] * t_call
    batch = batch_factor * t_batch
    return {"predict_n_batch_rows_s": t_batch, "per_call_s": t_call, "mcmc_extrapolated_s": mcmc,
            "batch_extrapolated_s": batch, "batch_factor": batch_factor, "n_sets": n_sets,
            "limit_s": sp["limit_s"], "pass": bool(mcmc < sp["limit_s"] and batch < sp["limit_s"])}


# --- report ------------------------------------------------------------------------------

def _f(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.{nd - 1}e}"
    return f"{x:.{nd}g}" if abs(x) >= 1 else f"{x:.{nd}f}"


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) if isinstance(c, str) else _f(c) for c in r) + " |")
    return out


def _cplx_text(z) -> str:
    re_, im = z
    return f"{re_:.6f}" if im == 0 else f"{re_:.6f}{im:+.6f}i"


def write_train_report(path: Path, ctx: dict) -> None:
    """Writes report.md with the 12 sections of spec section 1.7."""
    h = ctx["header"]
    L = [f"# Training report: {h['run_id']}", ""]
    L += ["## 1. Header", ""]
    L += _table(["Item", "Value"], [
        ["Run ID", h["run_id"]], ["Model", h["model"]],
        ["Git commit", f"{h['git_commit']}{'+dirty' if h['git_dirty'] else ''}"],
        ["Start (UTC)", h["start"]], ["Config", f"{h['config_path']} (sha256 {h['config_sha256']})"],
        ["Prepare run", h["prepare_run_id"]], ["arrays_sha256", h["arrays_sha256"]],
        ["n_fit, n_val", f"{h['n_fit']}, {h['n_val']}"], ["T_fit, k0", f"{h['T_fit']}, {h['k0']}"],
        ["t_cut", str(h["t_cut"])], ["settings_from", str(h["settings_from"])],
        ["Seeds", str(h.get("seeds"))], ["tune_seeds", str(h.get("tune_seeds"))],
        ["torch_threads, n_jobs", f"{h.get('torch_threads')}, {h.get('n_jobs')}"],
        ["Duration to this report (s)", h["duration_s"]],
        ["Validation values before k0 that are > 0", h["n_val_pre_onset_positive"]],
        ["Quarter length (days), min and max", f"{h['dt_min']}, {h['dt_max']}"],
    ])
    L += ["", "## 2. Search space and plan", ""]
    space = ctx["space"]
    if space:
        L += _table(["Dimension", "Type", "Values or range"],
                    [[k, v["type"], json.dumps(v.get("values", [v.get("low"), v.get("high")]))]
                     for k, v in space.items()])
    else:
        L += ["No tuning (benchmark, or settings from an earlier run)."]
    L += ["", "Trials per variant: " + ", ".join(f"{v}: {n}" for v, n in ctx["plan_counts"].items()), ""]

    L += ["## 3. Trials", ""]
    trials = ctx["trials"]
    if trials:
        dims = list(space) if space else sorted({k for t in trials for k in t.settings})
        ok = sorted([t for t in trials if t.status == "ok"], key=lambda t: (t.j_val, t.trial))
        bad = [t for t in trials if t.status != "ok"]
        extra_cols = ([c for c in ("r_eff", "basis", "n_clip_cont", "plausible")]
                      if any("plausible" in t.extra for t in trials) else [])
        rows = []
        for t in ok + bad:
            ex = [t.extra.get(c, "") if not isinstance(t.extra.get(c), str) else t.extra[c] for c in extra_cols]
            rows.append([t.trial, t.variant, t.status] + [str(t.settings.get(dd)) for dd in dims]
                        + [t.j_val, *t.nrmse_val.tolist(), *ex, t.fit_s, t.error or ""])
        L += _table(["trial", "variant", "status", *dims, "J_val", "NRMSE oil", "NRMSE gas", "NRMSE water",
                     *extra_cols, "fit_s", "error"], rows)
    else:
        L += ["No trials."]
    L += [""]

    L += ["## 4. Variants", ""]
    for v, vd in ctx["variants"].items():
        L += [f"### {vd['proxy_id']}", ""]
        L += ["Settings: `" + json.dumps(vd["settings"], sort_keys=True) + "`", ""]
        L += [f"- Best trial: {vd['best_trial']}; J of the trial: {_f(vd['j_val_trial'])}; "
              f"J of the final model: {_f(vd['j_val'])}"]
        if vd.get("members"):
            L += ["- Members (seed: best epoch / epochs): " + ", ".join(
                f"{m['seed']}: {m['best_epoch']}/{m['n_epochs']}" for m in vd["members"])]
        L += [""]
        L += _table(["Split", "RMSE oil (STB/d)", "RMSE gas (MSCF/d)", "RMSE water (STB/d)", "NRMSE oil",
                     "NRMSE gas", "NRMSE water"],
                    [[sp, *[vd[f"rmse_{sp}"][f] for f in FLUIDS], *[vd[f"nrmse_{sp}"][f] for f in FLUIDS]]
                     for sp in ("train", "val")])
        L += [""]

    L += ["## 5. Benchmarks (information only; V6 is applied in Task 3)", ""]
    bench = ctx["benchmarks"]
    rows = []
    for b in ("b0", "b1"):
        info = bench.get(b)
        if info is None:
            rows.append([b, "missing: no ok run with the same arrays_sha256 and t_cut", "", ""])
        else:
            rows.append([b, info["run_id"], info["rmse_val"]["oil"], info["rmse_val"]["water"]])
    L += _table(["Benchmark", "Run", "Val RMSE oil", "Val RMSE water"], rows)
    L += [""]
    if all(bench.get(b) for b in ("b0", "b1")):
        rows = []
        for v, vd in ctx["variants"].items():
            lo = all(vd["rmse_val"]["oil"] < bench[b]["rmse_val"]["oil"] for b in ("b0", "b1"))
            lw = all(vd["rmse_val"]["water"] < bench[b]["rmse_val"]["water"] for b in ("b0", "b1"))
            rows.append([vd["proxy_id"], vd["rmse_val"]["oil"], vd["rmse_val"]["water"], lo, lw])
        L += _table(["Proxy ID", "Val RMSE oil", "Val RMSE water", "Oil below B0 and B1", "Water below B0 and B1"],
                    rows)
        L += [""]

    L += ["## 6. Clipping", ""]
    L += _table(["Variant", "Train", "Validation", "Validation continuation"],
                [[v, vd["n_clipped"]["train"], vd["n_clipped"]["val"], vd["n_clipped"]["forecast_val"]]
                 for v, vd in ctx["variants"].items()])
    L += [""]

    L += ["## 7. Uncertainty preview (validation, k >= k0)", ""]
    rows = []
    for v, vd in ctx["variants"].items():
        cv, so = vd["coverage_val"], vd.get("std_over_rmse_val")
        rows.append([v] + ([cv[f] for f in FLUIDS] + [so[f] for f in FLUIDS] if cv else ["n/a"] * 6))
    L += _table(["Variant", "Coverage oil", "Coverage gas", "Coverage water", "Mean std / RMSE oil",
                 "Mean std / RMSE gas", "Mean std / RMSE water"], rows)
    L += ["", f"Share of values with |y - y_hat| <= z std, z = {_f(ctx['coverage_z'], 6)}. "
          "Task 3 computes the official coverage.", ""]

    L += ["## 8. Continuation check on the validation cases", ""]
    rows = []
    for v, vd in ctx["variants"].items():
        c = vd["continuation"]
        if c is None:
            rows.append([v, "benchmark: no continuation"] + [""] * 9)
        else:
            rows.append([v, c["steps"], c["min"]["oil"], c["max"]["oil"], c["min"]["gas"], c["max"]["gas"],
                         c["min"]["water"], c["max"]["water"], c["n_cases_oil_rising"], c["max_oil_ratio"],
                         c["n_clipped"]])
    L += _table(["Variant", "Steps", "Oil min", "Oil max", "Gas min", "Gas max", "Water min", "Water max",
                 "Cases with rising oil", "Max oil ratio to T_fit-1", "Clipped"], rows)
    L += ["", "No new wells after the history (assumption): a rising oil rate is a warning, not an error.", ""]

    L += ["## 9. GOR check (D-010)", ""]
    rows = []
    for v, vd in ctx["variants"].items():
        g = vd["gor"]
        inside = (g["pred_min"] is not None and g["train_min"] is not None
                  and g["pred_min"] >= g["train_min"] * (1 - GOR_REL_TOL)
                  and g["pred_max"] <= g["train_max"] * (1 + GOR_REL_TOL))
        rows.append([v, g["pred_min"], g["pred_max"], g["n"], g["train_min"], g["train_max"], g["c7_min"],
                     g["c7_max"], inside])
    L += _table(["Variant", "Pred GOR min", "Pred GOR max", "n", "Train GOR min", "Train GOR max", "C7 min",
                 "C7 max", "Inside the train range"], rows)
    L += ["", "GOR in MSCF/STB on the validation predictions where the predicted oil exceeds 1% of its maximum. "
          "The C7 range covers all cases (Task 1 report); the train range covers the fit cases only.", ""]

    L += ["## 10. Plateau and breakthrough errors (section 2.10), RMSE in physical units", ""]
    for v, vd in ctx["variants"].items():
        rows = []
        for sname, sd in vd["plateau"].items():
            for per in PERIODS:
                e = sd["errors"][per]
                rows.append([sname, per, e["n"], e["oil"], e["gas"], e["water"]])
        L += [f"### {v}", ""]
        L += _table(["Set", "Period", "n values", "Oil", "Gas", "Water"], rows)
        L += [""]

    L += ["## 11. DMD", ""]
    dmd = [(v, vd) for v, vd in ctx["variants"].items() if vd.get("dmd_full")]
    if not dmd:
        L += ["Not a DMD run.", ""]
    for v, vd in dmd:
        d = vd["dmd_full"]
        ps = d.get("plausible_selected")
        L += [f"- Basis: {d['basis']} (A2, D-019); tuned rank {d['rank']}; r used {d['r']}; r_99 = {d['r_99']} "
              f"(rank for {d['energy_report']} of the energy, PRD §6, report only); q = {d['q']}",
              f"- Plausibility gate P1 (D-020): plausible_selected = {_f(ps)}; "
              f"{d.get('n_plausible_trials', 'n/a')} plausible grid point(s); clipped values in the selected "
              f"validation continuation: {d.get('n_clip_cont', 'n/a')}"]
        if ps is False:
            L += ["- **no plausible grid point; Task 3 rejects this proxy under P1**"]
        vp = d.get("varpro")
        if vp:
            L += [f"- VarPro: best start S{vp['best_start']}, cost {_f(vp['cost'], 8)}, status {vp['status']}"
                  f"{' (max_nfev reached)' if vp['status'] == 0 else ''}, nfev {vp['nfev']}; start costs "
                  + ", ".join(f"S{i} {_f(c, 8)}" for i, c in enumerate(vp["start_costs"]))
                  + "; d at bound: " + (", ".join(f"d{i + 1} {b}" for i, b in enumerate(vp["d_at_bound"]) if b)
                                        or "none")]
        L += [f"- r_eff = {d['r_eff']} ({d['n_real']} real, {d['n_pair']} pairs), n_stabilised = "
              f"{d['n_stabilised']}, n_merged = {d['n_merged']}, max |lambda| of exact DMD = "
              f"{_f(d['max_abs_lam_raw'], 8)}, cond(B) = {_f(d['cond_B'])}"
              + (f", unstable = {d['unstable']}{' (EXPERIMENT ONLY)' if d['unstable'] == 'keep' else ''}"
                 if d["basis"] == "exact" else "")
              + f", kernel = {d['kernel']}, gp_target = {d['gp_target']}, weights = {d['fluid_weights_name']}", ""]
        L += _table(["i", "sigma", "cumulative energy"],
                    [[i + 1, s, f"{e:.8f}" if e is not None else "n/a"]
                     for i, (s, e) in enumerate(zip(d["sigma"], d["energy_cum"] + [None] *
                                                    (len(d["sigma"]) - len(d["energy_cum"]))))])
        L += [""]

        def eig_row(z):
            mod, ang = abs(z), abs(math.atan2(z.imag, z.real))
            period = (2 * math.pi / ang) if ang > 0 else None
            half = (math.log(0.5) / math.log(mod)) if 0 < mod < 1 else None
            return [mod, ang, period, half]

        if d["basis"] == "exact":
            rows = [[_cplx_text(raw), _cplx_text(cl), *eig_row(complex(*cl))]
                    for raw, cl in zip(d["lam_raw"], d["lam_clean"])]
            L += _table(["lambda raw", "lambda cleaned", "|lambda|", "angle (rad)", "period (quarters)",
                         "half-life (quarters)"], rows)
        else:
            L += ["Exact DMD eigenvalues at the same rank (start S0, not used):", ""]
            L += _table(["lambda_exact", "|lambda|", "angle (rad)", "period (quarters)", "half-life (quarters)"],
                        [[_cplx_text(z), *eig_row(complex(*z))] for z in d["lam_exact"]])
            L += ["", "Eigenvalues used (real, in (0, 1], VarPro):", ""]
            L += _table(["j", "lambda", "decay rate -ln(lambda) per quarter", "half-life (quarters)"],
                        [[j + 1, f"{z[0]:.8f}", -math.log(z[0]), eig_row(complex(*z))[3]]
                         for j, z in enumerate(d["lam_used"])])
        L += ["", f"GP hyperparameters ({d['n_convergence_warnings']} convergence warnings):", ""]
        L += _table(["Fluid", "j", "C", "length scales", "noise", "log ML", "at bound"],
                    [[g["fluid"], g["j"], g["constant"], ", ".join(_f(x) for x in g["length_scale"]), g["noise"],
                      g["log_marginal_likelihood"], ", ".join(g["at_bound"]) or "-"] for g in d["gp_hyper"]])
        L += [""]

    L += ["## 12. Speed (section 2.15)", ""]
    rows = []
    for v, vd in ctx["variants"].items():
        sp = vd["speed"]
        if sp is None:
            rows.append([v, "benchmark: not timed"] + [""] * 5)
        else:
            rows.append([v, sp["predict_n_batch_rows_s"], sp["per_call_s"], sp["mcmc_extrapolated_s"],
                         sp["batch_extrapolated_s"], sp["limit_s"], "pass" if sp["pass"] else "FAIL"])
    L += _table(["Variant", "t_batch (s)", "t_call (s)", "MCMC extrapolated (s)", "Batch extrapolated (s)",
                 "Limit (s)", "Result"], rows)
    L += [""]
    Path(path).write_text("\n".join(L), encoding="utf-8")
