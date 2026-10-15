"""`model_comparison.md` and `selection.md` (spec task3 section 1.7), rebuilt from the database.

Run chain: with a freeze row, the selection run of the active freeze row, its evaluate (E)
and backtest (B) runs, and the latest ok test scoring of that freeze. Otherwise the latest
ok select run and its E and B; with no select run, the latest ok E and B.
Numbers have 4 significant digits. RMSE is in STB/d (oil, water) or MSCF/d (gas).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.evaluation.events import EVENTS
from our_work.proxy_model_sijie.src.dwc26.evaluation.guards import active_freeze, run_record
from our_work.proxy_model_sijie.src.dwc26.evaluation.store import B2, Candidate, index_metrics, latest_ok_run, read_metrics

DASH = "—"
NOT_SCORED = "not scored"
UNIT = {"oil": "STB/d", "gas": "MSCF/d", "water": "STB/d"}
SECTIONS = ("## 1. Header", "## 2. All models and all splits", "## 3. V6 benchmarks", "## 4. Backtest",
            "## 5. Plausibility", "## 6. Event errors", "## 7. Mismatch", "## 8. Coverage", "## 9. Cost",
            "## 10. Training flags", "## 11. sigma_proxy for Task 4", "## 12. Clipping", "## 13. Figures")
SELECTION_PARTS = ("## 1. Header", "## 2. Steps", "## 3. Ranking trace", "## 4. Secondary", "## 5. Proposal",
                   "## 6. Caveats")


def g4(x) -> str:
    """4 significant digits; '—' for None."""
    if x is None:
        return DASH
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    x = float(x)
    if math.isnan(x):
        return DASH
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    if abs(x) >= 1e4:
        return f"{round(x, -int(math.floor(math.log10(abs(x)))) + 3):.0f}"
    if abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.4g}"


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(c if isinstance(c, str) else g4(c) for c in r) + " |")
    return out


def _flag(v) -> str:
    if v is None:
        return DASH
    return "FLAG" if float(v) == 1.0 else "ok"


def _pf(v: bool | None) -> str:
    return DASH if v is None else ("pass" if v else "FAIL")


# --- run chain ------------------------------------------------------------------------------------

def run_chain(con) -> dict:
    """{'freeze', 'S', 'E', 'B', 'T' (test run record or None), 'scorings'}."""
    fr = active_freeze(con)
    S = E = B = Tst = None
    if fr is not None:
        S = run_record(con, fr["selection_run"])
    else:
        s_id = latest_ok_run(con, "select")
        S = run_record(con, s_id) if s_id else None
    if S is not None:
        E = run_record(con, S["outputs"]["evaluate_run"])
        B = run_record(con, S["outputs"]["backtest_run"])
    else:
        e_id, b_id = latest_ok_run(con, "evaluate", where={"split": "val"}), latest_ok_run(con, "backtest")
        E = run_record(con, e_id) if e_id else None
        B = run_record(con, b_id) if b_id else None
    scorings = [dict(zip(("scoring_id", "run_id", "freeze_id", "tag_commit", "head_commit", "started", "finished",
                          "status", "override_reason"), r))
                for r in con.execute("SELECT scoring_id, run_id, freeze_id, tag_commit, head_commit, started, finished, "
                                     "status, override_reason FROM test_scoring ORDER BY scoring_id").fetchall()]
    if fr is not None:
        ok = [s for s in scorings if s["freeze_id"] == fr["freeze_id"] and s["status"] == "ok"]
        if ok:
            Tst = run_record(con, ok[-1]["run_id"])
    return {"freeze": fr, "S": S, "E": E, "B": B, "T": Tst, "scorings": scorings}


def _candidates(E: dict | None) -> list[Candidate]:
    return [Candidate.from_json(d) for d in E["outputs"]["candidate_list"]] if E else []


def _statuses(con, ch: dict, cands: list[Candidate], cfg: dict) -> dict:
    """proxy_id -> 'primary' | 'secondary' | 'eligible' | 'rejected: ...' | text."""
    out = {c.proxy_id: "benchmark" for c in cands if not c.is_proxy}
    if ch["E"] is None:
        return out
    scores = _scores(con, ch, cfg)
    roles = {}
    if ch["S"] is not None:
        roles = dict(con.execute("SELECT proxy_id, role FROM selection WHERE run_id = ?",
                                 (ch["S"]["run_id"],)).fetchall())
    for c in cands:
        if not c.is_proxy:
            continue
        s = scores.get(c.proxy_id) if scores else None
        if s is None:
            out[c.proxy_id] = "step 2 pending (no backtest run)"
            continue
        if roles.get(c.proxy_id):
            out[c.proxy_id] = roles[c.proxy_id]
        elif s.eligible:
            out[c.proxy_id] = "eligible"
        else:
            out[c.proxy_id] = "rejected: " + ", ".join(_fails(s, cfg["plausibility"]["p2_mode"]))
    return out


def _fails(s, p2_mode: str) -> list[str]:
    out = [f"step 1 ({f})" for f, ok in s.step1.items() if not ok]
    out += [f"step 2 ({f})" for f, ok in s.step2.items() if not ok]
    if not s.p1:
        out.append("P1")
    if not s.p2 and p2_mode == "reject":
        out.append("P2")
    return out


def _scores(con, ch: dict, cfg: dict) -> dict:
    from our_work.proxy_model_sijie.src.dwc26.evaluation.selection import build_scores

    if ch["E"] is None or ch["B"] is None:
        return {}
    e_rows = index_metrics(read_metrics(con, ch["E"]["run_id"], split="val"))
    b_rows = index_metrics(read_metrics(con, ch["B"]["run_id"], split="val"))
    scores, _ = build_scores(ch["E"], ch["B"], e_rows, b_rows, cfg)
    return {s.proxy_id: s for s in scores}


def _reason(s, p2_mode: str) -> str:
    d, parts = s.detail, []
    for f, ok in s.step1.items():
        if not ok:
            which = "B0" if d["b0"][f] <= d["b1"][f] else "B1"
            parts.append(f"{f}: {g4(d['rmse'][f])} >= {which} {g4(d['rmse_limit'][f])}")
    for f, ok in s.step2.items():
        if not ok:
            parts.append(f"backtest {f}: {g4(d['bt_v6'][f])} >= B2 {g4(d['b2_v6'][f])}")
    if not s.p1:
        parts.append(f"P1: {g4(d['p1_n_clipped'])} clipped continuation values")
    if not s.p2:
        parts.append(f"P2{'' if p2_mode == 'reject' else ' (report only)'}: {g4(d['p2_n_rising'])} of {d['n_val']} cases "
                     f"rise, max ratio {g4(d['p2_max_ratio'])}")
    return "; ".join(parts) or DASH


# --- model_comparison.md --------------------------------------------------------------------------

def _get(idx: dict | None, pid: str, split: str, fluid: str, metric: str, k: int = -1):
    if idx is None:
        return None
    return idx.get((pid, split, fluid, metric, k))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_model_comparison(con, cfg: dict) -> Path:
    """Writes outputs/reports/model_comparison.md with the 13 sections of spec section 1.7.1."""
    ch = run_chain(con)
    E, B, S, Tst, fr = ch["E"], ch["B"], ch["S"], ch["T"], ch["freeze"]
    cands = _candidates(E)
    proxies = [c for c in cands if c.is_proxy]
    lab = {c.proxy_id: c.label for c in cands}
    ei = index_metrics(read_metrics(con, E["run_id"])) if E else None
    bi = index_metrics(read_metrics(con, B["run_id"])) if B else None
    ti = index_metrics(read_metrics(con, Tst["run_id"])) if Tst else None
    eo = E["outputs"] if E else {}
    bo = B["outputs"] if B else {}
    fl = cfg["metrics"]["objective_fluids"]
    L = ["# Model comparison (Task 3)", ""]

    # 1. header
    L += [SECTIONS[0], ""]
    rows = []
    for name, rec in (("select", S), ("evaluate (val)", E), ("backtest", B), ("evaluate (test)", Tst)):
        rows.append([name, rec["run_id"] if rec else "none",
                     f"{rec['git_commit']}{'+dirty' if rec['git_dirty'] else ''}" if rec else DASH])
    L += _table(["Run", "Run ID", "Git commit"], rows)
    L += [""]
    if E:
        K = eo.get("mismatch", {})
        L += [f"- Prepare run {eo.get('prepare_run_id')}, arrays_sha256 `{eo.get('arrays_sha256')}`",
              f"- Config {E['config_path']} (sha256 `{E['config_sha256']}`)",
              f"- k0 = {eo.get('k0')}, T = {eo.get('T')}, |K| = {K.get('K_n')} ({K.get('K_first')} to {K.get('K_last')}), "
              f"n_fc = {eo.get('n_fc')}, t_cut = {bo.get('t_cut', DASH)}, H = {bo.get('H', DASH)}",
              f"- Cases: train {eo.get('n_cases', {}).get('train')}, validation {eo.get('n_cases', {}).get('val')}, "
              f"test {Tst['outputs']['n_cases']['test'] if Tst else NOT_SCORED}"]
    else:
        L += ["- No ok `evaluate --split val` run yet."]
    if fr is None:
        L += ["- Freeze: not frozen"]
    else:
        L += [f"- Freeze: freeze ID {fr['freeze_id']}, tag `{fr['tag']}`, tag commit {fr['tag_commit']}; primary "
              f"{fr['primary_id']}, secondary {fr['secondary_id']}"
              + (f"; OVERRIDE: {fr['override_reason']}" if fr["override_reason"] else "")]
    if not ch["scorings"]:
        L += ["- Test: test split not scored (V3)"]
    else:
        ok = [s for s in ch["scorings"] if s["status"] == "ok"]
        if Tst:
            L += [f"- Test: scored once by run {Tst['run_id']} at commit {Tst['git_commit']}"
                  + (" (with overrides, below)" if len(ch["scorings"]) > 1 else "")]
        elif not ok:
            L += ["- Test: test split not scored (V3); attempts are listed below"]
        for s in ch["scorings"]:
            L += [f"  - scoring {s['scoring_id']}: run {s['run_id']}, status {s['status']}, freeze {s['freeze_id']}, "
                  f"tag commit {s['tag_commit'][:12]}" + (f", override reason: {s['override_reason']}"
                                                         if s["override_reason"] else "")]
    L += [""]
    if cands:
        L += ["Legend: " + "; ".join(f"`{c.label}` = {c.proxy_id}" for c in cands), ""]

    # 2. table 1
    L += [SECTIONS[1], ""]
    status = _statuses(con, ch, cands, cfg) if E else {}
    lo, hi = cfg["backtest"]["v6_horizons"]
    head = ["Model", "Split", "J", "NRMSE oil", "NRMSE water", "RMSE oil", "RMSE gas", "RMSE water", "R² pooled oil",
            "R² pooled water", "R² case oil", "R² case water", "ρ", "local RMSE oil", "local RMSE water",
            "coverage oil", "coverage water", f"BT RMSE h {lo}–{hi} oil", f"BT RMSE h {lo}–{hi} water", "Status"]
    rows = []
    for c in cands:
        p = c.proxy_id
        for split in ("train", "val", "test"):
            if split == "test" and ti is None:
                rows.append([c.label, split] + [NOT_SCORED] * (len(head) - 3) + [DASH])
                continue
            idx = ti if split == "test" else ei
            bidx = ti if split == "test" else bi
            r = [c.label, split, _get(idx, p, split, "all", "j")]
            r += [_get(idx, p, split, f, "nrmse") for f in fl]
            r += [_get(idx, p, split, f, "rmse") for f in FLUIDS]
            r += [_get(idx, p, split, f, "r2_pooled") for f in fl]
            r += [_get(idx, p, split, f, "r2_case") for f in fl]
            if split == "train":
                r += [DASH] * 5
            else:
                r += [_get(idx, p, split, "all", "rho_O")] + [_get(idx, p, split, f, "local_rmse") for f in fl]
                r += [_get(idx, p, split, f, "coverage") for f in fl] if c.is_proxy else [DASH, DASH]
            if split == "train" or not c.is_proxy:
                r += [DASH, DASH]
            else:
                r += [_get(bidx, p, split, f, "bt_rmse_v6") if bidx else "not run" for f in fl]
            r.append(status.get(p, DASH) if split == "val" else DASH)
            rows.append(r)
    L += _table(head, rows) if rows else ["No candidates."]
    L += ["", "Validation decides (PRD section 7); test rows are report only. ρ is NULL (—) for B0, whose curve "
          "is the same for every case.", ""]

    # 3. V6
    L += [SECTIONS[2], ""]
    rows = []
    bench = {c.model: c.proxy_id for c in cands if not c.is_proxy}
    for split, idx, bidx in (("val", ei, bi), ("test", ti, ti)):
        if idx is None:
            continue
        for c in proxies:
            for f in fl:
                rm = _get(idx, c.proxy_id, split, f, "rmse")
                b0, b1 = (_get(idx, bench.get(b), split, f, "rmse") for b in ("b0", "b1"))
                s1 = None if None in (rm, b0, b1) else rm < min(b0, b1)
                bt = _get(bidx, c.proxy_id, split, f, "bt_rmse_v6") if bidx else None
                b2 = _get(bidx, B2, split, f, "bt_rmse_v6") if bidx else None
                s2 = None if None in (bt, b2) else bt < b2
                rows.append([c.label, split, f, rm, b0, b1, _pf(s1), bt if bt is not None else "not run",
                             b2 if b2 is not None else "not run", _pf(s2)])
    L += _table(["Proxy", "Split", "Fluid", "RMSE", "B0", "B1", "Step 1", f"BT RMSE h {lo}–{hi}", "B2", "Step 2"],
                rows) if rows else ["No proxy metrics yet."]
    L += ["", "Validation decides; test rows are report only.", ""]

    # 4. backtest
    L += [SECTIONS[3], ""]
    for split, bidx in (("val", bi), ("test", ti)):
        if bidx is None:
            L += [f"Backtest on the {split} cases: " + ("not run." if split == "val" else "after the freeze only."), ""]
            continue
        H = int(bo.get("H") or (Tst["outputs"]["H"] if Tst else 0))
        L += [f"### {split}", ""]
        for f in FLUIDS:
            rows = []
            for pid, name in [(c.proxy_id, c.label) for c in proxies] + [(B2, "B2 persistence")]:
                rows.append([name] + [_get(bidx, pid, split, f, "bt_rmse", h) for h in range(1, H + 1)])
            L += [f"RMSE by horizon, {f} ({UNIT[f]}):", ""]
            L += _table(["Model"] + [f"h {h}" for h in range(1, H + 1)], rows) + [""]
        rows = [[name] + [_get(bidx, pid, split, f, "bt_rmse_all") for f in FLUIDS]
                + [_get(bidx, pid, split, "all", "bt_score"),
                   _get(bidx, pid, split, "all", "bt_n_clipped") if pid != B2 else DASH]
                for pid, name in [(c.proxy_id, c.label) for c in proxies] + [(B2, "B2 persistence")]]
        L += _table(["Model", "RMSE h 1..H oil", "RMSE h 1..H gas", "RMSE h 1..H water", "BT", "clipped"], rows)
        L += [""]
    if bo:
        L += [f"Backtest train runs: " + ", ".join(f"{m} {r} ({g4(bo.get('bt_run_duration_s', {}).get(m))} s)"
                                                 for m, r in bo.get("bt_runs", {}).items()), ""]
    L += ["The backtest models reuse the Task 2 settings, which were tuned with full-window validation data "
          "(task2 section 2.13), so the backtest hyperparameters are not blind to the dates after the cutoff. "
          f"The backtest covers {bo.get('H', 'H')} quarters and the forecast {eo.get('n_fc', 'n_fc')}: it is evidence "
          "for the method, not proof of the long-range accuracy (PRD section 9).", ""]

    # 5. plausibility
    L += [SECTIONS[4], ""]
    gtr = eo.get("gor_train_range") or [None, None]
    for split, idx in (("val", ei), ("test", ti)):
        if idx is None or not proxies:
            continue
        n = (eo.get("n_cases", {}).get("val") if split == "val" else Tst["outputs"]["n_cases"]["test"])
        rows = []
        for c in proxies:
            p = c.proxy_id
            nr = _get(idx, p, split, "oil", "p2_n_rising")
            p2 = None if nr is None else nr <= cfg["plausibility"]["p2_case_share"] * n
            pr = _get(idx, p, split, "oil", "p3_R")
            rows.append([c.label, split, _get(idx, p, split, "all", "p1_n_clipped"),
                         _pf(None if _get(idx, p, split, "all", "p1_n_clipped") is None
                             else _get(idx, p, split, "all", "p1_n_clipped") == 0),
                         f"{g4(nr)} of {n}", _get(idx, p, split, "oil", "p2_max_ratio"), _pf(p2), pr,
                         _get(idx, p, split, "oil", "p3_n_used"),
                         DASH if pr is None else ("FLAG" if pr < cfg["plausibility"]["p3_ratio_min"] else "ok"),
                         _get(idx, p, split, "oil", "p3_D_hist_true")])
        L += _table(["Proxy", "Split", "P1 clipped", "P1", "P2 rising cases", "P2 max ratio", "P2", "P3 R",
                     "P3 cases used", "P3", "median true D_hist"], rows) + [""]
        rows = []
        for c in cands:
            p = c.proxy_id
            rows.append([c.label, split, _get(idx, p, split, "gas", "gor_pred_min"),
                         _get(idx, p, split, "gas", "gor_pred_max"), _flag(_get(idx, p, split, "gas", "gor_flag")),
                         _get(idx, p, split, "gas", "gor_cont_min") if c.is_proxy else DASH,
                         _get(idx, p, split, "gas", "gor_cont_max") if c.is_proxy else DASH,
                         _flag(_get(idx, p, split, "gas", "gor_cont_flag")) if c.is_proxy else DASH])
        L += [f"GOR (MSCF/STB) against the train range [{g4(gtr[0])}, {g4(gtr[1])}], tolerance "
              f"{cfg['plausibility']['gor_rel_tol']}:", ""]
        L += _table(["Model", "Split", "history min", "history max", "history", "continuation min",
                     "continuation max", "continuation"], rows) + [""]
    L += ["D-031: the user confirmed that there are no new wells and no development after the history period. "
          "P1 and P2 (D-020) rest on that answer; a proxy that fails P2 is not eligible. "
          f"p2_mode = {cfg['plausibility']['p2_mode']}.",
          "D-022: P3 is report only. A DMD proxy is a candidate for secondary only, and its flat forecast tail must be "
          "stated wherever its forecast measures the method uncertainty (Task 5).", ""]

    # 6. events
    L += [SECTIONS[5], ""]
    rows = []
    for split, idx in (("val", ei), ("test", ti)):
        if idx is None:
            continue
        for c in cands:
            for e in EVENTS:
                rows.append([c.label, split, e] + [_get(idx, c.proxy_id, split, "all", f"ev_{e}_{s}")
                                                   for s in ("n_both", "n_mismatch", "n_none", "mae_days",
                                                             "bias_days")])
    L += _table(["Model", "Split", "Event", "n_both", "n_mismatch", "n_none", "MAE (days)", "bias (days)"], rows) \
        if rows else ["No event metrics yet."]
    L += ["", "An event in one curve only is counted (n_mismatch), not scored. Resolution: one quarter.", ""]

    # 7. mismatch
    L += [SECTIONS[6], ""]
    for split, rec, idx in (("val", E, ei), ("test", Tst, ti)):
        if rec is None:
            continue
        mm = rec["outputs"].get("mismatch", {})
        L += [f"### {split}", "",
              f"|K| = {mm.get('K_n')} ({mm.get('K_first')} to {mm.get('K_last')}), alpha = {mm.get('alpha')}, "
              f"beta = {mm.get('beta')}, fluids {mm.get('fluids')}. sigma_obs: "
              + ", ".join(f"{f} {g4(v[0])} to {g4(v[1])}" for f, v in mm.get("sigma_obs", {}).items()), ""]
        local = set(mm.get("local_cases", []))
        osim = sorted(((float(v), int(k)) for k, v in mm.get("O_sim", {}).items()))
        L += _table(["Case", "O_sim", "Local"], [[str(k), v, "local" if k in local else ""] for v, k in osim]) + [""]
        rows = [[c.label, _get(idx, c.proxy_id, split, "all", "rho_O"), _get(idx, c.proxy_id, split, "all", "r2_O")]
                + [_get(idx, c.proxy_id, split, f, "local_rmse") for f in FLUIDS]
                + [_get(idx, c.proxy_id, split, "all", "local_dO_median")] for c in cands]
        L += _table(["Model", "ρ", "R²_O", "local RMSE oil", "local RMSE gas", "local RMSE water",
                     "median O_prx − O_sim (local)"], rows) + [""]
    n_val = eo.get("n_cases", {}).get("val")
    se = f"{1 / math.sqrt(n_val - 1):.2f}" if n_val and n_val > 1 else DASH
    L += [f"O is O_obs (sigma_proxy = 0) for every model (D-025). ρ rests on the cases of one split; with {n_val} "
          f"validation cases its standard error near 0 is about 1/sqrt(n - 1) = {se}, so it is a tie-break only.", ""]

    # 8. coverage
    L += [SECTIONS[7], ""]
    lo_c, hi_c = cfg["metrics"]["coverage_target"]
    rows = []
    for split, idx in (("val", ei), ("test", ti)):
        if idx is None:
            continue
        for c in proxies:
            for f in FLUIDS:
                cv = _get(idx, c.proxy_id, split, f, "coverage")
                flag = DASH if cv is None else "under" if cv < lo_c else "over" if cv > hi_c else "ok"
                rows.append([c.label, split, f, cv, _get(idx, c.proxy_id, split, f, "interval_width"), flag])
    L += _table(["Proxy", "Split", "Fluid", "95% coverage (k >= k0)", "mean width", "flag"], rows) \
        if rows else ["No coverage yet."]
    L += ["", f"Target {lo_c} to {hi_c}. D-021: no recalibration, because Task 4's sigma_proxy (the validation RMSE) "
          "and its std filter (relative to the validation cases) do not depend on a global variance scale. The seed spread "
          "captures only the optimisation spread (Lakshminarayanan et al. 2017).", ""]

    # 9. cost
    L += [SECTIONS[8], ""]
    rows = []
    for c in cands:
        sp = _train_variant(con, c).get("speed") or {}
        rows.append([c.label, _get(ei, c.proxy_id, "none", "all", "cost_fit_s"),
                     _get(ei, c.proxy_id, "none", "all", "cost_run_s"),
                     _get(ei, c.proxy_id, "none", "all", "cost_predict_s"),
                     sp.get("mcmc_extrapolated_s") if sp else DASH, sp.get("limit_s") if sp else DASH,
                     ("pass" if sp.get("pass") else "FAIL") if sp else DASH])
    L += _table(["Model", "fit (s)", "train run (s)", f"predict {cfg['cost']['n_rows']} sets (s)",
                 "Task 2 MCMC extrapolation (s)", "limit (s)", "speed"], rows) if rows else ["No cost yet."]
    L += [""]

    # 10. training flags
    L += [SECTIONS[9], ""]
    flags = eo.get("training_flags", {})
    if not flags:
        L += ["No neural candidate run."]
    for run_id, fd in flags.items():
        L += [f"- Run {run_id}: max_epochs {fd['max_epochs']}; {fd['n_trials_max_epochs']} trial(s) reached it"]
        for t in fd["trials"]:
            if t["flagged"]:
                L += [f"  - trial {t['trial']} (variant {t['variant']}, lr {g4(t['lr'])}): epochs per seed {t['n_epochs']}"]
        for v, sel in fd.get("selected_flagged", {}).items():
            mem = fd.get("n_members_max_epochs", {}).get(v, 0)
            L += [f"  - variant {v}: selected trial among them: {'yes' if sel else 'no'}; final members that reached "
                  f"it: {mem}"]
    L += [""]

    # 11. sigma_proxy
    L += [SECTIONS[10], ""]
    if E:
        k_lo = cfg["mismatch"]["window"]
        K_idx = _k_indices(con, k_lo)
        rows = []
        for c in cands:
            vals = []
            for f in ("oil", "water"):
                arr = [_get(ei, c.proxy_id, "val", f, "rmse_by_date", k) for k in K_idx]
                arr = [a for a in arr if a is not None]
                vals += [min(arr), float(np.median(arr)), max(arr)] if arr else [None] * 3
            rows.append([c.label] + vals)
        L += _table(["Model", "oil min", "oil median", "oil max", "water min", "water median", "water max"], rows)
        L += ["", f"Validation RMSE by date over K (run {E['run_id']}, metric `rmse_by_date`). Task 4 reads the frozen "
              "primary's values with `read_rmse_by_date`.", ""]
    else:
        L += ["No evaluate run yet.", ""]

    # 12. clipping
    L += [SECTIONS[11], ""]
    rows = []
    for c in cands:
        rows.append([c.label, _get(ei, c.proxy_id, "train", "all", "n_clipped_pred"),
                     _get(ei, c.proxy_id, "val", "all", "n_clipped_pred"),
                     _get(ti, c.proxy_id, "test", "all", "n_clipped_pred") if ti else NOT_SCORED])
    L += _table(["Model", "train", "val", "test"], rows) if rows else ["No predictions yet."]
    L += [""]

    # 13. figures
    L += [SECTIONS[12], ""]
    reports = Path(cfg["paths"]["reports"])
    figs = {}
    for rec in (E, B, Tst):
        if rec:
            figs.update(rec["outputs"].get("files", {}).get("figures", {}) or {})
    for name, p in figs.items():
        L += [f"- {name}: {_rel(p, reports)}"]
    if not figs:
        L += ["No figures (report.plots is false, or no run yet)."]
    L += [""]
    out = reports / "model_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def _rel(p: str, base: Path) -> str:
    try:
        import os

        return Path(os.path.relpath(Path(p).resolve(), base.resolve())).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _k_indices(con, window) -> list[int]:
    rows = con.execute("SELECT date_idx FROM history WHERE date >= ? AND date <= ? ORDER BY date_idx",
                       (window[0], window[1])).fetchall()
    return [int(r[0]) for r in rows]


def _train_variant(con, c: Candidate) -> dict:
    row = con.execute("SELECT outputs FROM runs WHERE run_id = ?", (c.run_id,)).fetchone()
    return (json.loads(row[0] or "{}").get("variants", {}).get(c.variant, {})) if row else {}


# --- selection.md -------------------------------------------------------------------------------

def write_selection_report(con, cfg: dict, select_run: str) -> Path:
    """Writes outputs/reports/selection.md with the 6 parts of spec section 1.7.2."""
    from our_work.proxy_model_sijie.src.dwc26.evaluation.selection import build_scores

    S = run_record(con, select_run)
    so = S["outputs"]
    E, B = run_record(con, so["evaluate_run"]), run_record(con, so["backtest_run"])
    e_rows = index_metrics(read_metrics(con, E["run_id"], split="val"))
    b_rows = index_metrics(read_metrics(con, B["run_id"], split="val"))
    scores, _ = build_scores(E, B, e_rows, b_rows, cfg)
    by_id = {s.proxy_id: s for s in scores}
    sel = {r[0]: r[1:] for r in con.execute("SELECT proxy_id, rank, role FROM selection WHERE run_id = ?",
                                            (select_run,)).fetchall()}
    p2_mode = so["p2_mode"]
    L = ["# Proxy selection (Task 3)", ""]
    L += [SELECTION_PARTS[0], ""]
    L += [f"- Select run {select_run}, git commit {S['git_commit']}{'+dirty' if S['git_dirty'] else ''}",
          f"- Evaluate run {E['run_id']}, backtest run {B['run_id']}",
          f"- Config {S['config_path']} (sha256 `{S['config_sha256']}`)",
          f"- p2_mode = {p2_mode}, band = {so['band']}, secondary_only = {so['secondary_only']}"]
    if so.get("override_reason"):
        L += [f"- OVERRIDE (select after a freeze): {so['override_reason']}"]
    L += ["", "Legend: " + "; ".join(f"`{s.label}` = {s.proxy_id}" for s in sorted(scores, key=lambda s: s.label)), ""]

    L += [SELECTION_PARTS[1], ""]
    rows = []
    for s in sorted(scores, key=lambda s: (sel.get(s.proxy_id, (None,))[0] is None,
                                           sel.get(s.proxy_id, (0,))[0] or 0, s.label)):
        rank, role = sel.get(s.proxy_id, (None, None))
        rows.append([s.label, s.strategy, _pf(s.step1["oil"]), _pf(s.step1["water"]), _pf(s.step2["oil"]),
                     _pf(s.step2["water"]), _pf(s.p1), _pf(s.p2), DASH if s.p3_flag is None else
                     ("FLAG" if s.p3_flag else "ok"), "yes" if s.eligible else "no", s.j, s.bt, s.rho,
                     DASH if rank is None else str(rank), role or DASH, _reason(s, p2_mode)])
    L += _table(["Proxy", "Strategy", "Step 1 oil", "Step 1 water", "Step 2 oil", "Step 2 water", "P1", "P2", "P3",
                 "Eligible", "J", "BT", "ρ", "Rank", "Role", "Reasons"], rows)
    L += ["", f"Step 1: validation RMSE below min(B0, B1). Step 2: backtest RMSE pooled over h "
          f"{cfg['backtest']['v6_horizons'][0]} to {cfg['backtest']['v6_horizons'][1]} below B2. P3 is report only.", ""]

    L += [SELECTION_PARTS[2], ""]
    tr = so.get("trace", {}).get("primary")
    if tr is None:
        L += ["No eligible primary candidate (models in `secondary_only` cannot be primary)."]
    else:
        L += [f"- j* = {g4(tr['j_star'])}; band limit (1 + {so['band']}) j* = {g4(tr['j_limit'])}; "
              f"T1 = {[by_id[p].label for p in tr['T1']]}",
              f"- b* = {g4(tr['b_star'])}; band limit = {g4(tr['b_limit'])}; T2 = {[by_id[p].label for p in tr['T2']]}",
              f"- ρ in T2: " + ", ".join(f"{by_id[p].label} {g4(v)}" for p, v in tr["rho"].items()),
              f"- Primary: {by_id[tr['best']].label} ({tr['best']}), decided by the {tr['rule']}",
              f"- Full ranking of the eligible primary candidates: {[by_id[p].label for p in so['ranking']]}"]
    L += [""]

    L += [SELECTION_PARTS[3], ""]
    if so["primary"] is None:
        L += ["No primary, so no secondary."]
    else:
        prim = by_id[so["primary"]]
        pool = so.get("trace", {}).get("secondary_pool", [])
        L += [f"- Rule (PRD section 7 step 5): the best eligible model with a strategy other than {prim.strategy}.",
              f"- Eligible candidates with the other strategy: {[by_id[p].label for p in pool] or 'none'}"]
        if so["fallback_secondary"]:
            L += [f"- D-020 fallback: no eligible model of the other strategy; the secondary is the best eligible S3 "
                  f"model of the other family: {by_id[so['secondary']].label}. The method uncertainty covers S3 only."]
        elif so["secondary"] is not None:
            L += [f"- Secondary: {by_id[so['secondary']].label} ({so['secondary']})"]
        else:
            L += ["- No eligible secondary (status no_secondary)."]
        if so["secondary"] is not None and by_id[so["secondary"]].model == "dmd":
            L += ["- D-022: the secondary is DMD. Task 5 must state the DMD tail limitation (P3) wherever the DMD "
                  "forecast measures the method uncertainty."]
    L += [""]

    L += [SELECTION_PARTS[4], ""]
    if so["primary"] is None:
        L += ["PRIMARY: none (status no_eligible_primary; `freeze` refuses and the researcher decides)",
              "SECONDARY: none", "",
              "Freeze commands (after a researcher decision): `python -m dwc26 freeze --model <proxy_id> "
              "[--secondary <proxy_id>]`"]
    else:
        prim = by_id[so["primary"]]
        L += [f"PRIMARY: {prim.proxy_id} ({prim.label}, {prim.strategy})"]
        if so["secondary"] is None:
            L += ["SECONDARY: none"]
        else:
            sec = by_id[so["secondary"]]
            L += [f"SECONDARY: {sec.proxy_id} ({sec.label}, {sec.strategy}"
                  + (", D-020 fallback" if so["fallback_secondary"] else "") + ")"]
        sec_arg = f" --secondary {so['secondary']}" if so["secondary"] else ""
        L += ["", f"Status: {so['status']}. After the review approves, the main session runs (spec section 0.1, step 4):",
              "", "```bash", "git checkout main && git merge task3",
              "# the researcher writes both proxy IDs in docs/decisions.md; the main session commits it",
              f"git tag -a {cfg['freeze']['tag']} -m \"D-0xx\"",
              f"python -m dwc26 freeze --model {prim.proxy_id}{sec_arg}",
              "python -m dwc26 evaluate --split test", "```"]
    L += [""]

    L += [SELECTION_PARTS[5], ""]
    L += ["- The validation J of each final model is optimistically biased: it was the tuning and early-stopping "
          "objective (task2 section 2.5).",
          "- The backtest settings were tuned on the full window (task2 section 2.13).",
          f"- ρ rests on {E['outputs']['n_cases']['val']} validation cases.",
          "- No new wells and no development after the history period: the user's answer (D-031). P2 and P3 rest "
          "on it (D-020, D-022).", ""]
    out = Path(cfg["paths"]["reports"]) / "selection.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out
