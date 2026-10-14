"""Selection of the primary and the secondary proxy (PRD section 7 steps 1 to 5; spec task3 section 2.9;
D-020, D-022, D-027).

1. Step 1 (V6): RMSE_val < min(B0, B1) for oil and for water.
2. Step 2 (V6): the backtest RMSE (v6 horizons) < B2 for oil and for water.
3. P1 and P2 (P2 only under p2_mode 'reject'). P3 is shown, never used.
4. Ranking within the eligible set: J band, then BT band, then rho (None last), J, BT, proxy ID.
5. Primary: the best eligible model outside `secondary_only`. Secondary: the best eligible
   model of the other strategy; else (primary S3) the best eligible S3 model of the other
   family (fallback, D-020); else none.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from dwc26.config import G7_BLOCKS, check_override_reason, load_eval_config
from dwc26.data.check import ContractError

LOG = logging.getLogger("dwc26.task3")


@dataclass(frozen=True)
class CandidateScore:
    proxy_id: str
    label: str
    model: str
    strategy: str
    step1: dict                     # fluid -> RMSE_val < min(B0, B1)
    step2: dict                     # fluid -> RMSE_bt(v6 horizons) < B2
    p1: bool
    p2: bool
    p3_flag: bool | None
    eligible: bool                  # all of step1, step2, p1, and p2 when p2_mode is 'reject'
    j: float
    bt: float
    rho: float | None
    detail: dict = field(default_factory=dict, compare=False)   # numbers for the failure reasons


@dataclass(frozen=True)
class Selection:
    primary: str | None
    secondary: str | None
    fallback_secondary: bool
    status: str                     # 'ok' | 'no_secondary' | 'no_eligible_primary'
    ranking: list                   # eligible primary candidates in rank order
    trace: dict                     # j*, T1, b*, T2, and the reasons (for selection.md)


def is_eligible(s: CandidateScore, p2_mode: str) -> bool:
    return (all(s.step1.values()) and all(s.step2.values()) and bool(s.p1)
            and (bool(s.p2) or p2_mode == "report"))


def candidate_score(*, proxy_id: str, label: str, model: str, strategy: str, step1: dict, step2: dict, p1: bool,
                    p2: bool, p3_flag: bool | None, j: float, bt: float, rho: float | None, p2_mode: str,
                    detail: dict | None = None) -> CandidateScore:
    """A CandidateScore with `eligible` computed from the steps (section 2.9 step 4)."""
    tmp = CandidateScore(proxy_id, label, model, strategy, dict(step1), dict(step2), bool(p1), bool(p2), p3_flag,
                         False, float(j), float(bt), rho, dict(detail or {}))
    return CandidateScore(**{**tmp.__dict__, "eligible": is_eligible(tmp, p2_mode)})


def _band_sets(scores: list[CandidateScore], band: float) -> tuple[float, list, float, list]:
    j_star = min(s.j for s in scores)
    t1 = [s for s in scores if s.j <= (1 + band) * j_star]
    b_star = min(s.bt for s in t1)
    t2 = [s for s in t1 if s.bt <= (1 + band) * b_star]
    return j_star, t1, b_star, t2


def _best(t2: list[CandidateScore]) -> CandidateScore:
    return min(t2, key=lambda s: (s.rho is None, -(s.rho if s.rho is not None else 0.0), s.j, s.bt, s.proxy_id))


def rank_candidates(scores: list[CandidateScore], band: float) -> list[str]:
    """Section 2.9 step 5: the full rank order of the proxy IDs."""
    rest, order = list(scores), []
    while rest:
        _, _, _, t2 = _band_sets(rest, band)
        best = _best(t2)
        order.append(best.proxy_id)
        rest = [s for s in rest if s.proxy_id != best.proxy_id]
    return order


def _check_scores(scores: list[CandidateScore], p2_mode: str) -> None:
    ids = [s.proxy_id for s in scores]
    if len(set(ids)) != len(ids):
        raise ContractError(f"select: duplicate proxy IDs {ids}")
    for s in scores:
        for name, v in (("j", s.j), ("bt", s.bt)):
            if not (isinstance(v, float) and math.isfinite(v) and v > 0):
                raise ContractError(f"select: {s.proxy_id} has {name} = {v}; it must be finite and > 0")
        if s.rho is not None and not math.isfinite(s.rho):
            raise ContractError(f"select: {s.proxy_id} has rho = {s.rho}")
        if s.eligible != is_eligible(s, p2_mode):
            raise ContractError(f"select: {s.proxy_id} has eligible = {s.eligible}, but its steps give "
                                f"{is_eligible(s, p2_mode)} under p2_mode {p2_mode!r}")


def _trace_of(pool: list[CandidateScore], band: float) -> dict | None:
    if not pool:
        return None
    j_star, t1, b_star, t2 = _band_sets(pool, band)
    best = _best(t2)
    if len(t1) == 1:
        rule = "lowest J (no other model within the J band)"
    elif len(t2) == 1:
        rule = "lowest backtest score BT within the J band"
    elif len({(s.rho is None, s.rho) for s in t2}) > 1 and best.rho is not None:
        rule = "highest mismatch rank rho within the J and BT bands"
    else:
        rule = "deterministic last keys (J, BT, proxy ID) within the J and BT bands"
    return {"j_star": j_star, "j_limit": (1 + band) * j_star, "T1": [s.proxy_id for s in t1], "b_star": b_star,
            "b_limit": (1 + band) * b_star, "T2": [s.proxy_id for s in t2],
            "rho": {s.proxy_id: s.rho for s in t2}, "best": best.proxy_id, "rule": rule}


def select(scores: list[CandidateScore], cfg: dict) -> Selection:
    """Section 2.9. The result does not depend on the input order."""
    p2_mode = cfg["plausibility"]["p2_mode"]
    band = float(cfg["selection"]["band"])
    sec_only = set(cfg["selection"]["secondary_only"])
    _check_scores(scores, p2_mode)
    scores = sorted(scores, key=lambda s: s.proxy_id)
    by_id = {s.proxy_id: s for s in scores}
    eligible = [s for s in scores if s.eligible]
    pool = [s for s in eligible if s.model not in sec_only]
    trace = {"eligible": [s.proxy_id for s in eligible], "primary_pool": [s.proxy_id for s in pool],
             "primary": _trace_of(pool, band), "p2_mode": p2_mode, "band": band, "secondary_only": sorted(sec_only)}
    if not pool:
        return Selection(None, None, False, "no_eligible_primary", [], trace)
    ranking = rank_candidates(pool, band)
    primary = by_id[ranking[0]]
    s = primary.strategy
    other = [c for c in eligible if c.strategy != s]
    trace["secondary_rule"] = f"eligible models with a strategy other than {s}"
    trace["secondary_pool"] = [c.proxy_id for c in other]
    fallback = False
    if other:
        secondary = rank_candidates(other, band)[0]
        trace["secondary"] = _trace_of(other, band)
    else:
        fb = [c for c in eligible if c.strategy == "S3" and c.model != primary.model] if s == "S3" else []
        trace["fallback_pool"] = [c.proxy_id for c in fb]
        if fb:
            secondary, fallback = rank_candidates(fb, band)[0], True
            trace["secondary"] = _trace_of(fb, band)
        else:
            secondary = None
    status = "ok" if secondary is not None else "no_secondary"
    return Selection(primary.proxy_id, secondary, fallback, status, ranking, trace)


# --- run_select (section 1.5.4) -------------------------------------------------------------

def _require(rows: dict, key: tuple):
    if key not in rows:
        raise ContractError(f"select: missing metric {key}")
    return rows[key]


def build_scores(E: dict, B: dict, e_rows: dict, b_rows: dict, cfg: dict) -> tuple[list[CandidateScore], dict]:
    """One CandidateScore for each proxy from the validation metrics of the evaluate run E and the
    backtest run B (split 'val' only). Returns the scores and the benchmark RMSE {b0, b1}."""
    from dwc26.evaluation.store import Candidate

    cands = [Candidate.from_json(d) for d in E["outputs"]["candidate_list"]]
    fl = cfg["metrics"]["objective_fluids"]
    bench = {c.model: c.proxy_id for c in cands if not c.is_proxy}
    b_rmse = {b: {f: _require(e_rows, (pid, "val", f, "rmse", -1)) for f in fl} for b, pid in bench.items()}
    n_val = int(E["outputs"]["n_cases"]["val"])
    pl = cfg["plausibility"]
    scores = []
    for c in cands:
        if not c.is_proxy:
            continue
        rm = {f: _require(e_rows, (c.proxy_id, "val", f, "rmse", -1)) for f in fl}
        lim = {f: min(b_rmse["b0"][f], b_rmse["b1"][f]) for f in fl}
        step1 = {f: rm[f] < lim[f] for f in fl}
        bt_v6 = {f: _require(b_rows, (c.proxy_id, "val", f, "bt_rmse_v6", -1)) for f in fl}
        b2_v6 = {f: _require(b_rows, ("b2", "val", f, "bt_rmse_v6", -1)) for f in fl}
        step2 = {f: _require(b_rows, (c.proxy_id, "val", f, "bt_v6_pass", -1)) == 1.0 for f in fl}
        n_clip = _require(e_rows, (c.proxy_id, "val", "all", "p1_n_clipped", -1))
        n_rise = _require(e_rows, (c.proxy_id, "val", "oil", "p2_n_rising", -1))
        p3_R = _require(e_rows, (c.proxy_id, "val", "oil", "p3_R", -1))
        detail = {"rmse": rm, "rmse_limit": lim, "b0": {f: b_rmse["b0"][f] for f in fl},
                  "b1": {f: b_rmse["b1"][f] for f in fl}, "bt_v6": bt_v6, "b2_v6": b2_v6, "p1_n_clipped": n_clip,
                  "p2_n_rising": n_rise, "p2_max_ratio": _require(e_rows, (c.proxy_id, "val", "oil", "p2_max_ratio", -1)),
                  "n_val": n_val, "p3_R": p3_R}
        scores.append(candidate_score(
            proxy_id=c.proxy_id, label=c.label, model=c.model, strategy=c.strategy, step1=step1, step2=step2,
            p1=n_clip == 0, p2=n_rise <= float(pl["p2_case_share"]) * n_val,
            p3_flag=None if p3_R is None else p3_R < float(pl["p3_ratio_min"]),
            j=_require(e_rows, (c.proxy_id, "val", "all", "j", -1)),
            bt=_require(b_rows, (c.proxy_id, "val", "all", "bt_score", -1)),
            rho=_require(e_rows, (c.proxy_id, "val", "all", "rho_O", -1)), p2_mode=pl["p2_mode"], detail=detail))
    return scores, b_rmse


def check_chain(E: dict, B: dict, cfg: dict) -> None:
    """E and B: ok, clean, the same candidate runs for the backtest models, the same arrays_sha256,
    and the same G7 config blocks as cfg. ContractError otherwise."""
    for name, rec in (("evaluate", E), ("backtest", B)):
        if rec["git_dirty"] != 0:
            raise ContractError(f"select: the {name} run {rec['run_id']} has git_dirty = {rec['git_dirty']} (V8)")
    e_runs = E["outputs"]["candidates"]
    for m, r in B["outputs"]["source_runs"].items():
        if e_runs.get(m) != r:
            raise ContractError(f"select: {m}: the evaluate run uses {e_runs.get(m)}, the backtest run {r}")
    if set(B["outputs"]["source_runs"]) != {m for m in e_runs if m in ("rnn", "lstm", "dmd")}:
        raise ContractError(f"select: the backtest models {sorted(B['outputs']['source_runs'])} differ from the "
                            f"evaluated proxy models")
    if E["outputs"]["arrays_sha256"] != B["outputs"]["arrays_sha256"]:
        raise ContractError("select: the evaluate and backtest runs used different Task 1 arrays")
    for block in G7_BLOCKS:
        vals = {json.dumps(x.get(block), sort_keys=True) for x in (E["config"], B["config"], cfg)}
        if len(vals) != 1:
            raise ContractError(f"select: config block {block!r} differs between the evaluate run, the backtest run "
                                "and this select run")


def run_select(config_path: Path, *, override_reason: str | None = None) -> str:
    """`python -m dwc26 select` (section 1.5.4). Returns the run ID."""
    from dwc26.evaluation import report
    from dwc26.evaluation.guards import GuardError, run_record
    from dwc26.evaluation.runctx import task3_run
    from dwc26.evaluation.store import index_metrics, latest_ok_run, read_metrics

    config_path = Path(config_path)
    cfg = load_eval_config(config_path)
    override_reason = check_override_reason(override_reason, cfg)
    with task3_run(cfg, config_path, "select") as ctx:
        con = ctx.con
        n_freeze = con.execute("SELECT COUNT(*) FROM freeze").fetchone()[0]        # 1. frozen choice (G6)
        if n_freeze:
            if override_reason is None:
                raise GuardError("G6", f"the proxy choice is frozen ({n_freeze} freeze row(s)); a new selection needs "
                                       "--override-reason and a researcher decision")
            LOG.warning("OVERRIDE: select after a freeze, reason: %s", override_reason)
        e_id = latest_ok_run(con, "evaluate", where={"split": "val"})               # 2. inputs
        b_id = latest_ok_run(con, "backtest")
        if e_id is None or b_id is None:
            raise ContractError(f"select: needs an ok `evaluate --split val` run ({e_id}) and an ok `backtest` run "
                                f"({b_id})")
        E, B = run_record(con, e_id), run_record(con, b_id)
        check_chain(E, B, cfg)
        e_rows = index_metrics(read_metrics(con, e_id, split="val"))               # 3. scores
        b_rows = index_metrics(read_metrics(con, b_id, split="val"))
        scores, b_rmse = build_scores(E, B, e_rows, b_rows, cfg)
        sel = select(scores, cfg)                                                  # 4. choice
        LOG.info("evaluate run %s, backtest run %s", e_id, b_id)
        for s in sorted(scores, key=lambda s: s.proxy_id):
            LOG.info("%s (%s): step1 %s step2 %s P1 %s P2 %s P3 flag %s eligible %s J %.6f BT %.6f rho %s", s.label,
                     s.proxy_id, s.step1, s.step2, s.p1, s.p2, s.p3_flag, s.eligible, s.j, s.bt, s.rho)
        LOG.info("selection: primary %s, secondary %s, fallback %s, status %s, ranking %s", sel.primary,
                 sel.secondary, sel.fallback_secondary, sel.status, sel.ranking)
        rank = {pid: i + 1 for i, pid in enumerate(sel.ranking)}                   # 5. database
        rows = []
        for s in sorted(scores, key=lambda s: s.proxy_id):
            role = "primary" if s.proxy_id == sel.primary else "secondary" if s.proxy_id == sel.secondary else None
            variant = s.proxy_id.split("/")[1]
            rows.append((ctx.run_id, s.proxy_id, s.model, variant, s.strategy, int(s.step1["oil"]),
                         int(s.step1["water"]), int(s.step2["oil"]), int(s.step2["water"]), int(s.p1), int(s.p2),
                         None if s.p3_flag is None else int(s.p3_flag), int(s.eligible), s.j, s.bt, s.rho,
                         rank.get(s.proxy_id), role))
        con.executemany("INSERT INTO selection (run_id, proxy_id, model, variant, strategy, step1_oil, step1_water, "
                        "step2_oil, step2_water, p1_pass, p2_pass, p3_flag, eligible, j_val, bt_score, rho, rank, role) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        reports = Path(cfg["paths"]["reports"])
        files = {"selection": (reports / "selection.md").as_posix(),
                 "model_comparison": (reports / "model_comparison.md").as_posix()}
        outputs = {"evaluate_run": e_id, "backtest_run": b_id, "primary": sel.primary, "secondary": sel.secondary,
                   "fallback_secondary": sel.fallback_secondary, "status": sel.status,
                   "p2_mode": cfg["plausibility"]["p2_mode"], "band": cfg["selection"]["band"],
                   "secondary_only": list(cfg["selection"]["secondary_only"]), "ranking": sel.ranking,
                   "trace": sel.trace, "override_reason": override_reason, "files": files}
        ctx.finish(outputs)
        ctx.post("selection.md", lambda: report.write_selection_report(con, cfg, ctx.run_id))   # R2
        ctx.post("model_comparison.md", lambda: report.write_model_comparison(con, cfg))
        return ctx.run_id
