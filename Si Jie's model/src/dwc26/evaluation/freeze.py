"""`python -m dwc26 freeze --model <primary> [--secondary <id>]` (spec task3 sections 1.5.4 and 2.10; D-028).

`freeze` verifies and records; it never creates, moves or deletes a tag. It needs the
tag `freeze.tag` (G1), HEAD containing it (G2), a clean tree (G3), the proxy IDs in the
decision text at the tag and the selection, evaluate and backtest runs at ancestors of
the tag with git_dirty 0 (G5), and no earlier freeze (G6, unless overridden).
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from dwc26.config import check_override_reason, load_eval_config
from dwc26.data.check import ContractError

LOG = logging.getLogger("dwc26.task3")


def run_freeze(config_path: Path, primary: str, secondary: str | None, *,
               override_reason: str | None = None) -> str:
    """Records the frozen choice in the table `freeze`. Returns the run ID."""
    from dwc26.evaluation.guards import GitOps, GuardError, check_freeze_tag, run_record
    from dwc26.evaluation.runctx import task3_run
    from dwc26.evaluation.store import latest_ok_run
    from dwc26.models.registry import load_model, parse_proxy_id

    config_path = Path(config_path)
    cfg = load_eval_config(config_path)
    override_reason = check_override_reason(override_reason, cfg)
    fz = cfg["freeze"]
    with task3_run(cfg, config_path, "freeze") as ctx:
        con = ctx.con
        n_prev = con.execute("SELECT COUNT(*) FROM freeze").fetchone()[0]          # G6
        if n_prev:
            if override_reason is None:
                raise GuardError("G6", f"the proxy choice is already frozen ({n_prev} freeze row(s)); a new freeze "
                                       "needs --override-reason and a researcher decision")
            LOG.warning("OVERRIDE: freeze number %d, reason: %s", n_prev + 1, override_reason)
        s_id = latest_ok_run(con, "select")                                          # 1. selection run
        if s_id is None:
            raise ContractError("freeze: no ok `select` run")
        S = run_record(con, s_id)
        so = S["outputs"]
        if so["status"] == "no_eligible_primary":
            raise ContractError(f"freeze: the selection run {s_id} found no eligible primary; the researcher decides")
        if secondary is None:                                                        # 2. proposal
            secondary = so["secondary"]
        diffs = []
        if primary != so["primary"]:
            diffs.append(f"primary {primary} != selected {so['primary']}")
        if secondary != so["secondary"]:
            diffs.append(f"secondary {secondary} != selected {so['secondary']}")
        if diffs:
            if override_reason is None:
                raise ContractError(f"freeze: the choice differs from the selection run {s_id}: {diffs}; this needs "
                                    "--override-reason and a researcher decision")
            LOG.warning("OVERRIDE: %s; reason: %s", diffs, override_reason)
        repo = fz["repo_dir"]
        git = GitOps(None if repo is None else Path(repo))
        tag_commit, head = check_freeze_tag(git, cfg)                                # 3. G1 to G3
        E, B = run_record(con, so["evaluate_run"]), run_record(con, so["backtest_run"])
        for name, rec in (("select", S), ("evaluate", E), ("backtest", B)):          # lineage (G5)
            if rec["git_dirty"] != 0:
                raise GuardError("G5", f"the {name} run {rec['run_id']} has git_dirty = {rec['git_dirty']}")
            if rec["git_commit"] in (None, "", "unknown") or not git.is_ancestor(rec["git_commit"], tag_commit):
                raise GuardError("G5", f"the {name} run {rec['run_id']} (commit {str(rec['git_commit'])[:12]}) is "
                                       f"not an ancestor of the tag commit {tag_commit[:12]}")
        text = git.show(tag_commit, fz["decisions_path"])                            # 4. decision text (G5)
        if text is None:
            raise GuardError("G5", f"{fz['decisions_path']} does not exist at the tag commit {tag_commit[:12]}")
        missing = [p for p in (primary, secondary) if p is not None and p not in text]
        if missing:
            raise GuardError("G5", f"{fz['decisions_path']} at the tag commit {tag_commit[:12]} does not contain "
                                   f"{missing}; the researcher writes the choice before the tag (PRD section 7 step 6)")
        models_dir = Path(cfg["paths"]["models"])                                    # 5. loadable
        for pid in (primary, secondary):
            if pid is not None:
                load_model(pid, models_dir, con=con, proxies_only=True)
        p_run, p_var = parse_proxy_id(primary)
        s_run, s_var = parse_proxy_id(secondary) if secondary else (None, None)
        model_of = {r: m for r, m in con.execute("SELECT run_id, model FROM runs WHERE run_id IN (?, ?)",
                                                 (p_run, s_run or p_run)).fetchall()}
        for r in (p_run, s_run):
            if r is not None and r not in model_of:
                raise ContractError(f"freeze: the train run {r} has no row in runs")
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        cur = con.execute(                                                           # 6. record (trigger: G6)
            "INSERT INTO freeze (run_id, selection_run, primary_id, primary_model, primary_variant, secondary_id, "
            "secondary_model, secondary_variant, fallback_secondary, tag, tag_commit, head_commit, created, "
            "override_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ctx.run_id, s_id, primary, model_of[p_run], p_var, secondary, model_of.get(s_run) if s_run else None,
             s_var, int(bool(so["fallback_secondary"]) and secondary == so["secondary"]), fz["tag"], tag_commit, head,
             now, override_reason))
        freeze_id = int(cur.lastrowid)
        LOG.info("freeze %d: primary %s, secondary %s, tag %s at %s, HEAD %s", freeze_id, primary, secondary,
                 fz["tag"], tag_commit, head)
        ctx.finish({"freeze_id": freeze_id, "selection_run": s_id, "evaluate_run": E["run_id"],
                    "backtest_run": B["run_id"], "primary": primary, "secondary": secondary, "tag": fz["tag"],
                    "tag_commit": tag_commit, "head_commit": head, "override_reason": override_reason})
        from dwc26.evaluation import report

        ctx.post("model_comparison.md", lambda: report.write_model_comparison(con, cfg))     # R2
        return ctx.run_id
