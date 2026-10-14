"""E11, E23: dwc26.evaluation.selection and the `select` command (spec task3 sections 1.5.4 and 2.9)."""

from __future__ import annotations

import json
import random
import shutil
import sqlite3

import pytest

from dwc26.cli import main
from dwc26.data import db
from dwc26.data.check import ContractError
from dwc26.evaluation.report import SELECTION_PARTS
from dwc26.evaluation.selection import candidate_score, rank_candidates, select
from synth_eval import set_key, tiny_eval_cfg_dict, write_eval_cfg

STRAT = {"rnn": "S3", "lstm": "S3", "dmd": "S4"}


def _cfg(p2_mode="reject", band=0.05, secondary_only=("dmd",)):
    return {"plausibility": {"p2_mode": p2_mode}, "selection": {"band": band, "secondary_only": list(secondary_only)}}


def mk(pid, model, j, bt, rho=None, s1=(True, True), s2=(True, True), p1=True, p2=True, p2_mode="reject",
       strategy=None):
    return candidate_score(proxy_id=pid, label=pid, model=model, strategy=strategy or STRAT[model],
                           step1={"oil": s1[0], "water": s1[1]}, step2={"oil": s2[0], "water": s2[1]}, p1=p1, p2=p2,
                           p3_flag=None, j=j, bt=bt, rho=rho, p2_mode=p2_mode)


def test_e11a_b_steps_and_p1():
    base = [mk("a", "lstm", 1.0, 1.0, 0.5), mk("b", "lstm", 0.5, 0.5, 0.9, s1=(True, False)),
            mk("c", "rnn", 0.4, 0.4, 0.9, s2=(False, True)), mk("d", "rnn", 0.3, 0.3, 0.9, p1=False)]
    assert [s.eligible for s in base] == [True, False, False, False]
    sel = select(base, _cfg())
    assert sel.primary == "a" and sel.ranking == ["a"]


def test_e11c_p2_mode():
    s_rej = mk("a", "lstm", 1.0, 1.0, p2=False, p2_mode="reject")
    s_rep = mk("a", "lstm", 1.0, 1.0, p2=False, p2_mode="report")
    assert s_rej.eligible is False and s_rep.eligible is True
    assert select([s_rej], _cfg("reject")).status == "no_eligible_primary"
    assert select([s_rep], _cfg("report")).primary == "a"
    with pytest.raises(ContractError):
        select([s_rep], _cfg("reject"))                    # eligible must agree with the steps and p2_mode


def test_e11d_bands():
    scores = [mk("x1", "lstm", 1.00, 2.0), mk("x2", "lstm", 1.04, 1.0), mk("x3", "rnn", 1.06, 0.5)]
    assert rank_candidates(scores, 0.05)[0] == "x2"         # BT inside the J band; x3 is outside the J band
    assert select(scores, _cfg()).primary == "x2"


def test_e11e_rho_tie_break():
    a, b = mk("a", "lstm", 1.0, 1.00, rho=0.2), mk("b", "lstm", 1.0, 1.02, rho=0.6)
    assert rank_candidates([a, b], 0.05) == ["b", "a"]
    c = mk("c", "lstm", 1.0, 1.0, rho=None)
    d = mk("d", "lstm", 1.0, 1.0, rho=-0.9)
    assert rank_candidates([c, d], 0.05) == ["d", "c"]       # None loses to any number


def test_e11f_secondary_other_strategy_and_dmd_never_primary():
    scores = [mk("dmd", "dmd", 0.5, 0.5, 0.9), mk("lstm", "lstm", 1.0, 1.0, 0.1), mk("rnn", "rnn", 0.9, 0.9, 0.1)]
    sel = select(scores, _cfg())
    assert sel.primary == "rnn" and sel.secondary == "dmd" and not sel.fallback_secondary and sel.status == "ok"
    assert "dmd" not in sel.ranking


def test_e11g_fallback_secondary():
    scores = [mk("dmd", "dmd", 0.5, 0.5, p1=False), mk("lstm", "lstm", 1.0, 1.0), mk("rnnA", "rnn", 1.3, 1.3),
              mk("rnnB", "rnn", 1.2, 1.2), mk("lstmB", "lstm", 1.1, 1.1)]
    sel = select(scores, _cfg())
    assert sel.primary == "lstm" and sel.secondary == "rnnB" and sel.fallback_secondary is True
    assert sel.status == "ok"


def test_e11h_no_secondary():
    sel = select([mk("lstm", "lstm", 1.0, 1.0), mk("lstmB", "lstm", 1.1, 1.1)], _cfg())
    assert sel.primary == "lstm" and sel.secondary is None and sel.status == "no_secondary"
    sel = select([mk("dmd", "dmd", 1.0, 1.0)], _cfg(secondary_only=()))
    assert sel.primary == "dmd" and sel.status == "no_secondary"                      # S4 primary: no fallback


def test_e11i_nothing_eligible():
    sel = select([mk("a", "lstm", 1.0, 1.0, p1=False), mk("dmd", "dmd", 0.5, 0.5)], _cfg())
    assert sel.status == "no_eligible_primary" and sel.primary is None and sel.secondary is None


def test_e11j_order_independent():
    rng = random.Random(3)
    scores = [mk(f"p{i}", rng.choice(["rnn", "lstm", "dmd"]), round(rng.uniform(1, 1.1), 3),
                 round(rng.uniform(1, 1.1), 3), rng.choice([None, 0.1, 0.5, 0.5])) for i in range(9)]
    ref = select(scores, _cfg())
    for _ in range(10):
        sh = scores[:]
        rng.shuffle(sh)
        assert select(sh, _cfg()) == ref


def test_select_input_checks():
    with pytest.raises(ContractError):
        select([mk("a", "lstm", 1.0, 1.0), mk("a", "lstm", 1.0, 1.0)], _cfg())       # one score for each proxy
    with pytest.raises(ContractError):
        select([mk("a", "lstm", 0.0, 1.0)], _cfg())                                   # j > 0


# --- E23: select end to end ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain(tiny_task2_db, tmp_path_factory):
    """evaluate --split val and backtest on the tiny database, with runs recorded as clean."""
    out = tmp_path_factory.mktemp("sel")
    cfg = write_eval_cfg(tiny_eval_cfg_dict(tiny_task2_db, out), out / "evaluate.yaml")
    with pytest.MonkeyPatch.context() as mp:
        commit, _ = db.git_state()
        mp.setattr(db, "git_state", lambda: (commit, 0))
        assert main(["evaluate", "--split", "val", "--config", str(cfg)]) == 0
        assert main(["backtest", "--config", str(cfg)]) == 0
    return {"cfg": cfg, "out": out, "t2": tiny_task2_db}


def _select_rows(db_path, run_id):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT proxy_id, role, rank, eligible FROM selection WHERE run_id = ?", (run_id,)).fetchall()
    out = con.execute("SELECT outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    con.close()
    return rows, json.loads(out)


def test_e23_select_end_to_end(chain, clean_git, capsys):
    assert main(["select", "--config", str(chain["cfg"])]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    rows, out = _select_rows(chain["t2"]["db"], run_id)
    proxies = {f"{chain['t2']['runs']['lstm']}/{v}" for v in "ABC"} | {f"{chain['t2']['runs']['dmd']}/main"}
    assert {r[0] for r in rows} == proxies
    roles = [r[1] for r in rows if r[1]]
    assert roles.count("primary") <= 1 and roles.count("secondary") <= 1
    assert {"evaluate_run", "backtest_run", "primary", "secondary", "fallback_secondary", "status", "p2_mode", "band",
            "ranking", "files"} <= set(out)
    text = (chain["out"] / "reports" / "selection.md").read_text(encoding="utf-8")
    for part in SELECTION_PARTS:
        assert part in text
    assert "PRIMARY:" in text and "python -m dwc26 freeze --model" in text
    assert "D-031" in text and "is open" not in text                                            # R10
    assert (chain["out"] / "reports" / "model_comparison.md").is_file()


def test_e23_select_refused_after_a_freeze(chain, clean_git, tmp_path, capsys):
    db_copy = tmp_path / "copy.sqlite"
    shutil.copy(chain["t2"]["db"], db_copy)
    cfg = tiny_eval_cfg_dict(chain["t2"], tmp_path)
    set_key(cfg, "paths.db", str(db_copy))
    path = write_eval_cfg(cfg, tmp_path / "evaluate.yaml")
    assert main(["select", "--config", str(path)]) == 0
    rid = capsys.readouterr().out.strip().splitlines()[-1]
    con = db.connect(db_copy)
    con.execute("INSERT INTO freeze (run_id, selection_run, primary_id, primary_model, primary_variant, "
                "fallback_secondary, tag, tag_commit, head_commit, created) VALUES (?, ?, 'x/A', 'lstm', 'A', 0, "
                "'freeze-proxy', 'c', 'c', '2099')", (rid, rid))
    con.commit()
    con.close()
    assert main(["select", "--config", str(path)]) == 3
    assert "G6" in capsys.readouterr().err
    reason = "synthetic reselection for E23"
    assert main(["select", "--config", str(path), "--override-reason", reason]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    _, out = _select_rows(db_copy, run_id)
    assert out["override_reason"] == reason
    log = (tmp_path / "logs" / f"{run_id}.log").read_text(encoding="utf-8")
    assert "OVERRIDE" in log and reason in log
