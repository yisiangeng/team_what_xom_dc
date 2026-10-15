"""E17, E18, E19: dwc26.evaluation.guards and the test-split access (spec task3 sections 1.10 and 4.1)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from dwc26.config import validate_eval_config
from dwc26.data.db import connect, init_schema
from dwc26.evaluation.guards import (GitOps, GuardError, TestTicket, check_freeze_tag, claim_test_scoring,
                                     finish_test_scoring, verify_ticket)
from dwc26.evaluation.splits import load_split
from our_work.proxy_model_sijie.tests.synth_eval import FakeGit, git, insert_run, tiny_eval_cfg_dict
from our_work.proxy_model_sijie.tests.synth_models import TEST_IDS, VAL_IDS, synthetic_task1_db


def _cfg():
    t2 = {"db": "x.sqlite", "models": "m", "runs": {"b0": "r0", "b1": "r1", "lstm": "r2", "dmd": "r3"},
          "model_cfgs": {"lstm": "l.yaml", "dmd": "d.yaml"}}
    return validate_eval_config(tiny_eval_cfg_dict(t2, "out"))


@pytest.fixture
def frozen(tmp_path):
    """A database with the chain E, B, S, one freeze row (tag commit t1) and a test run row."""
    con = connect(tmp_path / "g.sqlite")
    init_schema(con)
    cfg = _cfg()
    e = insert_run(con, command="evaluate", config=cfg, outputs={"split": "val"})
    b = insert_run(con, command="backtest", config=cfg)
    s = insert_run(con, command="select", config=cfg, outputs={"evaluate_run": e, "backtest_run": b})
    f = insert_run(con, command="freeze", config=cfg)
    con.execute("INSERT INTO freeze (run_id, selection_run, primary_id, primary_model, primary_variant, "
                "fallback_secondary, tag, tag_commit, head_commit, created) VALUES (?, ?, 'r2/A', 'lstm', 'A', 0, "
                "'freeze-proxy', 't1', 't1', '2099')", (f, s))
    con.commit()
    yield {"con": con, "cfg": cfg, "E": e}
    con.close()


def _claim(fz, git_obj, cfg=None, reason=None):
    t = insert_run(fz["con"], command="evaluate", outputs={"split": "test"})
    return claim_test_scoring(fz["con"], t, cfg or fz["cfg"], git_obj, reason)


def _guard(fn) -> str:
    with pytest.raises(GuardError) as ei:
        fn()
    return ei.value.guard


def test_e17_each_guard(frozen):
    con = frozen["con"]
    assert _guard(lambda: _claim(frozen, FakeGit(tag_commit=None))) == "G1"
    assert _guard(lambda: _claim(frozen, FakeGit(tag_commit="t1", head="h0"))) == "G2"
    assert _guard(lambda: _claim(frozen, FakeGit(dirty=True))) == "G3"
    g = FakeGit()
    g.changed = ["src/dwc26/evaluation/metrics.py"]
    assert _guard(lambda: _claim(frozen, g)) == "G4"
    assert _guard(lambda: _claim(frozen, FakeGit(tag_commit="t2", head="t2"))) == "G5"   # the tag was moved
    cfg = copy.deepcopy(frozen["cfg"])
    cfg["mismatch"]["alpha"] = 0.2
    assert _guard(lambda: _claim(frozen, FakeGit(), cfg=cfg)) == "G7"
    assert con.execute("SELECT COUNT(*) FROM test_scoring").fetchone()[0] == 0
    ticket = _claim(frozen, FakeGit())
    assert isinstance(ticket, TestTicket) and ticket.freeze_id == 1 and ticket.tag_commit == "t1"
    row = con.execute("SELECT status, run_id FROM test_scoring WHERE scoring_id = ?", (ticket.scoring_id,)).fetchone()
    assert row == ("running", ticket.run_id)
    verify_ticket(con, ticket)
    assert _guard(lambda: _claim(frozen, FakeGit())) == "G6"                             # a second claim
    finish_test_scoring(con, ticket, "ok")
    assert _guard(lambda: verify_ticket(con, ticket)) == "G6"
    ticket2 = _claim(frozen, FakeGit(), reason="second scoring for the E17 test")
    assert ticket2.scoring_id == ticket.scoring_id + 1


def test_e17_check_freeze_tag(frozen):
    g = FakeGit(tag_commit="t1", head="h1")
    assert _guard(lambda: check_freeze_tag(g, frozen["cfg"])) == "G2"
    g.ancestors.add(("t1", "h1"))
    assert check_freeze_tag(g, frozen["cfg"]) == ("t1", "h1")


def test_e18_real_git(tmp_git_repo, tmp_path):
    repo = tmp_git_repo
    ops = GitOps(repo)
    c1 = ops.head()
    assert ops.resolve_tag("freeze-proxy") is None
    git(repo, "tag", "-a", "freeze-proxy", "-m", "test tag")
    assert ops.resolve_tag("freeze-proxy") == c1                 # an annotated tag resolves to its commit
    assert ops.is_ancestor(c1, c1) is True
    assert ops.is_dirty() is False
    (repo / "src" / "dwc26" / "evaluation" / "placeholder.txt").write_text("changed\n", encoding="utf-8")
    assert ops.is_dirty() is True
    git(repo, "commit", "-q", "-am", "change a frozen file")
    c2 = ops.head()
    assert ops.is_ancestor(c1, c2) is True and ops.is_ancestor(c2, c1) is False
    assert ops.changed_paths(c1, c2, ["src/dwc26/evaluation", "configs/models"]) == [
        "src/dwc26/evaluation/placeholder.txt"]
    assert ops.changed_paths(c1, c2, ["configs/models"]) == []
    assert "No choice yet" in ops.show(c1, "docs/decisions.md")
    assert ops.show(c1, "docs/missing.md") is None
    with pytest.raises(GuardError):
        ops.is_ancestor("0" * 40, c2)                            # an unknown object: fail closed
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GuardError):
        GitOps(plain)                                            # not a repository
    with pytest.raises(GuardError):
        GitOps(repo / "docs")                                    # inside a repository, not its top level


def test_e18b_repo_dir_null_runs_git_at_the_top_level(tmp_git_repo, monkeypatch):
    """R1: with freeze.repo_dir null, GitOps starts from the package directory (here a subdirectory of
    the temporary repository) and runs every git command at the top level, so paths are repository paths."""
    from dwc26.evaluation import guards

    repo = tmp_git_repo
    sub = repo / "src" / "dwc26" / "evaluation"
    monkeypatch.setattr(guards, "package_dir", lambda: sub)
    ops = GitOps(None)
    c1 = ops.head()
    text = ops.show(c1, "docs/decisions.md")
    assert text is not None and "No choice yet" in text
    (sub / "placeholder.txt").write_text("changed\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "change a frozen file")
    c2 = ops.head()
    assert ops.changed_paths(c1, c2, ["src/dwc26/evaluation", "configs/models"]) == [
        "src/dwc26/evaluation/placeholder.txt"]
    assert ops.changed_paths(c1, c2, ["configs/evaluate.yaml"]) == []


def test_e19_load_split_access(tmp_path):
    s = synthetic_task1_db(tmp_path)
    con = connect(s["db"])
    with pytest.raises(GuardError):
        load_split(con, "test")                                  # no ticket
    with pytest.raises(GuardError):
        load_split(con, "test", ticket=TestTicket(99, "evaluate-x", 1, "t1"))    # forged: no claim row
    f = insert_run(con, command="freeze")
    con.execute("INSERT INTO freeze (run_id, selection_run, primary_id, primary_model, primary_variant, "
                "fallback_secondary, tag, tag_commit, head_commit, created) VALUES (?, ?, 'r/A', 'lstm', 'A', 0, "
                "'freeze-proxy', 't1', 't1', '2099')", (f, f))
    t = insert_run(con, command="evaluate")
    cur = con.execute("INSERT INTO test_scoring (run_id, freeze_id, tag_commit, head_commit, started, status) "
                      "VALUES (?, 1, 't1', 't1', '2099', 'running')", (t,))
    con.commit()
    ticket = TestTicket(int(cur.lastrowid), t, 1, "t1")
    test = load_split(con, "test", ticket=ticket)
    assert test.case_id.tolist() == list(TEST_IDS)
    finish_test_scoring(con, ticket, "ok")
    with pytest.raises(GuardError):
        load_split(con, "test", ticket=ticket)                   # a finished claim
    val = load_split(con, "val")
    assert val.case_id.tolist() == list(VAL_IDS) and val.split == "val"
    assert val.X.shape == (6, 4) and val.Y_phys.shape == (6, 13, 3)
    assert not set(val.case_id.tolist()) & set(TEST_IDS)
    with pytest.raises(Exception):
        load_split(con, "val", ticket=ticket)                    # a ticket is for the test split only
    np.testing.assert_array_equal(val.hist_phys, s["Y_phys"][0])
    con.close()
