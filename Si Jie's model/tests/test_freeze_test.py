"""E24: the full V3 sequence (freeze, then one test scoring) in a temporary git repository
(spec task3 sections 0.1, 1.5.4, 1.5.5 and 1.10).

`freeze.repo_dir` points to the temporary repository, and `dwc26.data.db.git_state` is
monkeypatched to return its HEAD and git_dirty 0, so that the lineage check (section 1.5.4
step 3) runs against one repository. The tiny models are not eligible (DMD clips in its
continuation, the 5-epoch LSTM loses to B1), so `select` is given a fixed outcome; the
selection algorithm itself is tested by E11 and E23. No tag of the project repository is
touched: every git command runs in the temporary repository.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from dwc26.cli import main
from dwc26.data import db
from dwc26.data.check import ContractError
from dwc26.evaluation import evaluate, plots, selection
from synth_eval import build_tiny_task2, git, make_git_repo, git_available, tiny_eval_cfg
from synth_models import T_DB, TEST_IDS

pytestmark = pytest.mark.skipif(not git_available(), reason="git is not on PATH")


def _fixed_select(primary: str, secondary: str | None):
    def fake(scores, cfg):
        return selection.Selection(primary=primary, secondary=secondary, fallback_secondary=False,
                                   status="ok" if secondary else "no_secondary", ranking=[primary],
                                   trace={"primary": None, "secondary_pool": [secondary] if secondary else []})
    return fake


def _counts(db_path) -> dict:
    con = sqlite3.connect(db_path)
    out = {"metrics_test": con.execute("SELECT COUNT(*) FROM metrics WHERE split = 'test'").fetchone()[0],
           "pred_test": con.execute("SELECT COUNT(*) FROM predictions WHERE split = 'test'").fetchone()[0],
           "scorings": con.execute("SELECT COUNT(*) FROM test_scoring").fetchone()[0]}
    con.close()
    return out


def _last_out(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def _chain(t2, repo, out, monkeypatch, primary, secondary, **overrides):
    monkeypatch.setattr(db, "git_state", lambda: (git(repo, "rev-parse", "HEAD"), 0))
    monkeypatch.setattr(selection, "select", _fixed_select(primary, secondary))
    cfg = tiny_eval_cfg(t2, out, repo, **overrides)
    for argv in (["evaluate", "--split", "val"], ["backtest"], ["select"]):
        assert main([*argv, "--config", str(cfg)]) == 0
    return cfg


def _write_decision(repo, *ids):
    p = repo / "docs" / "decisions.md"
    p.write_text(p.read_text(encoding="utf-8") + "\n| D-0xx | frozen choice | " + " ; ".join(ids) + " |\n",
                 encoding="utf-8")
    git(repo, "commit", "-q", "-am", "decision")


def test_e24_v3_sequence(tiny_task2_db, tmp_git_repo, tmp_path, monkeypatch, capsys):
    t2, repo = tiny_task2_db, tmp_git_repo
    primary, secondary = f"{t2['runs']['lstm']}/A", f"{t2['runs']['dmd']}/main"
    cfg = _chain(t2, repo, tmp_path, monkeypatch, primary, secondary)
    test_cmd = ["evaluate", "--split", "test", "--config", str(cfg)]
    freeze_cmd = ["freeze", "--model", primary, "--config", str(cfg)]
    capsys.readouterr()

    # (1) before the tag: G1, and no test row anywhere
    assert main(test_cmd) == 3
    assert "G1" in capsys.readouterr().err
    assert _counts(t2["db"]) == {"metrics_test": 0, "pred_test": 0, "scorings": 0}

    # (2) a tag whose decisions.md lacks the IDs: G5
    git(repo, "tag", "-a", "freeze-proxy", "-m", "first tag")
    assert main(freeze_cmd) == 3
    assert "G5" in capsys.readouterr().err

    # (3) write the IDs, commit, move the tag: the freeze is recorded
    _write_decision(repo, primary, secondary)
    git(repo, "tag", "-f", "-a", "freeze-proxy", "-m", "moved tag")
    assert main(freeze_cmd) == 0
    con = sqlite3.connect(t2["db"])
    fr = con.execute("SELECT primary_id, secondary_id, tag_commit FROM freeze").fetchall()
    con.close()
    assert fr == [(primary, secondary, git(repo, "rev-parse", "HEAD"))]
    capsys.readouterr()

    # (4) a dirty tree: G3
    dec = repo / "docs" / "decisions.md"
    dec.write_text(dec.read_text(encoding="utf-8") + "uncommitted\n", encoding="utf-8")
    assert main(test_cmd) == 3
    assert "G3" in capsys.readouterr().err
    git(repo, "checkout", "--", "docs/decisions.md")
    assert _counts(t2["db"])["scorings"] == 0

    # (5) clean tree: the single test scoring
    assert main(test_cmd) == 0
    run_id = _last_out(capsys)
    con = sqlite3.connect(t2["db"])
    out = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0])
    pids = [c["proxy_id"] for c in out["candidate_list"]]
    have = {r[0] for r in con.execute("SELECT DISTINCT proxy_id FROM metrics WHERE run_id = ? AND split = 'test'",
                                      (run_id,)).fetchall()}
    assert have == set(pids) | {"b2"}
    for c in out["candidate_list"]:
        n = con.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ? AND variant = ? AND split = 'test'",
                        (c["run_id"], c["variant"])).fetchone()[0]
        assert n == len(TEST_IDS) * T_DB
        ids = {r[0] for r in con.execute("SELECT DISTINCT case_id FROM predictions WHERE run_id = ? AND variant = ? "
                                         "AND split = 'test'", (c["run_id"], c["variant"])).fetchall()}
        assert ids == set(TEST_IDS)
    assert con.execute("SELECT status, override_reason FROM test_scoring").fetchall() == [("ok", None)]
    con.close()
    assert {"freeze_id", "scoring_id", "tag_commit", "override_reason", "selection_run", "evaluate_run",
            "backtest_run"} <= set(out)
    report = (tmp_path / "reports" / "model_comparison.md").read_text(encoding="utf-8")
    assert "scored once" in report
    before = _counts(t2["db"])

    # (6) a second run: G6, nothing written
    assert main(test_cmd) == 3
    assert "G6" in capsys.readouterr().err
    assert _counts(t2["db"]) == before

    # (7) an override: a second scoring row; the predictions are compared, not rewritten
    reason = "synthetic rerun for E24"
    assert main([*test_cmd, "--override-reason", reason]) == 0
    run2 = _last_out(capsys)
    after = _counts(t2["db"])
    assert after["scorings"] == 2 and after["pred_test"] == before["pred_test"]
    con = sqlite3.connect(t2["db"])
    assert con.execute("SELECT status, override_reason FROM test_scoring ORDER BY scoring_id").fetchall() == [
        ("ok", None), ("ok", reason)]
    out2 = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (run2,)).fetchone()[0])
    con.close()
    assert sorted(out2["predictions_compared"]) == sorted(pids) and out2["n_prediction_rows"] == 0
    assert out2["override_reason"] == reason
    assert reason in (tmp_path / "reports" / "model_comparison.md").read_text(encoding="utf-8")

    # (8) a committed change under a frozen path: G4, even with an override
    (repo / "src" / "dwc26" / "evaluation" / "placeholder.txt").write_text("changed\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "change frozen code")
    assert main([*test_cmd, "--override-reason", "another synthetic rerun"]) == 3
    assert "G4" in capsys.readouterr().err
    assert _counts(t2["db"]) == after


def test_e24_failed_first_attempt_counts(tmp_path, monkeypatch, capsys):
    t2 = build_tiny_task2(tmp_path / "t2b", models=("b0", "b1", "dmd"))
    repo = make_git_repo(tmp_path / "repo")
    primary = f"{t2['runs']['dmd']}/main"
    cfg = _chain(t2, repo, tmp_path / "out", monkeypatch, primary, None)
    _write_decision(repo, primary)
    git(repo, "tag", "-a", "freeze-proxy", "-m", "tag")
    assert main(["freeze", "--model", primary, "--config", str(cfg)]) == 0
    capsys.readouterr()

    def boom(*a, **k):
        raise ContractError("synthetic failure after the claim")

    with monkeypatch.context() as mp:
        mp.setattr(evaluate, "evaluate_models", boom)
        assert main(["evaluate", "--split", "test", "--config", str(cfg)]) == 1
    con = sqlite3.connect(t2["db"])
    assert con.execute("SELECT status FROM test_scoring").fetchall() == [("failed",)]
    assert con.execute("SELECT COUNT(*) FROM metrics WHERE split = 'test'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM predictions WHERE split = 'test'").fetchone()[0] == 0
    con.close()
    capsys.readouterr()
    assert main(["evaluate", "--split", "test", "--config", str(cfg)]) == 3
    assert "G6" in capsys.readouterr().err


def test_e24b_failure_after_finish_keeps_the_lineage(tmp_path, monkeypatch, capsys):
    """R2: a figure that fails after finish_run(ok) leaves the status and every outputs key of the test run,
    is stored as outputs.post_finish_error, lets write_model_comparison run, and the command exits 1."""
    t2 = build_tiny_task2(tmp_path / "t2c", models=("b0", "b1", "dmd"))
    repo = make_git_repo(tmp_path / "repo")
    primary = f"{t2['runs']['dmd']}/main"
    out = tmp_path / "out"
    cfg = _chain(t2, repo, out, monkeypatch, primary, None, report__plots=True)
    _write_decision(repo, primary)
    git(repo, "tag", "-a", "freeze-proxy", "-m", "tag")
    assert main(["freeze", "--model", primary, "--config", str(cfg)]) == 0
    capsys.readouterr()

    def boom(*a, **k):
        raise RuntimeError("synthetic figure failure")

    monkeypatch.setattr(plots, "plot_parity", boom)
    assert main(["evaluate", "--split", "test", "--config", str(cfg)]) == 1
    assert "synthetic figure failure" in capsys.readouterr().err
    con = sqlite3.connect(t2["db"])
    status, outputs = con.execute("SELECT status, outputs FROM runs WHERE command = 'evaluate' ORDER BY start DESC"
                                  ).fetchone()
    scoring = con.execute("SELECT status FROM test_scoring").fetchall()
    n_test = con.execute("SELECT COUNT(*) FROM metrics WHERE split = 'test'").fetchone()[0]
    con.close()
    o = json.loads(outputs)
    assert status == "ok" and scoring == [("ok",)] and n_test > 0
    assert o["split"] == "test" and o["freeze_id"] == 1 and o["scoring_id"] == 1
    assert o["tag_commit"] == git(repo, "rev-parse", "HEAD")
    assert {"selection_run", "evaluate_run", "backtest_run", "files"} <= set(o)
    assert "synthetic figure failure" in o["post_finish_error"]
    assert "scored once" in (out / "reports" / "model_comparison.md").read_text(encoding="utf-8")
    assert (out / "figures" / "backtest_test.png").is_file()           # the later figure still ran
