"""E15, E16: the Task 3 schema, triggers and dwc26.evaluation.store (spec task3 sections 1.5.1 and 1.6)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.data.db import connect, init_schema
from dwc26.evaluation.store import (MetricRow, latest_ok_run, metric_value, read_metrics, read_rmse_by_date,
                                    resolve_candidates, write_metrics)
from synth_eval import insert_run

TABLES = {"freeze", "test_scoring", "metrics", "selection"}
TRIGGERS = {"freeze_once", "freeze_no_update", "freeze_no_delete", "test_scoring_once", "test_scoring_final",
            "test_scoring_no_delete", "metrics_test_guard", "predictions_test_guard"}


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "s.sqlite")
    init_schema(c)
    yield c
    c.close()


def _names(con, kind):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,)).fetchall()}


def _freeze(con, run_id, reason=None):
    con.execute("INSERT INTO freeze (run_id, selection_run, primary_id, primary_model, primary_variant, "
                "fallback_secondary, tag, tag_commit, head_commit, created, override_reason) VALUES "
                "(?, ?, 'r/A', 'lstm', 'A', 0, 'freeze-proxy', 't', 'h', '2099', ?)", (run_id, run_id, reason))


def _scoring(con, run_id, freeze_id, reason=None, status="running"):
    con.execute("INSERT INTO test_scoring (run_id, freeze_id, tag_commit, head_commit, started, status, "
                "override_reason) VALUES (?, ?, 't', 'h', '2099', ?, ?)", (run_id, freeze_id, status, reason))


def test_e15_schema_idempotent(con):
    assert TABLES <= _names(con, "table")
    assert TRIGGERS <= _names(con, "trigger")
    init_schema(con)
    init_schema(con)
    assert TRIGGERS <= _names(con, "trigger")


def test_e15_test_rows_need_a_running_claim(con):
    r = insert_run(con)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO metrics (run_id, proxy_id, split, fluid, metric, idx, value) VALUES "
                    "(?, 'x/A', 'test', 'oil', 'rmse', -1, 1.0)", (r,))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO predictions (run_id, variant, case_id, split, date_idx, date, oil, gas, water) "
                    "VALUES (?, 'A', 25, 'test', 0, '2000-01-01', 1, 1, 1)", (r,))
    con.rollback()
    _freeze(con, r)
    t = insert_run(con, command="evaluate")
    _scoring(con, t, 1)
    con.execute("INSERT INTO metrics (run_id, proxy_id, split, fluid, metric, idx, value) VALUES "
                "(?, 'x/A', 'test', 'oil', 'rmse', -1, 1.0)", (t,))
    with pytest.raises(sqlite3.IntegrityError):              # a claim of another run does not admit this run
        con.execute("INSERT INTO metrics (run_id, proxy_id, split, fluid, metric, idx, value) VALUES "
                    "(?, 'x/A', 'test', 'oil', 'rmse', -1, 1.0)", (r,))
    con.execute("INSERT INTO predictions (run_id, variant, case_id, split, date_idx, date, oil, gas, water) "
                "VALUES (?, 'A', 25, 'test', 0, '2000-01-01', 1, 1, 1)", (r,))
    con.execute("UPDATE test_scoring SET status = 'ok' WHERE run_id = ?", (t,))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO metrics (run_id, proxy_id, split, fluid, metric, idx, value) VALUES "
                    "(?, 'x/A', 'test', 'gas', 'rmse', -1, 1.0)", (t,))


def test_e15_once_only_and_append_only(con):
    r = insert_run(con)
    _freeze(con, r)
    with pytest.raises(sqlite3.IntegrityError):
        _freeze(con, r)                                       # a second freeze without a reason
    with pytest.raises(sqlite3.IntegrityError):
        _freeze(con, r, "short")                              # the CHECK needs >= 10 characters
    _freeze(con, r, "a reason of enough length")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE freeze SET tag = 'x'")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("DELETE FROM freeze")
    t1, t2, t3 = insert_run(con), insert_run(con), insert_run(con)
    _scoring(con, t1, 1)
    with pytest.raises(sqlite3.IntegrityError):
        _scoring(con, t2, 1)
    with pytest.raises(sqlite3.IntegrityError):
        _scoring(con, t2, 1, "short")
    _scoring(con, t3, 1, "a second scoring reason")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("DELETE FROM test_scoring")
    con.execute("UPDATE test_scoring SET status = 'failed', finished = '2099' WHERE run_id = ?", (t1,))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE test_scoring SET status = 'running' WHERE run_id = ?", (t1,))


def test_e15_write_metrics_refusals(con):
    r = insert_run(con)
    ok = MetricRow("b2", "val", "oil", "bt_rmse", 2, 1.5)
    assert write_metrics(con, r, [ok], T=13, H=4) == 1
    bad = [MetricRow("x/A", "val", "oil", "no_such_metric", -1, 1.0),
           MetricRow("x/A", "val", "all", "rmse", -1, 1.0),                    # wrong fluid
           MetricRow("x/A", "val", "oil", "rmse", 3, 1.0),                     # idx must be -1
           MetricRow("x/A", "val", "oil", "rmse_by_date", 13, 1.0),            # idx > T - 1
           MetricRow("b2", "val", "oil", "bt_rmse", 5, 1.0),                   # idx > H
           MetricRow("x/A", "val", "oil", "rmse", -1, float("nan")),
           MetricRow("x/A", "val", "oil", "rmse", -1, float("inf")),
           MetricRow("x/A", "train", "oil", "coverage", -1, 0.9),              # coverage is val and test only
           MetricRow("b2", "val", "oil", "rmse", -1, 1.0)]                     # not a B2 metric
    for row in bad:
        with pytest.raises(ContractError):
            write_metrics(con, r, [row], T=13, H=4)
    t = insert_run(con, task="task2", command="train", model="b0")
    with pytest.raises(ContractError):                                         # coverage is for proxies only
        write_metrics(con, r, [MetricRow(f"{t}/main", "val", "oil", "coverage", -1, 0.9)], T=13, H=4)
    write_metrics(con, r, [MetricRow(f"{t}/main", "val", "all", "rho_O", -1, None)], T=13, H=4)
    rows = read_metrics(con, r)
    assert metric_value(rows, f"{t}/main", "val", "all", "rho_O") is None
    with pytest.raises(KeyError):
        metric_value(rows, f"{t}/main", "val", "all", "r2_O")


def _train(con, model, *, t_cut=None, settings_from=None, sha="h1", status="ok", dirty=0, start, variants=("main",)):
    return insert_run(con, task="task2", command="train", model=model, status=status, git_dirty=dirty, start=start,
                      outputs={"t_cut": t_cut, "settings_from": settings_from, "arrays_sha256": sha,
                               "variants": {v: {} for v in variants}})


def _cand_cfg(**runs):
    base = {"b0": "latest", "b1": "latest", "lstm": "latest"}
    base.update(runs)
    return {"candidates": {"runs": base, "require_clean": True}, "strategy": {"lstm": "S3"}}


def test_e16_resolve_candidates(con):
    b0 = _train(con, "b0", start="2099-01-01T00:00:01.000000Z")
    b1 = _train(con, "b1", start="2099-01-01T00:00:02.000000Z")
    good = _train(con, "lstm", start="2099-01-01T00:00:03.000000Z", variants=("C", "A", "B"))
    _train(con, "lstm", start="2099-01-01T00:00:04.000000Z", t_cut=9, settings_from=good, variants=("A",))
    _train(con, "lstm", start="2099-01-01T00:00:05.000000Z", status="failed")
    _train(con, "lstm", start="2099-01-01T00:00:06.000000Z", sha="other")
    cands = resolve_candidates(con, _cand_cfg(), "h1")
    assert [c.proxy_id for c in cands] == [f"{b0}/main", f"{b1}/main", f"{good}/A", f"{good}/B", f"{good}/C"]
    assert [c.label for c in cands][2:] == ["lstm/A", "lstm/B", "lstm/C"]
    assert cands[2].strategy == "S3" and cands[0].strategy is None and not cands[0].is_proxy
    assert [c.proxy_id for c in resolve_candidates(con, _cand_cfg(lstm=good), "h1")] == [c.proxy_id for c in cands]
    with pytest.raises(ContractError):
        resolve_candidates(con, _cand_cfg(), "h2")                              # no run with this hash
    with pytest.raises(ContractError):
        resolve_candidates(con, _cand_cfg(lstm=b0), "h1")                       # an explicit run of another model
    _train(con, "lstm", start="2099-01-01T00:00:07.000000Z", dirty=1)
    with pytest.raises(ContractError):
        resolve_candidates(con, _cand_cfg(), "h1")                              # require_clean
    cfg = _cand_cfg()
    cfg["candidates"]["require_clean"] = False
    assert len(resolve_candidates(con, cfg, "h1")) == 3


def test_e16_rmse_by_date_round_trip_and_latest(con):
    r = insert_run(con, command="evaluate", outputs={"split": "val"}, start="2099-01-01T00:00:01.000000Z")
    insert_run(con, command="evaluate", outputs={"split": "test"}, start="2099-01-01T00:00:02.000000Z")
    insert_run(con, command="evaluate", outputs={"split": "val"}, status="failed", start="2099-01-01T00:00:03.000000Z")
    a = np.random.default_rng(0).random((13, 3))
    rows = [MetricRow("x/A", "val", f, "rmse_by_date", k, float(a[k, j]))
            for k in range(13) for j, f in enumerate(("oil", "gas", "water"))]
    write_metrics(con, r, rows, T=13)
    np.testing.assert_array_equal(read_rmse_by_date(con, r, "x/A"), a)
    assert latest_ok_run(con, "evaluate", where={"split": "val"}) == r
    assert latest_ok_run(con, "select") is None
    assert json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (r,)).fetchone()[0])["split"] == "val"
