"""A24, the run log and M37: dwc26.data.db."""

from __future__ import annotations

import json
import sqlite3
import re

import pytest

from dwc26.data.check import CheckResult, ContractError
import numpy as np

from dwc26.data.db import (
    connect,
    finish_run,
    init_schema,
    read_task1_arrays,
    start_run,
    write_checks,
    write_predictions,
)
from dwc26.data.prepare import run_prepare

EXPECTED = {
    "runs": ["run_id", "task", "command", "model", "config", "config_path", "config_sha256", "seed",
             "git_commit", "git_dirty", "start", "duration_s", "status", "outputs", "python", "packages"],
    "parameters": ["case_id", "split", "fault_trans", "poro_mult", "perm_mult", "aquifer_pv", "run_id"],
    "production_native": ["case_id", "quantity", "date", "t_days", "value"],
    "production_grid": ["case_id", "date_idx", "date", "t_days", "oil_rate", "gas_rate", "water_rate",
                        "oil_cum", "gas_cum", "water_cum"],
    "history": ["date_idx", "date", "t_days", "oil_rate", "gas_rate", "water_rate", "oil_cum", "gas_cum",
                "water_cum", "oil_rate_reported", "gas_rate_reported", "water_rate_reported",
                "water_inj_rate_reported"],
    "scalers": ["name", "kind", "variable", "fluid", "factor", "lower", "upper", "unit", "source",
                "fitted_on", "run_id"],
    "data_checks": ["run_id", "check_id", "status", "action", "detail"],
    "predictions": ["run_id", "variant", "case_id", "split", "date_idx", "date", "oil", "gas", "water",
                    "oil_std", "gas_std", "water_std"],
}


def test_a24_schema(tmp_path):
    con = connect(tmp_path / "sub" / "x.sqlite")
    assert (tmp_path / "sub" / "x.sqlite").exists()
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    init_schema(con)
    init_schema(con)  # idempotent
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(EXPECTED) <= tables
    for table, cols in EXPECTED.items():
        got = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        assert got == cols, table
    con.close()


def test_run_log(tmp_path):
    con = connect(tmp_path / "x.sqlite")
    init_schema(con)
    cfg = {"seed": 0, "b": [1, 2], "a": {"z": 1}}
    run_id = start_run(con, "task1", "prepare", cfg, tmp_path / "c.yaml", 0)
    assert re.fullmatch(r"prepare-\d{8}T\d{12}Z", run_id)
    row = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    r = dict(zip(cols, row))
    assert r["status"] == "running" and r["duration_s"] is None
    assert r["config"] == json.dumps(cfg, sort_keys=True)
    assert len(r["config_sha256"]) == 64
    assert r["git_commit"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", r["start"])
    pk = json.loads(r["packages"])
    assert set(pk) == {"numpy", "pandas", "openpyxl", "scipy", "pyyaml", "torch", "scikit-learn", "matplotlib"}
    assert pk["scikit-learn"] != "not installed" and pk["torch"] != "not installed"

    write_checks(con, run_id, [CheckResult("C1", "pass", "continue", {"a": 1})])
    finish_run(con, run_id, "failed", {"x": 1})
    r = con.execute("SELECT status, duration_s, outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert r[0] == "failed" and r[1] >= 0 and json.loads(r[2]) == {"x": 1}
    assert con.execute("SELECT COUNT(*) FROM data_checks").fetchone()[0] == 1
    with pytest.raises(Exception):
        finish_run(con, run_id, "bogus", {})
    con.close()


# --- A29 (F2): read_task1_arrays checks the run that wrote the tables ----------

def _prepared(synthetic_raw):
    run_id = run_prepare(synthetic_raw.config_path())
    con = connect(synthetic_raw.cfg["paths"]["db"])
    assert str(read_task1_arrays(con)["rate_definition"]) == "R1"   # status 'ok': reads
    return con, run_id


@pytest.mark.parametrize("status", ["failed", "running"])
def test_a29_read_task1_arrays_rejects_run_status(synthetic_raw, status):
    con, run_id = _prepared(synthetic_raw)
    con.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
    con.commit()
    with pytest.raises(ContractError, match=status):
        read_task1_arrays(con)
    con.close()


@pytest.mark.parametrize("outputs", [None, "{}", json.dumps({"rate_definition": ""}),
                                     json.dumps({"rate_definition": "R3"})])
def test_a29_read_task1_arrays_rejects_rate_definition(synthetic_raw, outputs):
    con, run_id = _prepared(synthetic_raw)
    con.execute("UPDATE runs SET outputs = ? WHERE run_id = ?", (outputs, run_id))
    con.commit()
    with pytest.raises(ContractError, match="rate_definition"):
        read_task1_arrays(con)
    con.close()


def test_a29_read_task1_arrays_rejects_missing_run(synthetic_raw):
    con, run_id = _prepared(synthetic_raw)
    con.execute("PRAGMA foreign_keys = OFF")
    con.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    con.commit()
    with pytest.raises(ContractError, match="no row in runs"):
        read_task1_arrays(con)
    con.close()


# --- M37: predictions table -----------------------------------------------------

def _pred_args(split):
    n, T = len(split), 3
    return dict(case_id=np.arange(1, n + 1), split=np.array(split),
                dates=np.array(["2000-01-01", "2000-04-01", "2000-07-01"], dtype="datetime64[D]"),
                Y_phys=np.ones((n, T, 3)), S_phys=None)


def test_m37_predictions_schema_and_write(tmp_path):
    con = connect(tmp_path / "x.sqlite")
    init_schema(con)
    init_schema(con)
    fks = con.execute("PRAGMA foreign_key_list(predictions)").fetchall()
    assert {r[2] for r in fks} == {"runs"}          # no FK to parameters
    run_id = start_run(con, "task2", "train", {"seed": 0}, tmp_path / "c.yaml", 0, model="b0")
    assert write_predictions(con, run_id, "main", **_pred_args(["train", "val"])) == 6
    con.commit()
    rows = con.execute("SELECT split, date, oil, oil_std FROM predictions ORDER BY case_id, date_idx").fetchall()
    assert rows[0] == ("train", "2000-01-01", 1.0, None) and len(rows) == 6
    with pytest.raises(ContractError, match="V3"):
        write_predictions(con, run_id, "B", **_pred_args(["train", "test"]))
    # Task 3 (spec task3 section 1.6): even with allow_test, the trigger needs a running test_scoring claim
    with pytest.raises(sqlite3.IntegrityError, match="V3"):
        write_predictions(con, run_id, "C", **_pred_args(["test"]), allow_test=True)
    con.rollback()
    con.execute("INSERT INTO freeze (run_id, selection_run, primary_id, primary_model, primary_variant, "
                "fallback_secondary, tag, tag_commit, head_commit, created) VALUES (?, ?, 'r/A', 'lstm', 'A', 0, "
                "'freeze-proxy', 't', 't', '2099')", (run_id, run_id))
    con.execute("INSERT INTO test_scoring (run_id, freeze_id, tag_commit, head_commit, started, status) "
                "VALUES (?, 1, 't', 't', '2099', 'running')", (run_id,))
    assert write_predictions(con, run_id, "C", **_pred_args(["test"]), allow_test=True) == 3
    args = _pred_args(["val"])
    args["Y_phys"][0, 1, 2] = -1.0
    with pytest.raises(ContractError):
        write_predictions(con, run_id, "D", **args)
    args = _pred_args(["val"])
    args["S_phys"] = np.full((1, 2, 3), 1.0)
    with pytest.raises(ContractError):
        write_predictions(con, run_id, "D", **args)
    con.close()


def test_m37_prepare_after_train(synthetic_raw, tmp_path):
    """predictions has no FK to parameters, so prepare can refill the Task 1 tables after a train run."""
    from dwc26.models.train import run_train
    from our_work.proxy_model_sijie.tests.synth_models import tiny_cfg

    run_prepare(synthetic_raw.config_path())
    db_path = synthetic_raw.cfg["paths"]["db"]
    run_id = run_train(tiny_cfg(tmp_path / "t", "b0", db=db_path), "b0")
    con = connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ?", (run_id,)).fetchone()[0] > 0
    con.close()
    second = run_prepare(synthetic_raw.config_path())
    con = connect(db_path)
    assert con.execute("SELECT status FROM runs WHERE run_id = ?", (second,)).fetchone()[0] == "ok"
    assert con.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ?", (run_id,)).fetchone()[0] > 0
    con.close()
