"""E12, E22: dwc26.forecast.backtest (PRD section 9; spec task3 sections 1.5.3 and 2.8; D-026)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import yaml

from dwc26.cli import main
from dwc26.config import ConfigError
from dwc26.forecast.backtest import backtest_errors, check_window, cutoff_index, persistence
from our_work.proxy_model_sijie.tests.synth_eval import H_BT, T_CUT, set_key, tiny_eval_cfg_dict, write_eval_cfg
from our_work.proxy_model_sijie.tests.synth_models import db_dates

CFG = {"backtest": {"v6_horizons": [3, 4]}, "metrics": {"objective_fluids": ["oil", "water"]}}


def test_e12_cutoff_and_window():
    dates = db_dates()
    assert cutoff_index(dates, "2002-01-01") == 9
    assert cutoff_index(dates, "2001-12-31") == 8
    assert check_window(9, 13, 2, [3, 4]) == 4
    with pytest.raises(ConfigError):
        check_window(4, 13, 2, [3, 4])                       # t_cut < k0 + 3
    with pytest.raises(ConfigError):
        check_window(10, 13, 2, [3, 4])                      # H = 3 < v6_horizons[1]


def test_e12_persistence():
    Y = np.arange(2 * 6 * 3, dtype=float).reshape(2, 6, 3)
    P = persistence(Y, 4, 2)
    assert P.shape == (2, 2, 3)
    np.testing.assert_array_equal(P[:, 0], Y[:, 3])
    np.testing.assert_array_equal(P[:, 1], Y[:, 3])


def test_e12_backtest_errors_by_hand():
    yt = np.zeros((2, 4, 3))
    yp = np.zeros((2, 4, 3))
    yp[0, :, 0] = [1.0, 2.0, 3.0, 4.0]                       # oil errors, case 1
    yp[1, :, 0] = [1.0, 0.0, 1.0, 2.0]                       # oil errors, case 2
    yp[:, :, 2] = 2.0                                         # water error 2 everywhere
    span = np.array([10.0, 5.0, 4.0])
    rows = backtest_errors(yt, yp, CFG, span)
    get = {(r.metric, r.fluid, r.idx): r.value for r in rows}
    for h, (a, b) in enumerate([(1, 1), (2, 0), (3, 1), (4, 2)], start=1):
        assert get[("bt_rmse", "oil", h)] == pytest.approx(np.sqrt((a * a + b * b) / 2), abs=1e-12)
        assert get[("bt_rmse", "water", h)] == pytest.approx(2.0, abs=1e-12)
    v6_oil = np.sqrt((9 + 16 + 1 + 4) / 4)
    all_oil = np.sqrt((1 + 4 + 9 + 16 + 1 + 0 + 1 + 4) / 8)
    assert get[("bt_rmse_v6", "oil", -1)] == pytest.approx(v6_oil, abs=1e-12)
    assert get[("bt_rmse_all", "oil", -1)] == pytest.approx(all_oil, abs=1e-12)
    assert get[("bt_score", "all", -1)] == pytest.approx((all_oil / 10.0 + 2.0 / 4.0) / 2, abs=1e-12)
    assert all(r.proxy_id == "" and r.split == "" for r in rows)
    with pytest.raises(Exception):
        backtest_errors(yt[:, :3], yp[:, :3], CFG, span)     # H < v6_horizons[1]


# --- E22: backtest end to end -------------------------------------------------------------------

def _train_runs(db_path):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT run_id, model, outputs FROM runs WHERE task = 'task2' AND command = 'train' "
                       "ORDER BY start").fetchall()
    con.close()
    return [(r[0], r[1], json.loads(r[2] or "{}")) for r in rows]


def _bt_metrics(db_path, run_id):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT proxy_id, split, fluid, metric, idx, value FROM metrics WHERE run_id = ?",
                       (run_id,)).fetchall()
    out = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0])
    con.close()
    return rows, out


def test_e22_config_difference_refused_before_training(tiny_task2_db, tmp_path, capsys):
    t2 = tiny_task2_db
    changed = yaml.safe_load(Path(t2["model_cfgs"]["lstm"]).read_text(encoding="utf-8"))
    changed["training"]["patience"] = 4
    alt = tmp_path / "lstm_changed.yaml"
    alt.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    cfg = set_key(tiny_eval_cfg_dict(t2, tmp_path), "backtest.model_configs.lstm", str(alt))
    path = write_eval_cfg(cfg, tmp_path / "evaluate.yaml")
    n0 = len(_train_runs(t2["db"]))
    assert main(["backtest", "--config", str(path)]) == 1
    err = capsys.readouterr().err
    assert "training.patience" in err
    assert len(_train_runs(t2["db"])) == n0                  # nothing was trained


def test_e22_backtest_end_to_end_and_reuse(tiny_task2_db, tmp_path, capsys):
    from dwc26.data.db import connect
    from dwc26.models.registry import load_model

    t2 = tiny_task2_db
    path = write_eval_cfg(tiny_eval_cfg_dict(t2, tmp_path), tmp_path / "evaluate.yaml")
    n0 = len(_train_runs(t2["db"]))
    assert main(["backtest", "--config", str(path)]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    runs = _train_runs(t2["db"])
    new = runs[n0:]
    assert sorted(m for _, m, _ in new) == ["dmd", "lstm"]
    for rid, m, o in new:
        assert o["t_cut"] == T_CUT and o["settings_from"] == t2["runs"][m]
    rows, out = _bt_metrics(t2["db"], run_id)
    assert out["t_cut"] == T_CUT and out["H"] == H_BT
    lstm, dmd = t2["runs"]["lstm"], t2["runs"]["dmd"]
    expected = {f"{lstm}/{v}" for v in "ABC"} | {f"{dmd}/main"}
    assert set(out["mapping"]) == expected
    con = connect(t2["db"])
    for pid, bpid in out["mapping"].items():
        assert load_model(bpid, t2["models"], con=con).T_fit == T_CUT
    con.close()
    for pid in expected | {"b2"}:
        for f in ("oil", "gas", "water"):
            hs = sorted(r[4] for r in rows if r[0] == pid and r[3] == "bt_rmse" and r[2] == f)
            assert hs == list(range(1, H_BT + 1))
    assert all(r[1] == "val" for r in rows)
    assert set(out["v6"]) == expected
    # reuse: a second backtest trains nothing and gives equal metrics
    cfg = tiny_eval_cfg_dict(t2, tmp_path / "again")
    set_key(cfg, "backtest.reuse", {m: out["bt_runs"][m] for m in ("lstm", "dmd")})
    path2 = write_eval_cfg(cfg, tmp_path / "again" / "evaluate.yaml")
    n1 = len(_train_runs(t2["db"]))
    assert main(["backtest", "--config", str(path2)]) == 0
    run2 = capsys.readouterr().out.strip().splitlines()[-1]
    assert len(_train_runs(t2["db"])) == n1
    rows2, out2 = _bt_metrics(t2["db"], run2)
    assert sorted(r[:5] + (r[5],) for r in rows) == sorted(r[:5] + (r[5],) for r in rows2)
    assert out2["reused"] == ["dmd", "lstm"]
