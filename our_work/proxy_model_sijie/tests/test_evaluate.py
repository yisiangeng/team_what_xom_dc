"""E20, E21, E25, E26: `evaluate --split val` (spec task3 sections 1.5.2 and 4.2)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from dwc26.cli import main
from dwc26.data.db import read_task1_arrays
from dwc26.data.load import FLUIDS
from dwc26.evaluation.report import NOT_SCORED, SECTIONS
from dwc26.evaluation.store import expected_metrics
from our_work.proxy_model_sijie.tests.synth_eval import tiny_eval_cfg_dict, write_eval_cfg
from our_work.proxy_model_sijie.tests.synth_models import T_DB

OUTPUT_KEYS = {"split", "candidates", "candidate_list", "arrays_sha256", "prepare_run_id", "k0", "T", "n_fc",
               "n_cases", "consistency", "identity", "summary", "mismatch", "gor_train_range", "training_flags",
               "n_metric_rows", "files"}


def _cfg(t2, out, **kv):
    from our_work.proxy_model_sijie.tests.synth_eval import set_key

    cfg = tiny_eval_cfg_dict(t2, out)
    for k, v in kv.items():
        set_key(cfg, k.replace("__", "."), v)
    return write_eval_cfg(cfg, out / "evaluate.yaml")


def _run(db_path, run_id):
    con = sqlite3.connect(db_path)
    row = con.execute("SELECT task, command, status, outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    rows = con.execute("SELECT proxy_id, split, fluid, metric, idx, value FROM metrics WHERE run_id = ?",
                       (run_id,)).fetchall()
    con.close()
    return row[0], row[1], row[2], json.loads(row[3] or "{}"), rows


def test_e21_evaluate_val_end_to_end(tiny_task2_db, tmp_path, capsys):
    t2 = tiny_task2_db
    assert main(["evaluate", "--split", "val", "--config", str(_cfg(t2, tmp_path))]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    task, command, status, out, rows = _run(t2["db"], run_id)
    assert (task, command, status, out["split"]) == ("task3", "evaluate", "ok", "val")
    assert OUTPUT_KEYS <= set(out), OUTPUT_KEYS - set(out)
    have = {(r[0], r[1], r[2], r[3]) for r in rows if r[4] == -1}
    con = sqlite3.connect(t2["db"])
    for c in out["candidate_list"]:
        for split in ("train", "val"):
            for metric, fluid in expected_metrics(c["model"], split):
                assert (c["proxy_id"], split, fluid, metric) in have, (c["proxy_id"], split, fluid, metric)
        for metric, fluid in expected_metrics(c["model"], "none"):
            assert (c["proxy_id"], "none", fluid, metric) in have
        for f in FLUIDS:
            ks = sorted(r[4] for r in rows if r[0] == c["proxy_id"] and r[3] == "rmse_by_date" and r[2] == f)
            assert ks == list(range(T_DB))
        j = next(r[5] for r in rows if r[0] == c["proxy_id"] and r[1] == "val" and r[3] == "j")
        tv = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (c["run_id"],)).fetchone()[0])
        assert j == pytest.approx(tv["variants"][c["variant"]]["j_val"], abs=1e-12)
    con.close()
    b0 = f"{t2['runs']['b0']}/main"
    for f in FLUIDS:
        r2 = next(r[5] for r in rows if r[0] == b0 and r[1] == "val" and r[2] == f and r[3] == "r2_case")
        assert r2 == pytest.approx(0.0, abs=1e-12)
    assert next(r[5] for r in rows if r[0] == b0 and r[1] == "val" and r[3] == "rho_O") is None
    assert not any(r[1] == "test" for r in rows)
    text = (tmp_path / "reports" / "model_comparison.md").read_text(encoding="utf-8")
    for s in SECTIONS:
        assert s in text, s
    assert NOT_SCORED in text and "test split not scored (V3)" in text
    assert "D-031" in text and "is open" not in text and "question to the user" not in text      # R10


def test_e20_no_test_case_reaches_a_model(tiny_task2_db, tmp_path, monkeypatch):
    """V3 spy: wrappers on predict, predict_std, rollout and forecast during evaluate val and backtest."""
    from dwc26.evaluation import splits
    from dwc26.models.base import ModelBase

    t2 = tiny_task2_db
    con = sqlite3.connect(t2["db"])
    a = read_task1_arrays(con)
    con.close()
    X_test = a["X"][a["split"] == "test"]
    assert X_test.shape[0] == 6
    seen: list[np.ndarray] = []
    for name in ("predict", "predict_std", "rollout", "forecast"):
        orig = getattr(ModelBase, name)

        def wrap(self, X, *args, _orig=orig, **kw):
            seen.append(np.array(X, dtype=np.float64))
            return _orig(self, X, *args, **kw)

        monkeypatch.setattr(ModelBase, name, wrap)
    built, reads = [], []
    real_split, real_read = splits.SplitArrays, splits.read_task1_arrays

    def spy_split(*args, **kw):
        built.append(kw.get("split"))
        return real_split(*args, **kw)

    def spy_read(con):
        reads.append(1)
        return real_read(con)

    monkeypatch.setattr(splits, "SplitArrays", spy_split)
    monkeypatch.setattr(splits, "read_task1_arrays", spy_read)
    cfg = _cfg(t2, tmp_path)
    assert main(["evaluate", "--split", "val", "--config", str(cfg)]) == 0
    assert main(["backtest", "--config", str(cfg)]) == 0
    assert reads and seen
    assert "test" not in built and {"train", "val"} <= set(built)
    rows = np.concatenate([x for x in seen if x.shape[1] == X_test.shape[1]])
    for xt in X_test:
        assert not np.any(np.all(np.abs(rows - xt) <= 1e-12, axis=1)), "a test case reached a model (V3)"
    con = sqlite3.connect(t2["db"])
    assert con.execute("SELECT COUNT(*) FROM metrics WHERE split = 'test'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM predictions WHERE split = 'test'").fetchone()[0] == 0
    con.close()


def test_e25_determinism(tiny_task2_db, tmp_path, capsys):
    t2 = tiny_task2_db
    cfg = _cfg(t2, tmp_path)
    ids = []
    for _ in range(2):
        assert main(["evaluate", "--split", "val", "--config", str(cfg)]) == 0
        ids.append(capsys.readouterr().out.strip().splitlines()[-1])
    a = {r[:5]: r[5] for r in _run(t2["db"], ids[0])[4] if not r[3].startswith("cost_")}
    b = {r[:5]: r[5] for r in _run(t2["db"], ids[1])[4] if not r[3].startswith("cost_")}
    assert a == b and len(a) > 100


def test_e26_consistency_check(tiny_task2_db, tmp_path, capsys):
    t2 = tiny_task2_db
    run, factor = t2["runs"]["dmd"], float(t2["factor"][0])
    con = sqlite3.connect(t2["db"])
    key = con.execute("SELECT case_id, date_idx, oil FROM predictions WHERE run_id = ? AND split = 'val' "
                      "AND date_idx = 5 ORDER BY case_id LIMIT 1", (run,)).fetchone()
    con.execute("UPDATE predictions SET oil = ? WHERE run_id = ? AND case_id = ? AND date_idx = ?",
                (key[2] + 0.01 * factor, run, key[0], key[1]))
    con.commit()
    try:
        assert main(["evaluate", "--split", "val", "--config", str(_cfg(t2, tmp_path))]) == 1
        assert "consistency" in capsys.readouterr().err
        last = con.execute("SELECT status FROM runs WHERE command = 'evaluate' ORDER BY start DESC").fetchone()
        assert last == ("failed",)
    finally:
        con.execute("UPDATE predictions SET oil = ? WHERE run_id = ? AND case_id = ? AND date_idx = ?",
                    (key[2], run, key[0], key[1]))
        con.commit()
        con.close()
