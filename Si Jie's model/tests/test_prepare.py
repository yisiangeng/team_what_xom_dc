"""A25 to A27: dwc26.data.prepare end to end on the synthetic fixture."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dwc26.data.db import read_task1_arrays
from dwc26.data.prepare import run_prepare
from dwc26.data.scale import load_scalers
from synth import CASES, T, edit_xlsx

TASK1_TABLES = ["parameters", "production_native", "production_grid", "history", "scalers"]
OUTPUT_KEYS = {"files", "inputs", "arrays_sha256", "rate_definition", "c6_convention", "checks",
               "n_cases", "n_dates"}


def npz_spec(n, T):
    return {
        "X": ((n, 4), np.dtype("float64")),
        "Y": ((n, T, 3), np.dtype("float64")),
        "hist": ((T, 3), np.dtype("float64")),
        "dates": ((T,), np.dtype("datetime64[D]")),
        "t_days": ((T,), np.dtype("float64")),
        "case_id": ((n,), np.dtype("int64")),
        "split": ((n,), np.dtype("<U5")),
        "x_names": ((4,), None),
        "fluids": ((3,), None),
        "x_lower": ((4,), np.dtype("float64")),
        "x_upper": ((4,), np.dtype("float64")),
        "y_factor": ((3,), np.dtype("float64")),
        "X_phys": ((n, 4), np.dtype("float64")),
        "Y_phys": ((n, T, 3), np.dtype("float64")),
        "hist_phys": ((T, 3), np.dtype("float64")),
        "rate_definition": ((), None),
        "run_id": ((), None),
    }


def counts(con):
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in TASK1_TABLES + ["runs", "data_checks"]}


def arrays_sha(z):
    h = hashlib.sha256()
    for k in ["X", "Y", "hist"]:
        h.update(np.ascontiguousarray(z[k]).tobytes())
    h.update(np.ascontiguousarray(z["dates"].astype("int64")).tobytes())
    for k in ["t_days", "case_id", "x_lower", "x_upper", "y_factor"]:
        h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()


def test_a25_end_to_end(synthetic_raw):
    cfg = synthetic_raw.cfg
    run_id = run_prepare(synthetic_raw.config_path())
    n = len(CASES)
    con = sqlite3.connect(cfg["paths"]["db"])
    c = counts(con)
    assert c == {"parameters": n, "production_native": synthetic_raw.n_nonempty, "production_grid": n * T,
                 "history": T, "scalers": 7, "runs": 1, "data_checks": 10}
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    run = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    assert run["run_id"] == run_id
    assert run["status"] == "ok" and run["duration_s"] > 0 and run["git_commit"]
    assert run["task"] == "task1" and run["command"] == "prepare" and run["model"] is None
    assert run["seed"] == 0
    out = json.loads(run["outputs"])
    assert OUTPUT_KEYS <= set(out)
    assert out["rate_definition"] == "R1" and out["c6_convention"] == "next"
    assert out["n_cases"] == n and out["n_dates"] == T
    assert set(out["checks"]) == {f"C{i}" for i in range(1, 11)}
    assert set(out["files"]) == {"db", "arrays", "report", "log"}
    for p in out["files"].values():
        assert Path(p).exists(), p
    assert len(out["inputs"]) == 2

    # npz
    z = dict(np.load(cfg["paths"]["arrays"], allow_pickle=False))
    spec = npz_spec(n, T)
    assert set(z) == set(spec)
    for k, (shape, dtype) in spec.items():
        assert z[k].shape == shape, k
        if dtype is not None:
            assert z[k].dtype == dtype, k
    assert z["x_names"].tolist() == ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
    assert z["fluids"].tolist() == ["oil", "gas", "water"]
    assert str(z["rate_definition"]) == "R1" and str(z["run_id"]) == run_id
    assert z["split"].tolist() == ["train"] * 4 + ["val", "test"]
    assert out["arrays_sha256"] == arrays_sha(z)
    np.testing.assert_allclose(z["Y_phys"], z["Y"] * z["y_factor"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(z["hist_phys"], z["hist"] * z["y_factor"], rtol=1e-12, atol=0)
    assert z["X"].min() == 0.0 and z["X"].max() == 1.0

    # DB rebuild equals npz exactly
    arrays = read_task1_arrays(con)
    assert set(arrays) == set(z)
    for k in z:
        assert arrays[k].dtype == z[k].dtype, k
        assert np.array_equal(arrays[k], z[k]), k

    xs, ys = load_scalers(con)
    np.testing.assert_array_equal(xs.lower, z["x_lower"])
    np.testing.assert_array_equal(ys.factor, z["y_factor"])
    assert ys.fitted_on == (1, 2, 3, 4)
    assert con.execute("SELECT COUNT(*) FROM production_native WHERE case_id = 0").fetchone()[0] == 6 * T
    con.close()

    report = Path(cfg["paths"]["report"]).read_text(encoding="utf-8")
    assert run_id in report and "FAILED" not in report
    log = Path(cfg["paths"]["logs"]) / f"{run_id}.log"
    assert log.exists() and "C6" in log.read_text(encoding="utf-8")


def test_log_timestamps_utc(synthetic_raw):
    """Review note: log timestamps are UTC (ISO 8601, 'Z'), like run_id and runs.start."""
    import datetime as dt
    import re

    cfg = synthetic_raw.cfg
    run_id = run_prepare(synthetic_raw.config_path())
    start = dt.datetime.strptime(run_id.split("-", 1)[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=dt.timezone.utc)
    lines = (Path(cfg["paths"]["logs"]) / f"{run_id}.log").read_text(encoding="utf-8").splitlines()
    assert lines
    rx = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3})Z (INFO|WARNING|ERROR) ")
    for line in lines:
        if not line or line[0].isspace() or line.startswith("Traceback"):
            continue
        m = rx.match(line)
        assert m, line
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)
        assert -1.0 <= (t - start).total_seconds() <= 300.0, (line, run_id)


def test_a26_determinism(synthetic_raw):
    cfg = synthetic_raw.cfg
    p = synthetic_raw.config_path()
    r1 = run_prepare(p)
    con = sqlite3.connect(cfg["paths"]["db"])
    c1 = counts(con)
    r2 = run_prepare(p)
    c2 = counts(con)
    assert r1 != r2
    outs = [json.loads(o) for (o,) in con.execute("SELECT outputs FROM runs ORDER BY start")]
    assert outs[0]["arrays_sha256"] == outs[1]["arrays_sha256"]
    for t in TASK1_TABLES:
        assert c1[t] == c2[t], t
    assert c2["runs"] == 2 and c2["data_checks"] == 20
    assert con.execute("SELECT DISTINCT run_id FROM parameters").fetchall() == [(r2,)]
    con.close()


def _corrupt_a12a(wb):
    from test_check import corrupt_c3_lowered

    corrupt_c3_lowered(wb)


def _cli(config_path):
    return subprocess.run(
        [sys.executable, "-m", "dwc26", "prepare", "--config", str(config_path)],
        capture_output=True, text=True, timeout=300,
    )


def test_a27_failure_path_cli(synthetic_raw):
    cfg = synthetic_raw.cfg
    edit_xlsx(synthetic_raw.file02, _corrupt_a12a)
    proc = _cli(synthetic_raw.config_path())
    assert proc.returncode == 1, proc.stderr
    con = sqlite3.connect(cfg["paths"]["db"])
    assert con.execute("SELECT status FROM runs").fetchall() == [("failed",)]
    checks = con.execute("SELECT check_id, status FROM data_checks ORDER BY rowid").fetchall()
    assert checks == [("C1", "pass"), ("C2", "pass"), ("C3", "fail")]
    for t in TASK1_TABLES:
        assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0, t
    con.close()
    assert not Path(cfg["paths"]["arrays"]).exists()
    report = Path(cfg["paths"]["report"]).read_text(encoding="utf-8")
    assert "FAILED" in report and "not run" in report


def test_failed_run_leaves_previous_outputs_untouched(synthetic_raw):
    cfg = synthetic_raw.cfg
    p = synthetic_raw.config_path()
    run_prepare(p)
    npz = Path(cfg["paths"]["arrays"])
    before = npz.read_bytes()
    con = sqlite3.connect(cfg["paths"]["db"])
    c_before = counts(con)
    grid_before = con.execute("SELECT * FROM production_grid ORDER BY case_id, date_idx").fetchall()
    edit_xlsx(synthetic_raw.file02, _corrupt_a12a)
    from dwc26.data.check import DataCheckError

    with pytest.raises(DataCheckError):
        run_prepare(p)
    c_after = counts(con)
    for t in TASK1_TABLES:
        assert c_after[t] == c_before[t]
    assert con.execute("SELECT * FROM production_grid ORDER BY case_id, date_idx").fetchall() == grid_before
    assert c_after["runs"] == 2
    assert con.execute("SELECT status FROM runs ORDER BY start").fetchall() == [("ok",), ("failed",)]
    con.close()
    assert npz.read_bytes() == before
