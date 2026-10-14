"""B01 to B08: the real files 01 and 02 (marker rawdata; skipped without data/raw/)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

pytestmark = pytest.mark.rawdata


@pytest.fixture(scope="module")
def real(real_prepared):
    cfg = real_prepared["cfg"]
    z = dict(np.load(cfg["paths"]["arrays"], allow_pickle=False))
    con = sqlite3.connect(cfg["paths"]["db"])
    yield {**real_prepared, "z": z, "con": con}
    con.close()


def test_b01_shapes_and_grid(real):
    z = real["z"]
    assert z["X"].shape == (100, 4)
    assert z["Y"].shape == (100, 41, 3)
    assert z["hist"].shape == (41, 3)
    assert z["dates"].shape == (41,) and z["dates"].dtype == np.dtype("datetime64[D]")
    assert z["dates"][0] == np.datetime64("1998-01-01")
    assert z["dates"][-1] == np.datetime64("2008-01-01")
    window = (z["dates"] >= np.datetime64("1999-01-01")) & (z["dates"] <= np.datetime64("2008-01-01"))
    assert window.sum() == 37
    s = z["split"]
    assert ((s == "train").sum(), (s == "val").sum(), (s == "test").sum()) == (70, 15, 15)
    np.testing.assert_array_equal(z["case_id"], np.arange(1, 101))


def _checks(con, run_id):
    rows = con.execute("SELECT check_id, status, detail FROM data_checks WHERE run_id = ?", (run_id,)).fetchall()
    return {c: (s, json.loads(d)) for c, s, d in rows}


def test_b02_checks(real):
    checks = _checks(real["con"], real["run_id"])
    assert len(checks) == 10
    for c in ["C1", "C2", "C3", "C4", "C5", "C8"]:
        assert checks[c][0] == "pass", (c, checks[c][1])
    status, d = checks["C6"]
    assert status == "pass"
    assert d["convention"] == "next"
    assert d["e_next"] <= 1e-6
    assert d["e_prev"] >= 0.5
    assert d["definition"] == "R1"
    for c in ["C7", "C9", "C10"]:
        assert checks[c][0] == "report"
    assert str(real["z"]["rate_definition"]) == "R1"
    # A1 (D-009): C10 counts with the configured tolerance; consistent with the exact count
    d10 = checks["C10"][1]
    assert d10["tol"] == real["cfg"]["checks"]["c10_rel_tol"]
    for fluid, v in d10["fluids"].items():
        assert v["n_outside"] == v["n_below"] + v["n_above"] == len(v["outside"]), fluid
        assert v["n_outside"] <= v["n_outside_exact"], fluid
        assert (v["n_outside"] == 0) == (v["max_rel_excess"] <= d10["tol"]), fluid


def test_b03_history_shift(real):
    z = real["z"]
    rep = np.array(real["con"].execute(
        "SELECT oil_rate_reported, gas_rate_reported, water_rate_reported FROM history ORDER BY date_idx"
    ).fetchall())
    hp = z["hist_phys"]
    np.testing.assert_allclose(hp[1:], rep[:-1], rtol=1e-6, atol=0)
    np.testing.assert_array_equal(hp[0], 0.0)


def test_b04_plausibility(real):
    z = real["z"]
    k = int(np.where(z["dates"] == np.datetime64("1999-01-01"))[0][0])
    c_oil = real["con"].execute("SELECT oil_cum FROM history WHERE date_idx = ?", (k,)).fetchone()[0]
    assert z["hist_phys"][k, 0] == pytest.approx(c_oil / 92.0, rel=1e-4)
    assert z["hist_phys"][k, 0] == pytest.approx(15507.77, rel=1e-4)


def test_b05_x(real):
    z = real["z"]
    np.testing.assert_array_equal(z["X"].min(axis=0), 0.0)
    np.testing.assert_array_equal(z["X"].max(axis=0), 1.0)
    np.testing.assert_array_equal(z["x_lower"], z["X_phys"].min(axis=0))
    np.testing.assert_array_equal(z["x_upper"], z["X_phys"].max(axis=0))
    import openpyxl

    wb = openpyxl.load_workbook(real["cfg"]["paths"]["file01"], read_only=True, data_only=True)
    rows = list(wb["Uncertainty Values"].iter_rows(values_only=True))
    wb.close()
    vals = np.array([r[1:5] for r in rows[1:] if r[0] is not None], dtype=float)
    np.testing.assert_array_equal(z["x_lower"], vals.min(axis=0))
    np.testing.assert_array_equal(z["x_upper"], vals.max(axis=0))


def test_b06_y(real):
    z = real["z"]
    train = z["split"] == "train"
    np.testing.assert_array_equal(z["Y"][train].max(axis=(0, 1)), 1.0)
    assert (z["Y"] >= 0).all() and np.isfinite(z["Y"]).all()
    assert (z["hist"] >= 0).all() and np.isfinite(z["hist"]).all()


def test_b07_native_store(real):
    con = real["con"]
    rows = con.execute(
        "SELECT quantity, COUNT(*) FROM production_native WHERE case_id = 0 GROUP BY quantity"
    ).fetchall()
    assert dict(rows) == {q: 41 for q in ["oil_rate", "oil_cum", "gas_rate", "gas_cum", "water_rate", "water_cum"]}
    mn = con.execute(
        "SELECT MIN(c), COUNT(*) FROM (SELECT case_id, quantity, COUNT(*) AS c FROM production_native "
        "WHERE case_id > 0 GROUP BY case_id, quantity)"
    ).fetchone()
    assert mn[0] >= 41 and mn[1] == 600


def test_b08_c5_exactness(real):
    con = real["con"]
    for f in ["oil", "gas", "water"]:
        rows = con.execute(
            f"""SELECT g.{f}_cum, n.value FROM production_grid g
                JOIN history h ON h.date_idx = g.date_idx
                JOIN production_native n
                  ON n.case_id = g.case_id AND n.quantity = '{f}_cum'
                 AND n.date = h.date || 'T00:00:00.000'"""
        ).fetchall()
        assert len(rows) == 100 * 41
        a = np.array(rows)
        assert np.array_equal(a[:, 0], a[:, 1])
