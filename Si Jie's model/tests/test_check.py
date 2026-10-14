"""A10 to A18: dwc26.data.check (and the pipeline paths of A14 to A16)."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dwc26.config import ConfigError, load_config
from dwc26.data.align import align_all, history_grid, to_days
from dwc26.data.check import (
    CheckResult,
    ContractError,
    DataCheckError,
    assert_array,
    check_c1,
    check_c2,
    check_c3,
    check_c4,
    check_c5,
    check_c6,
    check_c7,
    check_c8,
    check_c9,
    check_c10,
    rel,
    write_report,
)
from dwc26.data.load import load_parameters, load_production
from dwc26.data.prepare import run_prepare
from dwc26.data.scale import fit_x_scaler, fit_y_scaler
from dwc26.data.split import assign_split
from synth import GRID, build_raw, col_of, edit_xlsx, row_of, write_config


def _load(s):
    cfg = s.cfg
    params, problems = load_parameters(s.file01, cfg["file01"])
    raw = load_production(s.file02, cfg["file02"])
    return params, problems, raw


def _c1(s):
    params, problems, raw = _load(s)
    return check_c1(params, problems, raw, s.cfg)


def _grid_and_ids(s):
    params, problems, raw = _load(s)
    return params, raw, history_grid(raw), params["case_id"].to_numpy()


# --- rel ----------------------------------------------------------------------

def test_rel():
    assert rel(0.0, 0.0, 0.0) == 0.0
    assert rel(5.0, 5.0, 1.0) == 0.0
    assert rel(1.0, 2.0, 0.0) == 0.5
    assert rel(0.0, 1e-6, 1.0) == pytest.approx(1e-6)


# --- C1 -----------------------------------------------------------------------

def test_c1_passes_on_fixture(synthetic_raw):
    res = _c1(synthetic_raw)
    assert isinstance(res, CheckResult)
    assert res.check_id == "C1" and res.status == "pass", res.detail
    json.dumps(res.detail)


def _del_case3_gas_cum(wb):
    ws = wb["Gas production cumulative"]
    ws.delete_cols(col_of(ws, "Case_3"))


def _rename_history(wb):
    ws = wb["Oil production cumulative"]
    ws.cell(row=2, column=col_of(ws, "RAMP_History")).value = "RAMP_Hist"


def _extra_case(wb):
    ws = wb["Gas production rate"]
    c = ws.max_column + 1
    ws.cell(row=2, column=c).value = "Case_999"
    ws.cell(row=3, column=c).value = "Gas production rate [MSCF/d]"
    ws.cell(row=4, column=c).value = 1.0


def _bad_unit(wb):
    ws = wb["Water production rate"]
    ws.cell(row=3, column=col_of(ws, "Case_1")).value = "Water production rate [m3/d]"


def _remove_sheet(wb):
    wb.remove(wb["Water production rate"])


CORRUPT_02 = {
    "a_deleted_column": _del_case3_gas_cum,
    "b_renamed_history": _rename_history,
    "c_extra_column": _extra_case,
    "d_wrong_unit": _bad_unit,
    "e_missing_sheet": _remove_sheet,
}


@pytest.mark.parametrize("name", sorted(CORRUPT_02))
def test_a10_c1_fails_on_file02_corruption(synthetic_raw, name):
    edit_xlsx(synthetic_raw.file02, CORRUPT_02[name])
    res = _c1(synthetic_raw)
    assert res.status == "fail" and res.action == "stop"
    assert res.detail["failures"]
    json.dumps(res.detail)


def test_a10f_c1_fails_on_file01_label(synthetic_raw):
    def corrupt(wb):
        ws = wb["Uncertainty Values"]
        r = next(r for r in range(2, ws.max_row + 1) if ws.cell(row=r, column=1).value == "Case 3")
        ws.cell(row=r, column=1).value = "Case3"
    edit_xlsx(synthetic_raw.file01, corrupt)
    res = _c1(synthetic_raw)
    assert res.status == "fail"


def test_a10g_c1_fails_on_case_0(synthetic_raw):
    """A10 (review note): case 6 renamed to case 0 in file 01 and in all six
    sheets of file 02. The two files agree, so only the collision of case_id 0
    with HISTORY_ID (RAMP_History in the long tables) can fail C1(b)."""
    from dwc26.data.load import HISTORY_ID

    assert HISTORY_ID == 0

    def corrupt01(wb):
        ws = wb["Uncertainty Values"]
        r = next(r for r in range(2, ws.max_row + 1) if ws.cell(row=r, column=1).value == "Case 6")
        ws.cell(row=r, column=1).value = "Case 0"

    def corrupt02(wb):
        for ws in wb.worksheets:
            if ws.title in ("Plots", "Field Production"):
                continue
            ws.cell(row=2, column=col_of(ws, "Case_6")).value = "Case_0"

    edit_xlsx(synthetic_raw.file01, corrupt01)
    edit_xlsx(synthetic_raw.file02, corrupt02)
    res = _c1(synthetic_raw)
    assert res.status == "fail" and res.action == "stop"
    hits = [f for f in res.detail["failures"] if f["part"] == "b" and f["name"] == "Case_0"]
    assert len(hits) == 6  # one per data sheet
    assert all("RAMP_History" in f["problem"] for f in hits)
    json.dumps(res.detail)


def test_c1_fails_on_history_off_midnight(synthetic_raw):
    def corrupt(wb):
        for ws in wb.worksheets:
            if ws.title in ("Plots", "Field Production"):
                continue
            ws.cell(row=row_of(ws, GRID[1]), column=1).value = GRID[1].replace(hour=6)
    edit_xlsx(synthetic_raw.file02, corrupt)
    assert _c1(synthetic_raw).status == "fail"


def test_c1_fails_on_missing_field_column(synthetic_raw):
    def corrupt(wb):
        ws = wb["Field Production"]
        ws.cell(row=1, column=4).value = "Gas rate"
    edit_xlsx(synthetic_raw.file02, corrupt)
    assert _c1(synthetic_raw).status == "fail"


# --- C2 -----------------------------------------------------------------------

def test_a11_c2(synthetic_raw):
    params, raw, grid, ids = _grid_and_ids(synthetic_raw)
    assert check_c2(raw, grid, ids).status == "pass"

    def corrupt(wb):
        ws = wb["Oil production cumulative"]
        c = col_of(ws, "Case_2")
        r0 = row_of(ws, GRID[-1])
        for r in range(r0, ws.max_row + 1):
            ws.cell(row=r, column=c).value = None
    edit_xlsx(synthetic_raw.file02, corrupt)
    params, raw, grid, ids = _grid_and_ids(synthetic_raw)
    res = check_c2(raw, grid, ids)
    assert res.status == "fail" and res.action == "stop"
    json.dumps(res.detail)


# --- C3 -----------------------------------------------------------------------

def test_c3_passes_on_fixture(synthetic_raw):
    _, raw, _, _ = _grid_and_ids(synthetic_raw)
    assert check_c3(raw).status == "pass"


def corrupt_c3_lowered(wb):
    """A12(a): a case-4 cumulative lowered by 1 at one native date (in the
    zero-water first quarter, so the series decreases)."""
    ws = wb["Water production cumulative"]
    c = col_of(ws, "Case_4")
    r = row_of(ws, GRID[0] + dt.timedelta(days=50, hours=8, minutes=41, milliseconds=938))
    ws.cell(row=r, column=c).value = ws.cell(row=r, column=c).value - 1.0


def _negative_rate(wb):
    ws = wb["Gas production rate"]
    ws.cell(row=row_of(ws, GRID[2]), column=col_of(ws, "Case_3")).value = -1.0


def _text_cell(wb):
    ws = wb["Oil production cumulative"]
    ws.cell(row=row_of(ws, GRID[2]), column=col_of(ws, "Case_5")).value = "n/a"


def _decrease_positive(wb):
    ws = wb["Oil production cumulative"]
    c = col_of(ws, "Case_1")
    r1 = row_of(ws, GRID[2])
    # the next native date of case 1 after GRID[2] (set A, +20 days)
    r2 = row_of(ws, GRID[2] + dt.timedelta(days=20))
    ws.cell(row=r2, column=c).value = ws.cell(row=r1, column=c).value - 0.5


@pytest.mark.parametrize("fn", [corrupt_c3_lowered, _negative_rate, _text_cell, _decrease_positive])
def test_a12_c3(synthetic_raw, fn):
    edit_xlsx(synthetic_raw.file02, fn)
    _, _, raw = _load(synthetic_raw)
    res = check_c3(raw)
    assert res.status == "fail" and res.action == "stop"
    assert res.detail["n_violations"] >= 1
    assert res.detail["violations"]
    json.dumps(res.detail)


# --- C4 -----------------------------------------------------------------------

def test_a13_c4(synthetic_raw):
    _, raw, grid, _ = _grid_and_ids(synthetic_raw)
    res = check_c4(raw, grid, synthetic_raw.cfg)
    assert res.status == "pass", res.detail

    def corrupt(wb):
        ws = wb["Field Production"]
        r = row_of(ws, GRID[1], first_data_row=2, col=2)
        ws.cell(row=r, column=3).value = ws.cell(row=r, column=3).value * (1 + 1e-4)
    edit_xlsx(synthetic_raw.file02, corrupt)
    _, raw, grid, _ = _grid_and_ids(synthetic_raw)
    res = check_c4(raw, grid, synthetic_raw.cfg)
    assert res.status == "fail" and res.action == "stop"
    json.dumps(res.detail)


# --- C5 -----------------------------------------------------------------------

def test_c5_passes_on_fixture(synthetic_raw):
    _, raw, grid, ids = _grid_and_ids(synthetic_raw)
    res = check_c5(raw, grid, ids)
    assert res.status == "pass"


def test_a14_c5_interpolates(synthetic_raw):
    k = 2

    def corrupt(wb):
        for ws in wb.worksheets:
            if ws.title in ("Plots", "Field Production"):
                continue
            ws.cell(row=row_of(ws, GRID[k]), column=col_of(ws, "Case_5")).value = None
    edit_xlsx(synthetic_raw.file02, corrupt)
    params, raw, grid, ids = _grid_and_ids(synthetic_raw)
    res = check_c5(raw, grid, ids)
    assert res.status == "fail" and res.action == "interpolate (np.interp)"
    assert res.detail["n_off_native"] == 6

    al = align_all(raw, grid, ids, "R1", synthetic_raw.cfg)
    assert al.n_interpolated > 0

    run_id = run_prepare(synthetic_raw.config_path())
    con = sqlite3.connect(synthetic_raw.cfg["paths"]["db"])
    status = con.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert status == "ok"
    c5 = con.execute("SELECT status FROM data_checks WHERE run_id = ? AND check_id = 'C5'", (run_id,)).fetchone()[0]
    assert c5 == "fail"
    got = con.execute(
        "SELECT oil_cum, gas_cum, water_cum FROM production_grid WHERE case_id = 5 AND date_idx = ?", (k,)
    ).fetchone()
    con.close()
    nat = raw.native[raw.native.case_id == 5]
    t0 = grid[0]
    tk = to_days(grid[k : k + 1], t0)[0]
    for f, q in enumerate(("oil_cum", "gas_cum", "water_cum")):
        s = nat[nat.quantity == q].sort_values("date")
        tau = to_days(s["date"].to_numpy().astype("datetime64[ms]"), t0)
        v = s["value"].to_numpy()
        j = np.searchsorted(tau, tk)
        expected = v[j - 1] + (v[j] - v[j - 1]) * (tk - tau[j - 1]) / (tau[j] - tau[j - 1])
        assert got[f] == pytest.approx(expected, rel=1e-12)


# --- C6 -----------------------------------------------------------------------

def test_a15a_c6_next(synthetic_raw):
    _, raw, grid, _ = _grid_and_ids(synthetic_raw)
    res, definition = check_c6(raw, grid, synthetic_raw.cfg)
    assert res.status == "pass" and definition == "R1"
    assert res.detail["convention"] == "next"
    assert res.detail["e_next"] < 1e-12
    assert res.detail["e_prev"] > 0.01
    json.dumps(res.detail)


def test_a15b_c6_prev(tmp_path):
    s = build_raw(tmp_path, history_convention="prev")
    _, raw, grid, _ = _grid_and_ids(s)
    res, definition = check_c6(raw, grid, s.cfg)
    assert res.status == "pass" and definition == "R1"
    assert res.detail["convention"] == "prev"
    # the full pipeline also runs under 'prev'
    run_prepare(s.config_path())


def _scale_reported_oil(factor):
    def fn(wb):
        ws = wb["Oil production rate"]
        c = col_of(ws, "RAMP_History")
        r = row_of(ws, GRID[1])
        ws.cell(row=r, column=c).value = ws.cell(row=r, column=c).value * factor
        fp = wb["Field Production"]
        r = row_of(fp, GRID[1], first_data_row=2, col=2)
        fp.cell(row=r, column=3).value = fp.cell(row=r, column=3).value * factor
    return fn


def test_a15c_c6_fail_stops(synthetic_raw):
    edit_xlsx(synthetic_raw.file02, _scale_reported_oil(1.05))
    _, raw, grid, _ = _grid_and_ids(synthetic_raw)
    res, definition = check_c6(raw, grid, synthetic_raw.cfg)
    assert res.status == "fail" and definition is None and res.action == "stop"
    with pytest.raises(DataCheckError) as ei:
        run_prepare(synthetic_raw.config_path())
    ids = [r.check_id for r in ei.value.results]
    assert ids == ["C1", "C2", "C3", "C4", "C8", "C5", "C6"]


@pytest.mark.parametrize("forced", ["R1", "R2"])
def test_c6_forced_definition(synthetic_raw, forced):
    edit_xlsx(synthetic_raw.file02, _scale_reported_oil(1.05))
    _, raw, grid, _ = _grid_and_ids(synthetic_raw)
    cfg = dict(synthetic_raw.cfg, rates={"definition": forced})
    res, definition = check_c6(raw, grid, cfg)
    assert res.status == "fail" and definition == forced


# --- C8 -----------------------------------------------------------------------

def test_a16_c8(synthetic_raw):
    _, raw, grid, ids = _grid_and_ids(synthetic_raw)
    assert check_c8(raw, grid, ids, synthetic_raw.cfg).status == "pass"

    def corrupt(wb):
        ws = wb["Oil production cumulative"]
        ws.cell(row=row_of(ws, GRID[0]), column=col_of(ws, "Case_1")).value = 1.0
    edit_xlsx(synthetic_raw.file02, corrupt)
    _, raw, grid, ids = _grid_and_ids(synthetic_raw)
    res = check_c8(raw, grid, ids, synthetic_raw.cfg)
    assert res.status == "fail" and res.action == "stop"
    json.dumps(res.detail)
    with pytest.raises(DataCheckError):
        run_prepare(synthetic_raw.config_path())


# --- C7, C9, C10 --------------------------------------------------------------

def test_a17_report_checks(synthetic_raw):
    cfg = synthetic_raw.cfg
    params, raw, grid, ids = _grid_and_ids(synthetic_raw)
    split = assign_split(ids, cfg["split"])
    al = align_all(raw, grid, ids, "R1", cfg)
    xs = fit_x_scaler(params, cfg)
    ys = fit_y_scaler(al.rate, ids, split)

    c7 = check_c7(al)
    assert c7.check_id == "C7" and c7.status == "report"
    for key in ["gor_min", "gor_max", "gor_median", "n_excluded_oil_zero", "per_date",
                "n_cases_slope", "n_cases_slope_positive", "median_slope", "history"]:
        assert key in c7.detail, key
    for key in ["case_id", "date", "value"]:
        assert key in c7.detail["gor_min"] and key in c7.detail["gor_max"]
    for key in ["gor_min", "gor_max", "slope"]:
        assert key in c7.detail["history"]
    assert len(c7.detail["per_date"]) == len(GRID)
    for key in ["date", "median", "min", "max"]:
        assert key in c7.detail["per_date"][1]
    # R1 rate at t_0 is 0 for every case -> excluded
    assert c7.detail["n_excluded_oil_zero"] >= len(ids)
    json.dumps(c7.detail)

    c9 = check_c9(params, xs)
    assert c9.check_id == "C9" and c9.status == "report"
    assert set(c9.detail["parameters"]) == set(xs.names)
    for name in xs.names:
        p = c9.detail["parameters"][name]
        for key in ["lower", "upper", "min", "max", "gap_lower", "gap_upper", "quartiles",
                    "ks_uniform", "ks_loguniform"]:
            assert key in p, key
        assert set(p["ks_uniform"]) == {"statistic", "pvalue"}
        assert p["gap_lower"] == 0.0 and p["gap_upper"] == 0.0  # design bounds
        assert len(p["quartiles"]) == 3
    assert np.array(c9.detail["spearman"]["matrix"]).shape == (4, 4)
    json.dumps(c9.detail)

    c10 = check_c10(al, split, ys, cfg)
    assert c10.check_id == "C10" and c10.status == "report"
    assert c10.detail["tol"] == cfg["checks"]["c10_rel_tol"]
    for fluid in ("oil", "gas", "water"):
        d = c10.detail["fluids"][fluid]
        for key in ["n_outside", "n_below", "n_above", "outside", "hist_max_scaled",
                    "val_max_scaled", "test_max_scaled", "n_outside_exact", "max_rel_excess"]:
            assert key in d, key
        # case 2 (train) carries the history rates: inside the envelope, no excess
        assert d["n_outside_exact"] == 0 and d["max_rel_excess"] == 0.0
    z = c10.detail["zero_history"]
    assert "dates" in z and "max_case_rate" in z
    # the synthetic history is zero at t_0 and t_1 (R1)
    assert len(z["dates"]) == 2
    json.dumps(c10.detail)


def _c10_setup(s):
    cfg = s.cfg
    params, raw, grid, ids = _grid_and_ids(s)
    split = assign_split(ids, cfg["split"])
    al = align_all(raw, grid, ids, "R1", cfg)
    ys = fit_y_scaler(al.rate, ids, split)
    train = split == "train"
    return cfg, split, al, ys, al.rate[train].min(axis=0), al.rate[train].max(axis=0)


def _with_hist(al, k, f, value):
    h = al.hist_rate.copy()
    h[k, f] = value
    return dataclasses.replace(al, hist_rate=h)


@pytest.mark.parametrize("eps", [1e-7, 1e-3])
def test_a28_c10_tolerance_above(synthetic_raw, eps):
    """A28 (A1, D-009): history above the training maximum at one producing
    date by a relative (a) 1e-7: counted only without tolerance; (b) 1e-3: counted."""
    cfg, split, al, ys, L, U = _c10_setup(synthetic_raw)
    f = 0  # oil
    k = int(np.flatnonzero(al.hist_rate[:, f] > 0)[0])
    res = check_c10(_with_hist(al, k, f, U[k, f] * (1 + eps)), split, ys, cfg)
    d = res.detail["fluids"]["oil"]
    assert d["n_outside_exact"] == 1
    assert d["max_rel_excess"] == pytest.approx(eps, rel=2e-3)
    if eps < cfg["checks"]["c10_rel_tol"]:          # (a)
        assert d["n_outside"] == 0 and d["n_above"] == 0 and d["outside"] == []
    else:                                            # (b)
        assert d["n_outside"] == 1 and d["n_above"] == 1 and d["n_below"] == 0
        (o,) = d["outside"]
        assert o["k"] == k and o["side"] == "above"
        assert o["rel_excess"] == pytest.approx(eps, rel=2e-3)
    for other in ("gas", "water"):
        assert res.detail["fluids"][other]["n_outside_exact"] == 0
    json.dumps(res.detail)


def test_a28_c10_tolerance_below(synthetic_raw):
    """The lower side of A1: below the training minimum by a relative 1e-3."""
    cfg, split, al, ys, L, U = _c10_setup(synthetic_raw)
    f = 1  # gas
    k = int(np.flatnonzero(L[:, f] > 0)[0])
    d = check_c10(_with_hist(al, k, f, L[k, f] * (1 - 1e-3)), split, ys, cfg).detail["fluids"]["gas"]
    assert d["n_outside_exact"] == 1 and d["n_outside"] == 1 and d["n_below"] == 1 and d["n_above"] == 0
    assert d["outside"][0]["side"] == "below"
    assert d["max_rel_excess"] == pytest.approx(1e-3, rel=1e-9)  # rel(h, L; 0) = (L - h) / L


def test_a28_c10_tolerance_zero_counts_every_excess(synthetic_raw):
    cfg, split, al, ys, L, U = _c10_setup(synthetic_raw)
    cfg = copy.deepcopy(cfg)
    cfg["checks"]["c10_rel_tol"] = 0.0
    k = int(np.flatnonzero(al.hist_rate[:, 0] > 0)[0])
    d = check_c10(_with_hist(al, k, 0, U[k, 0] * (1 + 1e-12)), split, ys, cfg).detail["fluids"]["oil"]
    assert d["n_outside_exact"] == 1 and d["n_outside"] == 1


@pytest.mark.parametrize("value", [-1e-9, 1.1e-3])
def test_a28_c10_rel_tol_range(synthetic_raw, value):
    cfg = copy.deepcopy(synthetic_raw.cfg)
    cfg["checks"]["c10_rel_tol"] = value
    with pytest.raises(ConfigError):
        load_config(write_config(cfg, synthetic_raw.root / "bad.yaml"))


# --- assert_array -------------------------------------------------------------

def test_a18_assert_array():
    good = np.zeros((2, 3))
    assert_array("good", good, (2, 3), nonneg=True)
    assert_array("wild", good, (None, 3))
    with pytest.raises(ContractError):
        assert_array("nan", np.array([[np.nan, 0, 0], [0, 0, 0]]), (2, 3))
    with pytest.raises(ContractError):
        assert_array("neg", np.array([[-1.0, 0, 0], [0, 0, 0]]), (2, 3), nonneg=True)
    with pytest.raises(ContractError):
        assert_array("shape", good, (3, 2))
    with pytest.raises(ContractError):
        assert_array("dtype", good.astype(np.float32), (2, 3))
    with pytest.raises(ContractError):
        assert_array("notarray", [[0.0]], (1, 1))
    with pytest.raises(ContractError):
        assert_array("nat", np.array(["NaT"], dtype="datetime64[ms]"), (1,), dtype=np.dtype("datetime64[ms]"))
    assert issubclass(ContractError, ValueError)


# --- report -------------------------------------------------------------------

def test_write_report_failed(tmp_path):
    results = [CheckResult("C1", "pass", "continue", {}), CheckResult("C2", "fail", "stop", {"failures": [1]})]
    path = tmp_path / "r" / "data_checks.md"
    write_report(path, {"run_id": "prepare-x", "status": "failed"}, results)
    text = path.read_text(encoding="utf-8")
    assert "FAILED" in text
    assert text.count("not run") >= 8


def _cells(line: str) -> int:
    """Number of Markdown table cells in `line` (unescaped pipes minus one)."""
    return len(re.findall(r"(?<!\\)\|", line)) - 1


def test_report_escapes_pipes(tmp_path):
    """Review note: the C8 summary '|C(t_0)|' must not break the summary table."""
    c8 = CheckResult("C8", "pass", "continue", {"tol": 0.0, "t_0": "2001-01-01", "max_abs": {"oil": 0.0},
                                                 "n_failing": 0, "failing": []})
    path = tmp_path / "data_checks.md"
    write_report(path, {"run_id": "prepare-x", "status": "ok"}, [c8])
    lines = path.read_text(encoding="utf-8").splitlines()
    (row,) = [ln for ln in lines if ln.startswith("| C8 |")]
    assert _cells(row) == 4, row
    assert r"\|C(t_0)\|" in row


def test_report_tables_are_well_formed(synthetic_raw):
    """Every table row of a full report has as many cells as its header."""
    run_prepare(synthetic_raw.config_path())
    lines = Path(synthetic_raw.cfg["paths"]["report"]).read_text(encoding="utf-8").splitlines()
    n = None
    for prev, line in zip([""] + lines, lines):
        if not line.startswith("|"):
            n = None
            continue
        if n is None:
            assert not prev.startswith("|")
            n = _cells(line)
        assert _cells(line) == n, line
