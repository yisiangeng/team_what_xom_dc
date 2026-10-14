"""A01, A02: dwc26.data.load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dwc26.data.load import HISTORY_ID, QUANTITIES, load_parameters, load_production
from synth import CASES, GRID, PARAMS, QUANT, edit_xlsx


def test_a01_load_parameters(synthetic_raw):
    cfg = synthetic_raw.cfg
    params, problems = load_parameters(synthetic_raw.file01, cfg["file01"])
    assert problems == []
    assert list(params.columns) == ["case_id", "fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
    assert params["case_id"].dtype == np.int64
    assert params["case_id"].tolist() == CASES  # sorted, from "Case N"
    for c in ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]:
        assert params[c].dtype == np.float64
    expected = np.array([PARAMS[i] for i in CASES])
    np.testing.assert_array_equal(params.iloc[:, 1:].to_numpy(), expected)


def test_load_parameters_reports_problems(synthetic_raw):
    def corrupt(wb):
        ws = wb["Uncertainty Values"]
        ws["A3"] = "Case3"          # bad label
        ws["B4"] = "abc"            # non-numeric value
        ws["A5"] = "Case 1"         # duplicate id
    edit_xlsx(synthetic_raw.file01, corrupt)
    params, problems = load_parameters(synthetic_raw.file01, synthetic_raw.cfg["file01"])
    assert len(problems) >= 3
    assert params["case_id"].is_unique
    assert np.isfinite(params.iloc[:, 1:].to_numpy()).all()


def test_load_parameters_missing_sheet(synthetic_raw):
    cfg01 = dict(synthetic_raw.cfg["file01"], sheet="No Such Sheet")
    params, problems = load_parameters(synthetic_raw.file01, cfg01)
    assert len(params) == 0 and problems


def test_load_parameters_unreadable_file_raises(tmp_path):
    bad = tmp_path / "missing.xlsx"
    with pytest.raises(Exception):
        load_parameters(bad, {"sheet": "x", "label_column": "Item", "label_pattern": ".*", "columns": {}})


def test_a02_load_production(synthetic_raw):
    cfg = synthetic_raw.cfg
    raw = load_production(synthetic_raw.file02, cfg["file02"])
    # every configured sheet found, plus Plots and Field Production
    for q in QUANTITIES:
        assert QUANT[q][0] in raw.sheets_found
    # native rows = number of non-empty cells
    assert len(raw.native) == synthetic_raw.n_nonempty
    assert list(raw.native.columns) == ["case_id", "quantity", "date", "value"]
    assert raw.native["case_id"].dtype == np.int64
    assert raw.native["value"].dtype == np.float64
    assert raw.native["date"].dtype == np.dtype("datetime64[ms]")
    assert raw.native[["case_id", "quantity", "date"]].equals(
        raw.native[["case_id", "quantity", "date"]].sort_values(["case_id", "quantity", "date"]).reset_index(drop=True)
    )
    # RAMP_History -> case_id 0, on the grid dates only
    h = raw.native[(raw.native.case_id == HISTORY_ID) & (raw.native.quantity == "oil_cum")]
    assert h["date"].tolist() == [pd.Timestamp(g) for g in GRID]
    assert sorted(raw.native["case_id"].unique().tolist()) == [0] + CASES
    # the all-empty row of 'Oil production rate' is dropped; Date columns identical
    n_dates = len(raw.date_col["oil_cum"])
    for q in QUANTITIES:
        assert raw.date_col[q].dtype == np.dtype("datetime64[ms]")
        assert len(raw.date_col[q]) == n_dates
        assert not np.isnat(raw.date_col[q]).any()
    # time of day kept to the millisecond
    t = np.datetime64("2001-02-20T08:41:00.938", "ms")
    assert t in raw.date_col["oil_rate"]
    assert (raw.native["date"] == pd.Timestamp("2001-02-20 08:41:00.938")).any()
    # names and units parsed
    for q in QUANTITIES:
        assert raw.names[q] == [f"Case_{i}" for i in CASES] + ["RAMP_History"]
        assert raw.units[q] == [QUANT[q][1]] * (len(CASES) + 1)
    assert len(raw.bad_cells) == 0
    # Field Production
    assert list(raw.field.columns) == ["date", "oil_rate", "gas_rate", "water_rate", "water_inj_rate"]
    assert len(raw.field) == len(GRID)
    np.testing.assert_array_equal(raw.field["oil_rate"].to_numpy(), synthetic_raw.reported["oil"])


def test_load_production_bad_cells_and_unknown_columns(synthetic_raw):
    def corrupt(wb):
        ws = wb["Oil production rate"]
        ws.cell(row=6, column=2).value = "n/a"
        ws.cell(row=2, column=ws.max_column + 1).value = "Extra"
    edit_xlsx(synthetic_raw.file02, corrupt)
    raw = load_production(synthetic_raw.file02, synthetic_raw.cfg["file02"])
    assert len(raw.bad_cells) == 1
    assert raw.bad_cells.iloc[0]["raw"] == "n/a"
    assert raw.bad_cells.iloc[0]["case_id"] == 1
    assert "Extra" in raw.names["oil_rate"]


def test_load_production_missing_sheet_does_not_raise(synthetic_raw):
    edit_xlsx(synthetic_raw.file02, lambda wb: wb.remove(wb["Water production rate"]))
    raw = load_production(synthetic_raw.file02, synthetic_raw.cfg["file02"])
    assert "Water production rate" not in raw.sheets_found
    assert "water_rate" not in raw.names
    assert (raw.native.quantity != "water_rate").all()
