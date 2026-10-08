"""Helpers for reading one grid sheet (files 03-08) of the DWC26 dataset.

These functions live in a module, not in the notebook, so that worker processes
(multiprocessing on Windows) can import them.
"""
import re
from pathlib import Path

import numpy as np
import openpyxl

NA_VALUE = -999.25  # missing-value marker used in the grid sheets

# The six columns that differ from case to case (verified on all 100 cases in notebook 1).
VARYING_COLS = ["porv", "poro", "permx", "permy", "tranx", "trany"]

# Short names for the 29 raw columns, in file order.
SHORT_NAMES = [
    "i", "j", "k", "x", "y", "z",
    "porv", "poro", "permx", "permy", "permz", "ntg", "tranx", "trany", "tranz",
    "fipnum", "pvtnum", "swl", "swcr", "swu", "sgl", "sgcr", "sowcr",
    "pressure", "swat", "soil", "sgas", "rs", "rv",
]


def short_name(raw_header: str) -> str:
    """Raw header such as 'Porosity (PORO)' -> 'poro'. Positional columns keep a plain name."""
    m = re.search(r"\(([A-Z]+)\)\s*$", str(raw_header))
    if m:
        return m.group(1).lower()
    return str(raw_header).split()[0].lower()  # 'I Index' -> 'i', 'X Coordinate' -> 'x'


def read_sheet(path, sheet):
    """Read one grid sheet. Returns (raw_header, array[n_cells, 29] as float64)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb[sheet].iter_rows(values_only=True)
        header = next(rows)
        arr = np.array(list(rows), dtype=np.float64)
    finally:
        wb.close()
    return header, arr


def process_case(job):
    """Worker: read one case and compare it with the reference case.

    job = (case_id, path, sheet, reference_path)
    Returns a dict with the varying columns of the active cells and the per-column
    maximum absolute difference to the reference case.
    """
    case_id, path, sheet, ref_path = job
    header, arr = read_sheet(path, sheet)
    ref = np.load(ref_path)["raw"]  # reference case, all columns, float64

    names = SHORT_NAMES
    assert arr.shape == ref.shape, (case_id, arr.shape, ref.shape)
    assert [short_name(h) for h in header] == names, (case_id, header)

    active = arr[:, names.index("porv")] > 0
    ref_active = ref[:, names.index("porv")] > 0
    same_mask = bool(np.array_equal(active, ref_active))

    # Max |case - reference| per column over all cells; -999.25 equals -999.25, so it gives 0.
    max_diff = np.abs(arr - ref).max(axis=0)

    idx = [names.index(c) for c in VARYING_COLS]
    varying = arr[ref_active][:, idx].astype(np.float32)
    return {
        "case_id": case_id,
        "same_active_mask": same_mask,
        "n_active": int(active.sum()),
        "max_diff": max_diff,
        "varying": varying,
    }
