"""Train and validation arrays for Task 2 (spec task2 section 1.3, `load_dataset`).

The database is the single source (PRD section 5). Rows are selected by the
stored `split` column (V2, V5). Test rows are dropped before the Dataset is
built, so no test array or test case ID leaves this module (V3).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.db import read_task1_arrays
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.data.prepare import arrays_sha256
from our_work.proxy_model_sijie.src.dwc26.models.base import check_int, onset_index


@dataclass(frozen=True)
class Dataset:
    X_fit: np.ndarray          # (n_fit, p) scaled, split 'train'
    Y_fit: np.ndarray          # (n_fit, T_fit, 3) scaled
    X_val: np.ndarray          # (n_val, p)
    Y_val: np.ndarray          # (n_val, T_fit, 3)
    case_fit: np.ndarray       # (n_fit,) int64, ascending
    case_val: np.ndarray       # (n_val,) int64, ascending
    dates: np.ndarray          # (T_fit,) datetime64[D]
    t_days: np.ndarray         # (T_fit,)
    T_full: int                # T of the Task 1 arrays
    t_cut: int | None
    y_factor: np.ndarray       # (3,) STB/d, MSCF/d, STB/d
    x_lower: np.ndarray        # (p,)
    x_upper: np.ndarray        # (p,)
    x_names: tuple[str, ...]
    fluids: tuple[str, ...]
    prepare_run_id: str
    arrays_sha256: str

    @property
    def T_fit(self) -> int:
        return int(self.Y_fit.shape[1])

    @property
    def p(self) -> int:
        return int(self.X_fit.shape[1])


def _stored_sha(con: sqlite3.Connection, run_id: str) -> str | None:
    row = con.execute("SELECT outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        return json.loads(row[0]).get("arrays_sha256")
    except (json.JSONDecodeError, AttributeError):
        return None


def load_dataset(con: sqlite3.Connection, t_cut: int | None = None) -> Dataset:
    """Steps 1 to 6 of spec section 1.3."""
    a = read_task1_arrays(con)                              # 1. F2 guard
    run_id = str(a["run_id"])
    sha = arrays_sha256(a)                                  # 2. tables = the logged prepare run (V8)
    stored = _stored_sha(con, run_id)
    if sha != stored:
        raise ContractError(f"load_dataset: arrays_sha256 of the Task 1 tables ({sha[:12]}...) differs from "
                            f"runs.outputs.arrays_sha256 of prepare run {run_id} ({str(stored)[:12]}...); "
                            "rerun prepare")
    split = a["split"]                                      # 3. stored split column (V2, V5)
    fit, val = split == "train", split == "val"
    X_fit, Y_fit = a["X"][fit].copy(), a["Y"][fit].copy()
    X_val, Y_val = a["X"][val].copy(), a["Y"][val].copy()
    case_fit, case_val = a["case_id"][fit].astype(np.int64), a["case_id"][val].astype(np.int64)
    dates, t_days = a["dates"].copy(), a["t_days"].copy()
    T_full = int(a["Y"].shape[1])
    meta = dict(y_factor=a["y_factor"].copy(), x_lower=a["x_lower"].copy(), x_upper=a["x_upper"].copy(),
                x_names=tuple(str(s) for s in a["x_names"]), fluids=tuple(str(s) for s in a["fluids"]))
    del a, split                                            # 4. no test array or test ID is kept (V3)

    if len(case_fit) < 2 or len(case_val) < 1:
        raise ContractError(f"load_dataset: n_fit = {len(case_fit)} (>= 2) and n_val = {len(case_val)} (>= 1) needed")
    k0 = onset_index(Y_fit)
    if t_cut is not None:                                   # 5. truncated window (Task 5 backtest)
        t_cut = check_int(t_cut, "t_cut")
        if not k0 + 3 <= t_cut <= T_full:
            raise ContractError(f"load_dataset: t_cut = {t_cut} must lie in [k0 + 3, T] = [{k0 + 3}, {T_full}]")
        Y_fit, Y_val = Y_fit[:, :t_cut].copy(), Y_val[:, :t_cut].copy()
        dates, t_days = dates[:t_cut].copy(), t_days[:t_cut].copy()

    n_fit, n_val, T, p = len(case_fit), len(case_val), Y_fit.shape[1], X_fit.shape[1]   # 6. V9
    for name, X, n in (("X_fit", X_fit, n_fit), ("X_val", X_val, n_val)):
        assert_array(f"load_dataset {name}", X, (n, p))
        if (X < 0).any() or (X > 1).any():
            raise ContractError(f"load_dataset {name}: values outside [0, 1]")
    assert_array("load_dataset Y_fit", Y_fit, (n_fit, T, len(FLUIDS)), nonneg=True)
    assert_array("load_dataset Y_val", Y_val, (n_val, T, len(FLUIDS)), nonneg=True)
    assert_array("load_dataset dates", dates, (T,), dtype="datetime64[D]")
    assert_array("load_dataset t_days", t_days, (T,))
    if not (np.diff(dates.astype(np.int64)) > 0).all():
        raise ContractError("load_dataset: dates are not strictly increasing")
    for name, ids in (("case_fit", case_fit), ("case_val", case_val)):
        if not (np.diff(ids) > 0).all():
            raise ContractError(f"load_dataset: {name} is not strictly ascending")
    if set(case_fit.tolist()) & set(case_val.tolist()):
        raise ContractError("load_dataset: a case is in both train and val")
    if meta["fluids"] != FLUIDS:
        raise ContractError(f"load_dataset: fluids {meta['fluids']}, expected {FLUIDS}")
    return Dataset(X_fit=X_fit, Y_fit=Y_fit, X_val=X_val, Y_val=Y_val, case_fit=case_fit, case_val=case_val,
                   dates=dates, t_days=t_days, T_full=T_full, t_cut=t_cut, prepare_run_id=run_id,
                   arrays_sha256=sha, **meta)
